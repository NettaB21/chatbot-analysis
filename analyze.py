# ─────────────────────────────────────────────────────────────────────────────
# analyze.py  —  Main analysis script.
#
# What this does, step by step:
#   1. Reads the CSV and keeps only last week's conversations (Mon 9am → Mon 9am)
#   2. Groups each conversation's Q&A pairs into a readable transcript
#   3. Sends batches of 5 transcripts to Claude for classification
#   4. Saves results as a CSV, a JSON metrics file, and a Markdown summary
#
# Run manually with:  python analyze.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import csv
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import anthropic
from dotenv import load_dotenv

# Load the ANTHROPIC_API_KEY from the .env file
load_dotenv()
import config

# Set up logging so progress messages appear in the terminal with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Step 1: Figure out the date window ──────────────────────────────────────

def get_weekly_window():
    """
    Returns (start, end) representing last week's window:
        start = previous Monday at 09:00 local time
        end   = this Monday at 09:00 local time
    """
    tz = ZoneInfo(config.TIMEZONE)
    now = datetime.now(tz)

    # Move back to the most recent Monday, then set time to 09:00:00
    current_monday = (now - timedelta(days=now.weekday())).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    previous_monday = current_monday - timedelta(weeks=1)

    return previous_monday, current_monday


# ─── Step 2: Load, filter, and group conversations ───────────────────────────

def load_conversations():
    """
    Reads the CSV, keeps only rows within the weekly window, and groups them
    into full conversation transcripts — one string per conversation ID.

    Returns a list of dicts: [{"id": "...", "text": "full transcript"}, ...]
    """
    start, end = get_weekly_window()
    log.info(f"Analyzing window: {start.strftime('%a %b %d %Y %H:%M')} → {end.strftime('%a %b %d %Y %H:%M')} ({config.TIMEZONE})")

    # Read the CSV into a pandas DataFrame (a spreadsheet-like table in memory)
    import gspread
    from google.oauth2 import service_account
    import json, os
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(config.GOOGLE_SHEET_ID)
    worksheet = sh.worksheet(config.GOOGLE_SHEET_TAB)
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)

    # Parse the 'created_at' column as real timestamps.
    # format='ISO8601' handles rows with or without milliseconds gracefully.
    # Timestamps have no timezone info, so we mark them UTC (standard for DB exports)
    # then convert to local time for filtering.
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True).dt.tz_convert(config.TIMEZONE)

    # Keep only rows that fall inside the weekly window
    df = df[(df["created_at"] >= start) & (df["created_at"] < end)].copy()

    if df.empty:
        return [], start, end

    # Group rows by conversation ID, sort each group by time,
    # then concatenate question/answer pairs into one readable transcript
    conversations = []
    for conv_id, group in df.groupby("id"):
        group = group.sort_values("created_at")
        lines = [f"--- Conversation ID: {conv_id} ---"]
        for _, row in group.iterrows():
            question = str(row.get("question", "") or "").strip()
            answer   = str(row.get("answer", "")   or "").strip()
            if question:
                lines.append(f"[user]: {question}")
            if answer:
                lines.append(f"[assistant]: {answer}")
        conversations.append({"id": conv_id, "text": "\n".join(lines)})

    log.info(f"Found {len(conversations)} conversations to classify")
    return conversations, start, end


# ─── Step 3: Classification prompt sent to Claude ────────────────────────────

# This is the exact instruction we send to Claude for each batch of conversations.
# The double curly braces {{ }} are how Python includes a literal { } in a template string.
CLASSIFICATION_PROMPT = """\
You are analyzing conversations from an immigration services chatbot.
Classify each conversation below on all dimensions listed. Return ONLY a valid JSON array \
— one object per conversation, in the same order. No explanation, no markdown, no extra text.

Conversations:
{batch_text}

Each object in the array must follow this exact structure:
{{
  "conversation_id": "the id field from the conversation header",
  "objection_topic": "string or null",
  "legal_risk_flag": "yes or no",
  "legal_risk_type": "one of: incorrect immigration information | overpromising outcomes | legal advice risk | misleading process guidance | insufficient qualification of answer | other | null",
  "question_type": "one of: general informational | specific to user situation | unclear",
  "trend_theme": "3-5 word summary of main topic",
  "product_topic": "immigration product or service mentioned, or null",
  "existing_product_or_status_check": "yes or no",
  "customer_type": "one of: likely existing customer | likely new prospect | unclear",
  "human_in_loop_value": "yes or no",
  "human_in_loop_category": "one of: sales close opportunity | support rescue opportunity | legal escalation needed | trust recovery | complex/confusing case | null",
  "conversion_blocker": "one of: pricing concerns | eligibility doubt | trust concerns | process confusion | timeline concerns | technical issues | support gaps | null",
  "urgency_level": "one of: low | medium | high",
  "language_detected": "ISO 639-1 code e.g. en es fr",
  "bot_language_match": "yes or no",
  "notable_notes": "string or null"
}}

Classification rules:
- Use null when not confident. Never guess.
- legal_risk_flag = yes if the bot gave incorrect info, overpromised, or gave what could be construed as legal advice
- existing_product_or_status_check = yes if user mentions an existing case, purchase, or account
- customer_type = likely existing customer only with clear evidence. Default to unclear
- bot_language_match = no if bot responded in a different language than the user\
"""


# ─── Step 4: Send a batch to Claude and get back classifications ──────────────

def classify_batch(client, conversations):
    """
    Sends up to BATCH_SIZE conversations to Claude and returns a list of
    classification dicts. Tries twice if the first attempt fails.
    """
    batch_text = "\n\n".join(c["text"] for c in conversations)
    prompt = CLASSIFICATION_PROMPT.format(batch_text=batch_text)

    for attempt in range(1, 3):  # try at most twice
        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Claude sometimes wraps JSON in ```json ... ``` — strip that out
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            results = json.loads(raw)
            if isinstance(results, list):
                return results

            log.warning(f"Attempt {attempt}: Response was not a JSON array — retrying")

        except json.JSONDecodeError as e:
            log.warning(f"Attempt {attempt}: Could not parse JSON response — {e}")
        except Exception as e:
            log.warning(f"Attempt {attempt}: API error — {e}")

    log.error("Batch failed after 2 attempts — skipping this batch")
    return []


# ─── Step 5: Validate each classification has all expected fields ─────────────

REQUIRED_FIELDS = [
    "conversation_id", "objection_topic", "legal_risk_flag", "legal_risk_type",
    "question_type", "trend_theme", "product_topic", "existing_product_or_status_check",
    "customer_type", "human_in_loop_value", "human_in_loop_category",
    "conversion_blocker", "urgency_level", "language_detected",
    "bot_language_match", "notable_notes",
]

def validate(obj):
    """Fill any fields that Claude may have accidentally omitted with None."""
    for field in REQUIRED_FIELDS:
        obj.setdefault(field, None)
    return obj


# ─── Step 6: Aggregate counts and percentages across all conversations ────────

def pct_breakdown(values):
    """Returns {value: {count, pct}} for every distinct value in the list."""
    total = len(values)
    if total == 0:
        return {}
    counts = {}
    for v in values:
        label = str(v) if v not in (None, "None", "null") else "not flagged"
        counts[label] = counts.get(label, 0) + 1
    return {
        k: {"count": v, "pct": round(100 * v / total, 1)}
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    }

def top5_counts(values):
    """Returns the top 5 non-null values and their counts, sorted by frequency."""
    counts = {}
    for v in values:
        if v and v not in ("None", "null", "unclear"):
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:5])

def aggregate_metrics(classified):
    """Builds the full metrics dictionary from all classified conversations."""
    n = len(classified)

    def vals(field):
        return [c.get(field) for c in classified]

    return {
        "total_conversations": n,
        "breakdowns": {
            "question_type":                    pct_breakdown(vals("question_type")),
            "customer_type":                    pct_breakdown(vals("customer_type")),
            "legal_risk_flag":                  pct_breakdown(vals("legal_risk_flag")),
            "human_in_loop_value":              pct_breakdown(vals("human_in_loop_value")),
            "existing_product_or_status_check": pct_breakdown(vals("existing_product_or_status_check")),
            "bot_language_match":               pct_breakdown(vals("bot_language_match")),
            "urgency_level":                    pct_breakdown(vals("urgency_level")),
        },
        "top5": {
            "objection_topic":        top5_counts(vals("objection_topic")),
            "product_topic":          top5_counts(vals("product_topic")),
            "trend_theme":            top5_counts(vals("trend_theme")),
            "conversion_blocker":     top5_counts(vals("conversion_blocker")),
            "human_in_loop_category": top5_counts(vals("human_in_loop_category")),
            "legal_risk_type":        top5_counts(vals("legal_risk_type")),
        },
        "language_distribution": pct_breakdown(vals("language_detected")),
    }


# ─── Step 7: Build data-driven recommended actions ────────────────────────────

def build_recommendations(metrics):
    """
    Generates 5 recommended actions based on what the data actually shows.
    These are tailored to the highest-impact issues found this week.
    """
    b = metrics["breakdowns"]
    t = metrics["top5"]
    recs = []

    # Recommendation based on legal risk
    legal_yes = b.get("legal_risk_flag", {}).get("yes", {})
    if legal_yes.get("count", 0) > 0:
        top_risk = next(iter(t["legal_risk_type"]), "unknown risk type")
        recs.append(
            f"**Review {legal_yes['count']} legal-risk conversations** "
            f"({legal_yes['pct']}% of total) — top risk type: _{top_risk}_"
        )

    # Recommendation based on human-in-loop opportunities
    hil = b.get("human_in_loop_value", {}).get("yes", {})
    if hil.get("count", 0) > 0:
        top_hil = next(iter(t["human_in_loop_category"]), "general")
        recs.append(
            f"**Follow up on {hil['count']} human-in-loop opportunities** "
            f"({hil['pct']}% of conversations) — top category: _{top_hil}_"
        )

    # Recommendation based on language mismatches
    lang_no = b.get("bot_language_match", {}).get("no", {})
    if lang_no.get("count", 0) > 0:
        recs.append(
            f"**Fix {lang_no['count']} bot language mismatches** — "
            f"users asked in one language and got a response in another"
        )

    # Recommendation based on top conversion blocker
    top_blocker = next(iter(t["conversion_blocker"]), None)
    if top_blocker:
        recs.append(
            f"**Address top conversion blocker: _{top_blocker}_** — "
            f"update bot responses and FAQs to resolve this friction"
        )

    # Recommendation based on top objection theme
    top_objection = next(iter(t["objection_topic"]), None)
    if top_objection:
        recs.append(
            f"**Prepare improved responses for top objection: _{top_objection}_** — "
            f"train the bot or add a human handoff trigger for this topic"
        )

    # Fill up to 5 with generic high-value recommendations if needed
    fallbacks = [
        "Share this report with the product team and align on the top 3 bot improvements",
        "Update FAQ and bot knowledge base based on top trend themes this week",
        "Review bot responses for the most frequently asked question types",
    ]
    for rec in fallbacks:
        if len(recs) >= 5:
            break
        recs.append(rec)

    return recs[:5]


# ─── Step 8: Write the three output files ────────────────────────────────────

def write_outputs(classified, metrics, start, end):
    """
    Saves three files inside outputs/YYYY-MM-DD/:
      - classified_conversations.csv  (one row per conversation)
      - metrics.json                  (all counts and percentages)
      - weekly_summary.md             (executive summary in plain language)
    """
    date_str = start.strftime("%Y-%m-%d")
    out_dir = Path(config.OUTPUT_FOLDER) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── classified_conversations.csv ──
    csv_path = out_dir / "classified_conversations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(classified)
    log.info(f"Saved → {csv_path}")

    # ── metrics.json ──
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"Saved → {json_path}")

    # ── weekly_summary.md ──
    b = metrics["breakdowns"]
    t = metrics["top5"]
    n = metrics["total_conversations"]
    recs = build_recommendations(metrics)

    def fmt_top5(d):
        if not d:
            return "  - _(none this week)_\n"
        return "".join(f"  - **{k}** — {v}\n" for k, v in d.items())

    def fmt_pct(d):
        return "".join(f"  - {k}: {v['count']} ({v['pct']}%)\n" for k, v in d.items())

    summary = f"""# Weekly Chatbot Analysis Summary

**Period:** {start.strftime("%B %d, %Y")} – {end.strftime("%B %d, %Y")}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## Total Conversations Analyzed
**{n} conversations**

---

## Top 5 Objection Themes
{fmt_top5(t['objection_topic'])}
## Top 5 Products Asked About
{fmt_top5(t['product_topic'])}
## Question Type Breakdown
{fmt_pct(b['question_type'])}
## Customer Type Breakdown
{fmt_pct(b['customer_type'])}
## Existing Product / Status Check Conversations
{fmt_pct(b['existing_product_or_status_check'])}
## Human-in-Loop Opportunities
{fmt_pct(b['human_in_loop_value'])}
**Top human-in-loop categories:**
{fmt_top5(t['human_in_loop_category'])}
## Language Distribution
{fmt_pct(metrics['language_distribution'])}
## Legal / Compliance Risks
{fmt_pct(b['legal_risk_flag'])}
**Top risk types:**
{fmt_top5(t['legal_risk_type'])}
## Top 5 Trend Themes
{fmt_top5(t['trend_theme'])}
## Top 5 Recommended Actions

{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs))}

---
_Generated by chatbot-analysis pipeline_
"""

    md_path = out_dir / "weekly_summary.md"
    md_path.write_text(summary, encoding="utf-8")
    log.info(f"Saved → {md_path}")

    return out_dir


# ─── Google Drive upload ──────────────────────────────────────────────────────

def upload_to_google_drive(file_path, folder_id, credentials_file):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2 import service_account

    if not credentials_file or not os.path.exists(credentials_file):
        print("No credentials file found — skipping Google Drive upload")
        return
    if not folder_id:
        print("No GOOGLE_DRIVE_FOLDER_ID set — skipping Google Drive upload")
        return

    creds = service_account.Credentials.from_service_account_file(
        credentials_file,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    file_name = f"Chatbot Weekly Analysis — {datetime.now().strftime('%b %d %Y')}"
    file_metadata = {"name": file_name, "parents": [folder_id], "mimeType": "application/vnd.google-apps.document"}
    media = MediaFileUpload(file_path, mimetype="text/html")
    service.files().create(body=file_metadata, media_body=media).execute()
    print(f"Report uploaded to Google Drive: {file_name}")


# ─── Slack notification ───────────────────────────────────────────────────────

def send_slack_notification(summary_path, webhook_url):
    import urllib.request
    import json
    if not webhook_url:
        print("No SLACK_WEBHOOK_URL set — skipping Slack notification")
        return
    with open(summary_path, 'r') as f:
        summary = f.read()
    # Send headline message
    headline = summary[:2000]
    payload = json.dumps({"text": f"*Weekly Chatbot Analysis*\n{headline}"})
    req = urllib.request.Request(webhook_url, payload.encode(), {'Content-Type': 'application/json'})
    urllib.request.urlopen(req)
    print("Slack notification sent")


# ─── Main entry point ─────────────────────────────────────────────────────────

def main(limit=None):
    """
    Runs the full analysis pipeline.
    Pass limit=10 to process only the first 10 conversations (useful for testing).
    """

    # Guard: make sure the API key is available before doing anything else
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY not found.\n"
            "Create a file named .env in this folder and add:\n"
            "  ANTHROPIC_API_KEY=your-key-here"
        )

    conversations, start, end = load_conversations()

    # If the weekly window has no data, write a minimal summary and stop
    if not conversations:
        out_dir = Path(config.OUTPUT_FOLDER) / start.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        msg = (
            f"# Weekly Chatbot Analysis Summary\n\n"
            f"No conversations found for {start.strftime('%B %d')} – {end.strftime('%B %d, %Y')}.\n\n"
            f"Check that your CSV file covers this date range and that SOURCE_FILE in config.py is correct.\n"
        )
        (out_dir / "weekly_summary.md").write_text(msg, encoding="utf-8")
        log.info(f"No data found — wrote empty summary to {out_dir}/")
        return

    # Apply optional limit (used for test runs)
    if limit:
        log.info(f"Test mode: processing only the first {limit} conversations")
        conversations = conversations[:limit]

    # Connect to the Anthropic API
    client = anthropic.Anthropic()
    classified = []

    # Process conversations in batches to reduce API calls
    total_batches = (len(conversations) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    for i in range(0, len(conversations), config.BATCH_SIZE):
        batch_num = i // config.BATCH_SIZE + 1
        batch = conversations[i : i + config.BATCH_SIZE]
        log.info(f"Batch {batch_num}/{total_batches} — classifying {len(batch)} conversations...")
        results = classify_batch(client, batch)
        classified.extend(validate(r) for r in results)

    if not classified:
        sys.exit(
            "ERROR: No classifications were produced.\n"
            "Check your ANTHROPIC_API_KEY and ANTHROPIC_MODEL in config.py."
        )

    metrics = aggregate_metrics(classified)
    out_dir = write_outputs(classified, metrics, start, end)
    log.info(f"✓ Analysis complete — results saved to {out_dir}/")

    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    send_slack_notification(f"{out_dir}/weekly_summary.md", webhook_url)

    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    try:
        upload_to_google_drive(f"{out_dir}/weekly_summary.md", folder_id, credentials_file)
    except Exception as e:
        log.warning(f"Google Drive upload failed: {e}")


if __name__ == "__main__":
    # To run a quick test on 10 conversations: python analyze.py --test
    import sys
    test_mode = "--test" in sys.argv
    main(limit=10 if test_mode else None)
