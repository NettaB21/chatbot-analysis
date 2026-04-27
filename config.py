# ─────────────────────────────────────────────────────────────────────────────
# config.py  —  Central settings for the chatbot analysis pipeline.
#
# This is the ONLY file you need to edit to change how the analysis works.
# ─────────────────────────────────────────────────────────────────────────────

# Path to your exported CSV file (relative to where you run the script)
# Update this whenever you drop a new CSV export in the data/ folder
SOURCE_FILE = "data/Chat_Response - FR - Q&A Responses.csv"

# Folder where analysis results are saved
# A new sub-folder named YYYY-MM-DD will be created automatically each run
OUTPUT_FOLDER = "outputs"

# Your local timezone — used to define the Mon 9am → Mon 9am analysis window
# Full list of valid values: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
TIMEZONE = "America/Los_Angeles"

# Which Claude AI model to use for conversation classification
# You can update this to a newer model if Anthropic releases one
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# How many conversations to send to Claude in a single API call
# Keeping this at 5 balances speed and reliability
BATCH_SIZE = 5
