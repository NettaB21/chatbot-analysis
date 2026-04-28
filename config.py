# config.py  —  Central settings for the chatbot analysis pipeline.

# Google Sheet ID (from the sheet URL)
GOOGLE_SHEET_ID = "1FISvmBFsrhh1ggr8cW07qcT55LEHZxoGhDxSXwkvxlM"

# The exact tab name inside the sheet
GOOGLE_SHEET_TAB = "FR - Q&A Responses"

# Folder where analysis results are saved
OUTPUT_FOLDER = "outputs"

# Your local timezone
TIMEZONE = "America/Los_Angeles"

# Which Claude model to use
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# Conversations per API batch
BATCH_SIZE = 5
