"""
config.py
Loads API keys from the .env file.

Your folder structure:
helloo_there/
├── .env                      <-- api_key=..., LLAMA_PARSE_KEY=...
└── rag-chat-bot/
    ├── config.py             <-- this file
    ├── rag_chatbot.py
    └── ...

Because .env sits ONE FOLDER ABOVE rag-chat-bot, we point dotenv to "../.env".
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent           # .../helloo_there/rag-chat-bot/rag_chatbot
ENV_PATH = BASE_DIR.parent.parent / ".env"           # .../helloo_there/.env

load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("api_key")
LLAMA_PARSE_KEY = os.getenv("LLAMA_PARSE_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"'api_key' not found. Checked .env at: {ENV_PATH}. "
        "Make sure your .env has a line like: api_key=YOUR_GEMINI_KEY"
    )

if not LLAMA_PARSE_KEY:
    raise ValueError(
        f"'LLAMA_PARSE_KEY' not found. Checked .env at: {ENV_PATH}. "
        "Make sure your .env has a line like: LLAMA_PARSE_KEY=YOUR_LLAMA_KEY"
    )