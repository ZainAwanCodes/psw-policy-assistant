"""
Configuration module for PSW Policy Assistant.
Centralizes all paths, model identifiers, chunking parameters, and runtime settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
INDEX_DIR = DATA_DIR / "index"

# Load .env file if present
env_file = BASE_DIR / ".env"
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=False)
    except Exception:
        # Fallback naive .env reader
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip().strip("'\"")

# Ensure runtime directories exist
INDEX_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

# Expected four PSW policy documents
POLICY_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "general_code_of_conduct": {
        "title": "General Code of Conduct",
        "primary_filename": "general_code_of_conduct.pdf",
        "alternate_filenames": ["General-Code-of-Conduct.pdf", "general-code-of-conduct.pdf"],
    },
    "whistle_blowing_policy": {
        "title": "Whistle Blowing Policy",
        "primary_filename": "whistle_blowing_policy.pdf",
        "alternate_filenames": ["Whistleblowing-Policy-website.pdf", "whistleblowing-policy.pdf", "whistle_blowing.pdf"],
    },
    "anti_corruption_policy": {
        "title": "Anti Corruption Policy",
        "primary_filename": "anti_corruption_policy.pdf",
        "alternate_filenames": ["Anticorruption-Policy.pdf", "anti-corruption-policy.pdf", "anticorruption.pdf"],
    },
    "harassment_policy": {
        "title": "Harassment Policy",
        "primary_filename": "harassment_policy.pdf",
        "alternate_filenames": ["Harrasment-Policy.pdf", "harassment-policy.pdf", "harrasment_policy.pdf"],
    },
}

# Ingestion & Chunking
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
APPROX_CHARS_PER_TOKEN = 4

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# Vector Store Persistence
FAISS_INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "chunks_metadata.json"
MANIFEST_FILE = INDEX_DIR / "index_manifest.json"

# Retrieval
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.0"))

# LLM Generation
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_GEMINI_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1500"))

# Conversation
CONVERSATION_HISTORY_WINDOW = int(os.getenv("CONVERSATION_HISTORY_WINDOW", "6"))


def get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieves the Gemini API key from environment variable or Streamlit secrets.
    Never raises an unhandled exception or logs the key.
    """
    # 1. Check environment variable
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            sec_key = st.secrets["GEMINI_API_KEY"]
            if isinstance(sec_key, str) and sec_key.strip():
                return sec_key.strip()
    except Exception:
        pass

    return None
