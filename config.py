"""
Configuration: API keys, paths, thresholds.
Copy .env.example to .env and fill in your keys.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "The Truman Show - train"   # default world; override via CLI
CHROMA_DIR = BASE_DIR / ".chromadb"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model (via OpenRouter) ─────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Any OpenRouter model slug — default: GPT-4o-mini (fast, cheap)
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-flash")

# Embedding model (no GPU needed on macOS with sentence-transformers)
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── RAG ────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "fraud_transactions"
RAG_TOP_K = 3

# ── Thresholds ─────────────────────────────────────────────────────────────
HIGH_VALUE_THRESHOLD = 10_000     # €
HIGH_VALUE_PROB_THRESHOLD = 0.30  # for high-value transactions
DEFAULT_PROB_THRESHOLD = 0.20     # catches the 0.35 phishing-floor transactions

# ── Langfuse ───────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
TEAM_NAME = os.getenv("TEAM_NAME", "your-team-name")

# ── Geographic ─────────────────────────────────────────────────────────────
GEO_ANOMALY_KM = 100   # distance threshold in km

# ── NLP ────────────────────────────────────────────────────────────────────
URGENT_WORDS = {
    "urgent", "password", "verify", "verification", "click", "immediately",
    "confirm", "suspend", "suspended", "alert", "warning", "action required",
    "limited time", "expire", "expired", "unauthorized", "compromised",
    "reset", "update your", "security", "prize", "winner", "won",
}

# Adaugă în config.py
ECONOMIC_PANIC_THRESHOLD = 5000  # Orice peste 5k este verificat dublu
SENSITIVITY_HIGH_VALUE = 0.2     # Scădem pragul de probabilitate pentru sume mari