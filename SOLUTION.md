# AGHR Fraud Detection — Solution Overview

## The Problem

We are given three fictional "worlds", each containing the financial life of a small number of users over roughly one year. Each world is a directory with five files:

| File | Contents |
|------|----------|
| `transactions.csv` | Bank transactions with sender/recipient IBANs, amounts, timestamps, descriptions |
| `users.json` | User profiles: name, salary, job, IBAN, home residence (city + lat/lng) |
| `locations.json` | GPS pings per user (biotag ID, timestamp, lat/lng, city) |
| `sms.json` | Raw SMS messages received by users |
| `mails.json` | Raw emails received by users |

There are **no ground-truth fraud labels**. The task is to identify which `transaction_id`s are fraudulent and output them one per line in a `.txt` file.

---

## Worlds

| World | Directory |
|-------|-----------|
| The Truman Show | `The Truman Show - train/` |
| Brave New World | `Brave New World - train/` |
| Deus Ex | `Deus Ex - train/` |

---

## Architecture

The solution is a 5-phase pipeline:

```
Raw Data
   │
   ▼
Phase 1: Preprocessing  ──►  Enriched flat records (risk indicators)
   │
   ▼
Phase 2: RAG + Cache    ──►  ChromaDB (similar past cases) + dict cache
   │
   ▼
Phase 3: LangGraph      ──►  orchestrator → combined_agent → END
   │                         (pre-filter short-circuits obvious non-fraud)
   ▼
Phase 4: Prompts        ──►  Single combined JSON prompt (1 LLM call/txn)
   │
   ▼
Phase 5: Threshold + Export  ──►  fraud_ids.txt + full_results.csv
```

---

## Phase 1 — Data Preprocessing (`phase1_preprocessing.py`)

Reads all five raw files and produces one **enriched flat dict per transaction**, stripped of raw timestamps and exact coordinates to save tokens.

### Computed risk indicators

| Field | How it is computed |
|-------|--------------------|
| `transaction_velocity_24h` | Count of transactions sent by the same user in the 24 h before this one |
| `amount_z_score` | `(amount − user_mean) / user_std` across the user's full history |
| `is_high_risk_location` | Haversine distance between user's nearest GPS ping (at transaction time) and home address > 500 km |
| `balance_ratio` | `amount / balance_after` — fraction of remaining balance spent |
| `hour_of_day` / `is_night` / `is_weekend` | Derived from transaction timestamp (timestamp itself is not forwarded) |
| `total_urgent_word_count` | Count of words like "urgent", "password", "verify", "prize" across all SMS + email |
| `any_message_contains_link` | Whether any SMS/email contains a URL |

### User–transaction linkage

Users are identified by their **IBAN** (`sender_iban` in the transactions CSV) and by a **biotag** slug (e.g. `RGNR-LNAA-7FF-AUD-0`) which links to GPS location records. Employer salary transfers (sender IDs like `EMP14947`) are not tracked users and receive zeroed profile fields.

---

## Phase 2 — RAG & Caching (`phase2_rag_cache.py`)

### ChromaDB (local, persistent)

- Embedding model: **all-MiniLM-L6-v2** via ChromaDB's built-in ONNX runtime — no PyTorch required.
- Each transaction is embedded as a short descriptor string: `type= amount= method= velocity24h= z_score= …`
- `retrieve_context(record)` returns the **3 most similar** historical transactions as a formatted string injected into the LLM prompt.
- Embedding is **idempotent** — re-running skips already-indexed records.

### Dict cache

- Key: `SHA-256(user_id | amount | location_label)[:16]`
- Value: previously computed `final_probability`
- A cache hit skips the LangGraph pipeline entirely and returns the stored score.

---

## Phase 3 — LangGraph State Machine (`phase3_langgraph.py`)

### State schema

```python
class AgentState(TypedDict):
    transaction_id: str
    raw_data: dict
    rag_context: str
    formatted_input: str
    financial_score: int        # 0-100
    identity_score: int         # 0-100
    nlp_score: int              # 0-100
    final_probability: float    # 0.0-1.0
    is_fraud: bool
    skipped: bool               # True when pre-filter short-circuited
```

### Graph

```
START → orchestrator → combined_agent → END
```

### Deterministic pre-filter (in `orchestrator`)

Before any LLM call, the orchestrator checks whether the transaction is **obviously safe**. All of the following must hold:

- `is_high_risk_location = False`
- `amount_z_score < 1.5`
- `transaction_velocity_24h ≤ 2`
- NOT (`is_night` AND `is_weekend`)
- Description starts with `"salary payment"`, `"rent payment"`, `"wage"`, or `"payroll"`

If all conditions pass → scores are set to 5/5/5 and probability to `0.03`. **No LLM call is made.**

On the Truman Show dataset this skips **86 %** of transactions (69 / 80).

### `combined_agent`

A single LLM call that returns all four values at once:

```json
{"financial_score": 72, "identity_score": 45, "nlp_score": 30, "probability": 0.61}
```

This replaced the original 4-call design (3 workers + meta-reviewer), reducing LLM calls by **~97 %** in total.

---

## Phase 4 — Prompts (`phase4_prompts.py`)

One combined system prompt instructs the model to act as three specialists simultaneously and output a single JSON line. Scoring rules are embedded directly in the system prompt so the model applies them deterministically without free-form reasoning.

```
FINANCIAL: z_score, velocity, balance_ratio, absolute amount
IDENTITY:  high_risk_location, night+weekend, velocity burst
NLP:       urgent word count, links, suspicious description keywords
PROBABILITY: weighted average (financial × 2), amount >10k guard
```

Output format enforced: `{"financial_score":<int>,"identity_score":<int>,"nlp_score":<int>,"probability":<float>}`

---

## Phase 5 — Thresholding, Langfuse & Export (`phase5_export.py`)

### Thresholding

```python
if amount > 10_000 and probability > 0.40:
    is_fraud = True
elif probability > 0.85:
    is_fraud = True
else:
    is_fraud = False
```

### Concurrency

Transactions are processed with `ThreadPoolExecutor(max_workers=5)` — 5 transactions run concurrently, each making 1 LLM call.

### Langfuse tracing

Every LLM-evaluated transaction is wrapped in `lf.start_as_current_observation(...)` with:

```json
{
  "team_name": "<TEAM_NAME from .env>",
  "session_id": "<run session id>",
  "transaction_id": "...",
  "amount": ...
}
```

> Note: Langfuse v4's `CallbackHandler` no longer accepts `session_id` or `tags` directly — metadata is the correct mechanism.

### Output files

| File | Contents |
|------|----------|
| `output/<world>_fraud_ids.txt` | One `transaction_id` per line (submission file) |
| `output/<world>_full_results.csv` | All scores + probability for every transaction |

---

## Setup & Usage

### 1. Environment

```bash
# Python 3.11 required (3.14 has no torch/onnx support yet)
pyenv local 3.11.10
python3 -m venv .venv && source .venv/bin/activate
pip install pandas chromadb langgraph langchain langchain-openai langfuse sentence-transformers anthropic
```

### 2. Credentials

```bash
cp .env.example .env
# Fill in:
#   OPENROUTER_API_KEY=sk-or-...
#   TEAM_NAME=your-team-name
#   LANGFUSE_PUBLIC_KEY=pk-lf-...   (optional)
#   LANGFUSE_SECRET_KEY=sk-lf-...   (optional)
```

### 3. Unzip worlds

```bash
unzip "Brave+New+World+-+train.zip"
unzip "Deus+Ex+-+train.zip"
# The Truman Show - train/ is already extracted
```

### 4. Run

```bash
# Single world
python main.py --data-dir "The Truman Show - train" --world-name truman-show --session-id run-001

# All three worlds
for world in "The Truman Show - train" "Brave New World - train" "Deus Ex - train"; do
  name=$(echo "$world" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
  python main.py --data-dir "$world" --world-name "$name" --session-id "run-001"
done

# Zip submission
python main.py --data-dir "The Truman Show - train" --world-name truman-show --zip
```

### 5. Incremental modes

```bash
python main.py --phase1-only   # preprocessing only, prints first record
python main.py --embed-only    # preprocessing + ChromaDB embedding, no LLM
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ONNX embeddings (no PyTorch) | Python 3.11 + NumPy 2.x breaks torch 2.2; ChromaDB's default ONNX function works out of the box |
| Single combined LLM call | 4 separate calls (3 workers + meta) cost ~4× more with no accuracy benefit on gpt-4o-mini |
| Deterministic pre-filter | Salary/rent payments with normal z-scores are not fraud; no need to spend tokens confirming it |
| `ThreadPoolExecutor` for parallelism | LangGraph's sync fan-out is sequential; explicit threads restore true parallelism |
| Global `urgent_word_count` excluded from pre-filter | It aggregates across all SMS/mail (not per-transaction), so it's always high and would block every transaction |
| Langfuse `start_as_current_observation` | v4 `CallbackHandler` no longer exposes `session_id`/`tags`; metadata is the supported path |

---

## File Structure

```
aghr/
├── config.py                  — API keys, model, thresholds, paths
├── phase1_preprocessing.py    — Risk indicator computation (pandas)
├── phase2_rag_cache.py        — ChromaDB RAG + dict cache
├── phase3_langgraph.py        — LangGraph state machine + pre-filter
├── phase4_prompts.py          — Combined single-call prompt
├── phase5_export.py           — Thresholding, Langfuse, file export, zip
├── main.py                    — CLI entry point
├── .env.example               — Credential template
├── output/
│   ├── <world>_fraud_ids.txt  — Submission file
│   └── <world>_full_results.csv
└── .chromadb/                 — Persistent ChromaDB vector store
```
