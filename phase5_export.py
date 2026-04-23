"""
Phase 5: Thresholding, Langfuse Tracking, & Export
=====================================================
Runs the LangGraph pipeline once per user (holistic analysis),
collects fraudulent transaction IDs, exports submission files.
"""

from __future__ import annotations

import csv
import threading
import zipfile
from pathlib import Path
from typing import Any

import ulid

from config import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    OUTPUT_DIR,
    TEAM_NAME,
)
from phase2_rag_cache import embed_dataset, retrieve_context_for_user
from phase3_langgraph import AgentState, build_graph


# ── Langfuse ────────────────────────────────────────────────────────────────

def generate_session_id() -> str:
    team = TEAM_NAME.replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def _get_langfuse_client():
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("[Langfuse] No credentials — tracing disabled.")
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception as exc:
        print(f"[Langfuse] Init failed: {exc}")
        return None


def _get_langfuse_handler():
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as exc:
        print(f"[Langfuse] Handler failed: {exc}")
        return None


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    per_txn_records: list[dict[str, Any]],   # from preprocess() — for RAG
    user_contexts:   list[dict[str, Any]],   # from build_user_contexts()
    world_name: str = "truman-show",
) -> list[dict[str, Any]]:
    """
    For each user:
      1. Retrieve RAG context
      2. Run LangGraph (orchestrator → 3 parallel agents → meta-reviewer)
      3. Collect fraudulent transaction IDs

    Returns list of result dicts:
      {transaction_id, is_fraud, user_id, flagged_by}
    """
    # Embed per-transaction records for RAG
    embed_dataset(per_txn_records)

    session_id = generate_session_id()
    print(f"[Langfuse] team={TEAM_NAME}  session={session_id}")

    lf_client  = _get_langfuse_client()
    lf_handler = _get_langfuse_handler()

    graph = build_graph()

    all_fraud_ids: set[str] = set()
    per_user_results: list[dict] = []

    total = len(user_contexts)
    for i, uctx in enumerate(user_contexts, 1):
        user_id    = uctx["user_id"]
        first_name = uctx["first_name"]
        print(f"\n[{i}/{total}] Analysing user: {first_name} ({user_id})")

        rag_ctx = retrieve_context_for_user(uctx)

        initial: AgentState = {
            "user_id":         user_id,
            "context":         uctx["context"],
            "rag_context":     rag_ctx,
            "financial_flags": [],
            "identity_flags":  [],
            "nlp_flags":       [],
            "fraudulent_ids":  [],
        }

        invoke_config: dict = {}
        if lf_handler:
            invoke_config = {
                "callbacks": [lf_handler],
                "metadata":  {"langfuse_session_id": session_id},
            }

        if lf_client and invoke_config:
            with lf_client.start_as_current_observation(
                name=f"fraud-user-{first_name}",
                as_type="agent",
                metadata={"team_name": TEAM_NAME, "session_id": session_id, "user": first_name},
            ):
                output = graph.invoke(initial, config=invoke_config)
        else:
            output = graph.invoke(initial)

        fraud_ids = output.get("fraudulent_ids", [])
        all_fraud_ids.update(fraud_ids)
        per_user_results.append({
            "user_id":       user_id,
            "fraudulent_ids": fraud_ids,
            "financial_flags": output.get("financial_flags", []),
            "identity_flags":  output.get("identity_flags",  []),
            "nlp_flags":       output.get("nlp_flags",       []),
        })
        print(f"  → Fraud flagged for {first_name}: {fraud_ids}")

    if lf_client:
        lf_client.flush()

    # Build flat result list preserving all transaction IDs
    all_txn_ids = [r["transaction_id"] for r in per_txn_records]
    results: list[dict[str, Any]] = []
    for txn_id in all_txn_ids:
        results.append({
            "transaction_id": txn_id,
            "is_fraud":       txn_id in all_fraud_ids,
        })

    fraud_total = sum(r["is_fraud"] for r in results)
    print(f"\nDone. Fraud flagged: {fraud_total}/{len(results)}")
    return results


# ── Export ───────────────────────────────────────────────────────────────────

def export_fraud_ids(results: list[dict[str, Any]], world_name: str = "truman-show") -> Path:
    out_file = OUTPUT_DIR / f"{world_name}_fraud_ids.txt"
    flagged  = [r["transaction_id"] for r in results if r["is_fraud"]]
    out_file.write_text("\n".join(flagged) + ("\n" if flagged else ""))
    print(f"[Export] {len(flagged)} fraud IDs → {out_file}")
    return out_file


def export_full_results(results: list[dict[str, Any]], world_name: str = "truman-show") -> Path:
    out_file = OUTPUT_DIR / f"{world_name}_full_results.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transaction_id", "is_fraud"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Export] Full results → {out_file}")
    return out_file


def zip_working_directory(base_dir: str | Path = ".", zip_name: str = "submission.zip") -> Path:
    base_dir = Path(base_dir).resolve()
    out_zip  = base_dir / zip_name
    EXCLUDE  = {".venv", ".chromadb", "__pycache__", ".git", ".DS_Store"}
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(base_dir.rglob("*")):
            if any(part in EXCLUDE for part in path.parts):
                continue
            if path == out_zip or not path.is_file():
                continue
            zf.write(path, path.relative_to(base_dir))
    print(f"[Zip] {out_zip} ({out_zip.stat().st_size // 1024} KB)")
    return out_zip


if __name__ == "__main__":
    from phase1_preprocessing import build_user_contexts, preprocess
    records  = preprocess("The Truman Show - train")
    contexts = build_user_contexts("The Truman Show - train")
    results  = run_pipeline(records, contexts, world_name="truman-show")
    export_fraud_ids(results, world_name="truman-show")
    export_full_results(results, world_name="truman-show")
