"""
Phase 2: RAG & Caching Setup
==============================
- ChromaDB (local) for few-shot retrieval of similar historical transactions.
- Python dict cache keyed by hash(user_id + amount + location) to skip
  redundant LangGraph executions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL, RAG_TOP_K


# ── Embedding function ──────────────────────────────────────────────────────
# Use ChromaDB's built-in DefaultEmbeddingFunction which runs all-MiniLM-L6-v2
# via ONNX Runtime — no PyTorch dependency required.

_EMBED_FN = embedding_functions.DefaultEmbeddingFunction()


# ── ChromaDB client (persistent, local) ────────────────────────────────────

def _get_collection(collection_name: str = CHROMA_COLLECTION) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def _record_to_doc(record: dict[str, Any]) -> str:
    """Convert an enriched record to a short embeddable text."""
    return (
        f"type={record['transaction_type']} "
        f"amount={record['amount']:.2f} "
        f"method={record['payment_method']} "
        f"velocity24h={record['transaction_velocity_24h']} "
        f"z_score={record['amount_z_score']} "
        f"high_risk_loc={record['is_high_risk_location']} "
        f"night={record['is_night']} "
        f"weekend={record['is_weekend']} "
        f"phishing={record['recent_phishing_count']} "
        f"desc={record['description'][:60]}"
    )


# ── Public API ─────────────────────────────────────────────────────────────

def embed_dataset(records: list[dict[str, Any]], collection_name: str = CHROMA_COLLECTION) -> None:
    """
    Embed the full enriched dataset into ChromaDB using batching to avoid size limits.
    """
    col = _get_collection(collection_name)
    existing_ids: set[str] = set(col.get(include=[])["ids"])

    ids, docs, metadatas = [], [], []
    for rec in records:
        tid = rec["transaction_id"]
        if tid in existing_ids:
            continue
        ids.append(tid)
        docs.append(_record_to_doc(rec))
        metadatas.append({
            "transaction_id": tid,
            "user_id": rec["user_id"],
            "amount": rec["amount"],
            "amount_z_score": rec["amount_z_score"],
            "is_fraud": str(rec.get("is_fraud", "unknown")),
            "is_high_risk_location": str(rec["is_high_risk_location"]),
            "transaction_velocity_24h": rec["transaction_velocity_24h"],
            "recent_phishing_count": rec["recent_phishing_count"],
        })

    if ids:
        batch_size = 5000 
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i: i + batch_size]
            batch_docs = docs[i: i + batch_size]
            batch_metas = metadatas[i: i + batch_size]

            col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            print(f"[RAG] Added batch {i // batch_size + 1} ({len(batch_ids)} records).")

        print(f"[RAG] Finished embedding {len(ids)} new records into '{collection_name}'.")
    else:
        print(f"[RAG] All records already in '{collection_name}', skipping embed.")

def retrieve_context(
    transaction_data: dict[str, Any],
    collection_name: str = CHROMA_COLLECTION,
    k: int = RAG_TOP_K,
) -> str:
    """
    Query ChromaDB for the k most similar historical transactions.
    Returns a formatted string to be inserted into the LLM prompt.
    """
    col = _get_collection(collection_name)
    query_doc = _record_to_doc(transaction_data)

    results = col.query(
        query_texts=[query_doc],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    lines = ["=== Similar Historical Transactions ==="]
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        fraud_label = meta.get("is_fraud", "unknown")
        similarity = round(1 - dist, 3)  # cosine: distance 0 = identical
        lines.append(
            f"[{i}] similarity={similarity} fraud={fraud_label} | {doc}"
        )
    return "\n".join(lines)


def retrieve_context_for_user(
    user_ctx: dict[str, Any],
    collection_name: str = CHROMA_COLLECTION,
    k: int = RAG_TOP_K,
) -> str:
    """
    Query ChromaDB using the user's first transaction as the anchor.
    Returns similar historical cases as a string for the per-user prompt.
    """
    col = _get_collection(collection_name)
    if col.count() == 0:
        return "No historical cases available."

    # Use the user_id as query text
    query = f"user={user_ctx['user_id']} phishing=high night=possible"
    results = col.query(
        query_texts=[query],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    lines = ["Similar historical cases:"]
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        fraud_label = meta.get("is_fraud", "unknown")
        phishing    = meta.get("recent_phishing_count", "?")
        lines.append(f"  fraud={fraud_label} phishing={phishing} | {doc[:80]}")
    return "\n".join(lines)


# ── Local dict cache ────────────────────────────────────────────────────────

class RiskScoreCache:
    """
    In-memory dict cache.
    Key = SHA-256(user_id + amount + location_label)[:16]
    Value = risk_score (float 0-1)
    """

    def __init__(self) -> None:
        self._store: dict[str, float] = {}

    @staticmethod
    def _make_key(user_id: str, amount: float, location_label: str) -> str:
        raw = f"{user_id}|{amount:.4f}|{location_label}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, user_id: str, amount: float, location_label: str) -> float | None:
        return self._store.get(self._make_key(user_id, amount, location_label))

    def set(self, user_id: str, amount: float, location_label: str, score: float) -> None:
        self._store[self._make_key(user_id, amount, location_label)] = score

    def __len__(self) -> int:
        return len(self._store)


# ── Singleton cache ─────────────────────────────────────────────────────────
CACHE = RiskScoreCache()


if __name__ == "__main__":
    from phase1_preprocessing import preprocess
    records = preprocess("The Truman Show - train")
    embed_dataset(records)
    ctx = retrieve_context(records[0])
    print(ctx)
