"""
main.py — Entry point for the AGHR fraud detection pipeline.

Usage:
    python main.py [--data-dir PATH] [--world-name NAME] [--zip]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AGHR Fraud Detection Pipeline")
    p.add_argument("--data-dir",    default="The Truman Show - train")
    p.add_argument("--world-name",  default="truman-show")
    p.add_argument("--zip",         action="store_true")
    p.add_argument("--phase1-only", action="store_true")
    p.add_argument("--embed-only",  action="store_true")
    return p.parse_args()


def main() -> int:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    # ── Phase 1b (Audio) ──────────────────────────────────────────────────
    from phase1b_audio import process_audio_directory
    process_audio_directory(data_dir)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Preprocessing\n{'='*60}")
    from phase1_preprocessing import build_user_contexts, preprocess
    records  = preprocess(data_dir)
    contexts = build_user_contexts(data_dir)
    print(f"Preprocessed {len(records)} transactions across {len(contexts)} users.")

    if args.phase1_only:
        for ctx in contexts:
            print(ctx["context"])
        return 0

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Embedding into ChromaDB\n{'='*60}")
    from phase2_rag_cache import embed_dataset
    embed_dataset(records)

    if args.embed_only:
        return 0

    # ── Phases 3–5 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhases 3–5: LangGraph + Export\n{'='*60}")
    from phase5_export import export_fraud_ids, export_full_results, run_pipeline
    results = run_pipeline(records, contexts, world_name=args.world_name)
    export_fraud_ids(results,    world_name=args.world_name)
    export_full_results(results, world_name=args.world_name)

    if args.zip:
        from phase5_export import zip_working_directory
        zip_working_directory(zip_name=f"{args.world_name}_submission.zip")

    return 0


if __name__ == "__main__":
    sys.exit(main())