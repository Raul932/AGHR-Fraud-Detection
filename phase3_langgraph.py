"""
Phase 3: LangGraph Multi-Agent State Machine — per-user holistic analysis
==========================================================================
Graph per user:
  START → orchestrator → [financial_agent, identity_agent, nlp_agent] (parallel)
                       → meta_reviewer → END

Each agent sees the FULL user context (all transactions + phishing messages).
Output: list of fraudulent transaction IDs for that user.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from phase4_prompts import (
    FINANCIAL_SYSTEM, FINANCIAL_USER,
    IDENTITY_SYSTEM,  IDENTITY_USER,
    META_SYSTEM,      META_USER,
    NLP_SYSTEM,       NLP_USER,
)


# ── State schema ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_id:          str
    context:          str          # full formatted user context
    rag_context:      str
    financial_flags:  list[str]    # IDs flagged by financial agent
    identity_flags:   list[str]    # IDs flagged by identity agent
    nlp_flags:        list[str]    # IDs flagged by NLP agent
    fraudulent_ids:   list[str]    # final output from meta-reviewer


# ── LLM factory ────────────────────────────────────────────────────────────

def _make_llm(callbacks=None) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_tokens=512,
        callbacks=callbacks or [],
    )


# ── JSON helpers ────────────────────────────────────────────────────────────

def _parse_ids(text: str, key: str) -> list[str]:
    """Extract a list of IDs from a JSON response."""
    try:
        data = json.loads(text.strip())
        return [str(i) for i in data.get(key, [])]
    except json.JSONDecodeError:
        pass
    # Fallback: find the list inside the text
    match = re.search(r'"' + key + r'"\s*:\s*(\[[^\]]*\])', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return []


def _call_llm(system: str, user: str, callbacks=None) -> str:
    llm = _make_llm(callbacks)
    return llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]).content


# ── Nodes ────────────────────────────────────────────────────────────────────

def orchestrator_node(state: AgentState, callbacks=None) -> dict:
    """Pass-through — context was assembled in phase1."""
    return {}


def _run_workers_parallel(state: AgentState, callbacks=None) -> dict:
    """
    Fire all three specialist agents simultaneously using threads.
    Each returns a list of suspicious transaction IDs.
    """
    ctx = state["context"]
    rag = state["rag_context"]

    tasks = {
        "financial": (FINANCIAL_SYSTEM, FINANCIAL_USER.format(context=ctx, rag_context=rag)),
        "identity":  (IDENTITY_SYSTEM,  IDENTITY_USER.format(context=ctx,  rag_context=rag)),
        "nlp":       (NLP_SYSTEM,       NLP_USER.format(context=ctx,       rag_context=rag)),
    }

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_llm, sys_p, usr_p, callbacks): name
            for name, (sys_p, usr_p) in tasks.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            raw  = future.result()
            results[name] = _parse_ids(raw, "flagged_ids")

    print(f"  [financial] {results['financial']}")
    print(f"  [identity]  {results['identity']}")
    print(f"  [nlp]       {results['nlp']}")

    return {
        "financial_flags": results["financial"],
        "identity_flags":  results["identity"],
        "nlp_flags":       results["nlp"],
    }


def meta_reviewer_node(state: AgentState, callbacks=None) -> dict:
    import re
    # [NEW] The Regex Dragnet: forcefully catch IDs that triggered critical flags
    critical_rule_ids = re.findall(r"\[([a-f0-9\-]{36})\].*?(?:BURST|GEO-ANOMALY)⚠️", state["context"])

    user_msg = META_USER.format(
        financial_flags=state["financial_flags"],
        identity_flags=state["identity_flags"],
        nlp_flags=state["nlp_flags"],
        context=state["context"],
    )
    raw = _call_llm(META_SYSTEM, user_msg, callbacks)
    ids = _parse_ids(raw, "fraudulent_ids")
    
    all_flags = state["financial_flags"] + state["identity_flags"] + state["nlp_flags"]
    consensus_ids = [fid for fid in set(all_flags) if all_flags.count(fid) >= 2]
    
    # THE ULTIMATE UNION: LLM picks + Consensus + Phishing + Hard Rules
    final_ids = list(set(ids) | set(consensus_ids) | set(state["nlp_flags"]) | set(critical_rule_ids))
    print(f"  [meta] Evaluated: LLM={len(ids)}, Consensus={len(consensus_ids)}, Dragnet={len(critical_rule_ids)} -> Total Union: {len(final_ids)}")
    
    return {"fraudulent_ids": final_ids}


# ── Graph factory ────────────────────────────────────────────────────────────

def build_graph(callbacks=None):
    def _orch(state):    return orchestrator_node(state, callbacks)
    def _workers(state): return _run_workers_parallel(state, callbacks)
    def _meta(state):    return meta_reviewer_node(state, callbacks)

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator",   _orch)
    builder.add_node("workers",        _workers)   # parallel financial+identity+nlp
    builder.add_node("meta_reviewer",  _meta)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "workers")
    builder.add_edge("workers", "meta_reviewer")
    builder.add_edge("meta_reviewer", END)

    return builder.compile()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from phase1_preprocessing import build_user_contexts, preprocess
    from phase2_rag_cache import embed_dataset, retrieve_context_for_user

    records = preprocess("The Truman Show - train")
    embed_dataset(records)

    user_ctxs = build_user_contexts("The Truman Show - train")
    graph = build_graph()

    for uctx in user_ctxs[:1]:
        rag = retrieve_context_for_user(uctx)
        initial: AgentState = {
            "user_id":        uctx["user_id"],
            "context":        uctx["context"],
            "rag_context":    rag,
            "financial_flags": [],
            "identity_flags":  [],
            "nlp_flags":       [],
            "fraudulent_ids":  [],
        }
        result = graph.invoke(initial)
        print(f"\nUser {result['user_id']}: fraudulent = {result['fraudulent_ids']}")
