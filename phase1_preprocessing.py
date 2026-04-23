"""
Phase 1: Data Pre-processing Pipeline
======================================
Two outputs:
  - preprocess()          → per-transaction enriched records (for RAG embedding)
  - build_user_contexts() → per-user holistic context (for LangGraph agents)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import GEO_ANOMALY_KM, URGENT_WORDS, ECONOMIC_PANIC_THRESHOLD


# ── Haversine distance ─────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Text helpers ───────────────────────────────────────────────────────────

_URL_RE      = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+", re.I)
_DATE_RE     = re.compile(r"Date:\s*(.+)", re.I)
_TO_RE       = re.compile(r"^To:\s*(\S+)", re.I | re.M)
_PHISHING_RE = re.compile(
    r"paypa1|amaz0n-verify|ub3r|netfl1x|deutschebank-secure|r1d3share|"
    r"citydriv3|ub3r-verify|ub3r-secure|amaz0n|paypa1-secure|uber-secure-verify|"
    r"uber-secure-login|r1d3share-verify",
    re.I,
)

def _clean(val: Any, default: str = "") -> str:
    if val is None:
        return default
    s = str(val)
    return default if s.lower() == "nan" else s


# ── Loaders ────────────────────────────────────────────────────────────────

def load_world(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    retval = {
        "transactions": pd.read_csv(data_dir / "transactions.csv"),
        "users":        json.loads((data_dir / "users.json").read_text(encoding="utf-8")),
        "locations":    json.loads((data_dir / "locations.json").read_text(encoding="utf-8")),
        "sms":          [r["sms"]  for r in json.loads((data_dir / "sms.json").read_text(encoding="utf-8"))],
        "mails":        [r["mail"] for r in json.loads((data_dir / "mails.json").read_text(encoding="utf-8"))],
    }
    
    calls_path = data_dir / "calls.json"
    if calls_path.exists():
        calls = []
        for c in json.loads(calls_path.read_text(encoding="utf-8")):
            calls.append(f"Date: {c.get('ts', '')}\nTo: {c.get('raw_name_identifier', '')}\nMessage: [PHISHING CALL] {c.get('text', '')}")
        retval["calls"] = calls
    
    return retval


# ── Indexes ────────────────────────────────────────────────────────────────

def _build_user_index(users: list[dict]) -> dict[str, dict]:
    idx = {}
    for u in users:
        iban = u["iban"].strip()
        res  = u.get("residence", {})
        idx[iban] = {**u,
                     "home_lat":  float(res.get("lat", 0)),
                     "home_lng":  float(res.get("lng", 0)),
                     "home_city": res.get("city", "")}
    return idx

def _build_biotag_index(locations: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for rec in locations:
        idx.setdefault(rec["biotag"], []).append(rec)
    for v in idx.values():
        v.sort(key=lambda r: r["timestamp"])
    return idx

def _sender_to_biotag(df: pd.DataFrame) -> dict[str, str]:
    m = {}
    for _, row in df.iterrows():
        sid  = str(row["sender_id"])
        siban = str(row.get("sender_iban", ""))
        if sid.count("-") >= 3:
            m[siban] = sid
    return m

def _nearest_location(biotag, ts, bio_idx):
    recs = bio_idx.get(biotag)
    if not recs:
        return None
    ts_str = ts.isoformat(sep="T")
    before = [r for r in recs if r["timestamp"] <= ts_str]
    best   = before[-1] if before else recs[0]
    return best["lat"], best["lng"]


# ── Message parsing ────────────────────────────────────────────────────────

def _parse_messages(messages: list[str]) -> list[dict]:
    """Parse SMS/email into structured records."""
    out = []
    for msg in messages:
        date_m = _DATE_RE.search(msg)
        to_m   = _TO_RE.search(msg)
        ts_str = date_m.group(1).strip() if date_m else ""
        try:
            ts = pd.to_datetime(ts_str, utc=False)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
        except Exception:
            ts = None
        out.append({
            "ts":          ts,
            "to":          to_m.group(1).strip() if to_m else "",
            "text":        msg,
            "is_phishing": bool(_PHISHING_RE.search(msg)),
            "has_link":    bool(_URL_RE.search(msg)),
        })
    return out

def _map_phones_to_users(messages: list[dict], users: list[dict]) -> dict[str, str]:
    """Map phone number → user first_name by counting name mentions per phone."""
    from collections import Counter, defaultdict
    phone_name_counts: dict[str, Counter] = defaultdict(Counter)
    for m in messages:
        phone = m["to"]
        if not phone.startswith("+"):
            continue
        for u in users:
            name = u["first_name"]
            if name in m["text"]:
                phone_name_counts[phone][name] += 1
    result = {}
    for phone, counter in phone_name_counts.items():
        if counter:
            result[phone] = counter.most_common(1)[0][0]
    return result  # {phone: first_name}


# ── Per-transaction preprocessing (for RAG) ────────────────────────────────

def preprocess(data_dir: str | Path) -> list[dict[str, Any]]:
    """Per-transaction enriched records for RAG embedding."""
    raw   = load_world(data_dir)
    df    = raw["transactions"].copy()
    users = raw["users"]

    user_idx       = _build_user_index(users)
    bio_idx        = _build_biotag_index(raw["locations"])
    iban_to_biotag = _sender_to_biotag(df)
    all_messages   = _parse_messages(raw["sms"] + raw["mails"] + raw.get("calls", []))

    df["ts"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("ts").reset_index(drop=True)

    user_stats: dict[str, dict] = {}
    for iban, grp in df.groupby("sender_iban"):
        amounts = grp["amount"].dropna()
        user_stats[iban] = {
            "mean": float(amounts.mean()) if len(amounts) else 0.0,
            "std":  float(amounts.std())  if len(amounts) > 1 else 1.0,
        }

    records = []
    for _, row in df.iterrows():
        txn_id      = str(row["transaction_id"])
        sender_iban = str(row.get("sender_iban", ""))
        ts: pd.Timestamp = row["ts"]

        user    = user_idx.get(sender_iban, {})
        biotag  = iban_to_biotag.get(sender_iban, str(row.get("sender_id", "")))
        home_lat = user.get("home_lat", 0.0)
        home_lng = user.get("home_lng", 0.0)

        window_start  = ts - pd.Timedelta(hours=24)
        user_txns     = df[df["sender_iban"] == sender_iban]
        velocity_24h  = int(((user_txns["ts"] >= window_start) & (user_txns["ts"] < ts)).sum())

        amount    = float(row.get("amount", 0) or 0)
        stats     = user_stats.get(sender_iban, {"mean": 0.0, "std": 1.0})
        std       = stats["std"] if stats["std"] > 0 else 1.0
        amount_z  = round(float((amount - stats["mean"]) / std), 3)

        is_high_risk = False
        loc_pt = _nearest_location(biotag, ts, bio_idx)
        if loc_pt and home_lat and home_lng:
            is_high_risk = _haversine_km(home_lat, home_lng, loc_pt[0], loc_pt[1]) > GEO_ANOMALY_KM

        hour       = ts.hour
        is_night   = hour < 6 or hour >= 22
        is_weekend = ts.day_of_week >= 5

        balance_after = float(row.get("balance_after", 0) or 0)
        balance_ratio = round(amount / balance_after, 3) if balance_after > 0 else 0.0

        # Per-user phishing count in 60-day window
        first_name = user.get("first_name", "")
        cutoff     = ts - pd.Timedelta(days=60)
        phishing_count = sum(
            1 for m in all_messages
            if m["ts"] is not None
            and cutoff <= m["ts"] < ts
            and m["is_phishing"]
            and first_name in m["text"]
        )
        salary_monthly = user.get("salary", 0) / 12
        amt_vs_salary = amount / salary_monthly if salary_monthly > 0 else 0

        records.append({
            "transaction_id":         txn_id,
            "user_id":                biotag,
            "sender_iban_hash":       hashlib.sha256(sender_iban.encode()).hexdigest()[:12],
            "transaction_type":       _clean(row.get("transaction_type")),
            "amount":                 amount,
            "payment_method":         _clean(row.get("payment_method")),
            "description":            _clean(row.get("description")),
            "location_label":         _clean(row.get("location")),
            "transaction_velocity_24h": velocity_24h,
            "amount_z_score":         amount_z,
            "is_high_risk_location":  is_high_risk,
            "balance_ratio":          balance_ratio,
            "balance_after":          balance_after,
            "hour_of_day":            hour,
            "is_weekend":             is_weekend,
            "is_night":               is_night,
            "recent_phishing_count":  phishing_count,
            "user_salary":            int(user.get("salary", 0)),
            "user_job":               _clean(user.get("job")),
            "user_home_city":         _clean(user.get("home_city")),
            "amt_vs_salary_ratio": round(amt_vs_salary, 2),
            "is_whale_txn": amount > ECONOMIC_PANIC_THRESHOLD
        })
    return records


# ── Per-user holistic context (for LangGraph agents) ───────────────────────

def build_user_contexts(data_dir: str | Path) -> list[dict[str, Any]]:
    """
    Returns one dict per user containing:
      - user_id, profile text
      - formatted transaction table (with IDs, amounts, risk flags)
      - formatted phishing + communications summary
      - raw transaction list (for post-processing)
    """
    raw = load_world(data_dir)
    df = raw["transactions"].copy()
    users = raw["users"]

    user_idx = _build_user_index(users)

    # [FIX AICI] -> Trebuie să încărcăm indexul BioTag-urilor pentru a calcula distanța
    bio_idx = _build_biotag_index(raw["locations"])

    iban_to_biotag = _sender_to_biotag(df)
    all_messages = _parse_messages(raw["sms"] + raw["mails"] + raw.get("calls", []))
    phone_to_name = _map_phones_to_users(all_messages, users)
    name_to_phone = {v: k for k, v in phone_to_name.items()}

    df["ts"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("ts").reset_index(drop=True)

    user_stats: dict[str, dict] = {}
    for iban, grp in df.groupby("sender_iban"):
        amounts = grp["amount"].dropna()
        user_stats[iban] = {
            "mean": float(amounts.mean()) if len(amounts) else 0.0,
            "std":  float(amounts.std())  if len(amounts) > 1 else 1.0,
        }

    # Group by user (sender_iban belonging to a known user)
    known_ibans = {u["iban"]: u for u in users}
    user_contexts = []

    for iban, user in known_ibans.items():
        first_name = user["first_name"]
        biotag     = next((iban_to_biotag[ib] for ib in iban_to_biotag if ib == iban), iban)
        user_txns  = df[df["sender_iban"] == iban].copy()
        stats      = user_stats.get(iban, {"mean": 0.0, "std": 1.0})
        std        = stats["std"] if stats["std"] > 0 else 1.0

        # ── Transaction table ──────────────────────────────────────────
        txn_lines = []
        txn_records = []
        prev_ts = None

        # Ensure we have the user's home coordinates for distance checking
        home_lat = user.get("home_lat", 0.0)
        home_lng = user.get("home_lng", 0.0)

        for _, row in user_txns.iterrows():
            tid = str(row["transaction_id"])
            ts = row["ts"]
            amount = float(row.get("amount", 0) or 0)
            z = round(float((amount - stats["mean"]) / std), 2)
            hour = ts.hour
            night = hour < 6 or hour >= 22
            desc = _clean(row.get("description"))
            method = _clean(row.get("payment_method"))
            loc = _clean(row.get("location"))

            flags = []
            if night:       flags.append("NIGHT⚠️")
            if abs(z) > 2:  flags.append(f"z={z}⚠️")

            # 1. Burst Detection
            if prev_ts is not None:
                minutes_diff = (ts - prev_ts).total_seconds() / 60
                if minutes_diff < 60:
                    flags.append(f"BURST({int(minutes_diff)}m)⚠️")

            # 2. [NEW] Geo-Anomaly Detection directly in the prompt
            loc_pt = _nearest_location(biotag, ts, bio_idx)
            if loc_pt and home_lat and home_lng:
                dist = _haversine_km(home_lat, home_lng, loc_pt[0], loc_pt[1])
                if dist > GEO_ANOMALY_KM:
                    flags.append(f"GEO-ANOMALY({int(dist)}km)⚠️")

            if method and method not in ("", "nan"):
                flags.append(f"via {method}")
            if loc:         flags.append(f"@ {loc}")

            txn_lines.append(
                f"  [{tid}] {ts.strftime('%Y-%m-%d %H:%M')} | "
                f"{desc or '(no description)'} | €{amount:.2f} | "
                + (" | ".join(flags) if flags else "routine")
            )
            txn_records.append({"transaction_id": tid, "amount": amount,
                                "is_night": night, "description": desc,
                                "payment_method": method})
            prev_ts = ts  # Actualizăm timestamp-ul

        # ── Communications ─────────────────────────────────────────────
        phone     = name_to_phone.get(first_name, "")
        user_msgs = [m for m in all_messages
                     if m["ts"] is not None
                     and (phone and phone in m["to"] or 
                          first_name.lower() in m["to"].lower() or 
                          first_name in m["text"][:200])]
        user_msgs.sort(key=lambda m: m["ts"])

        phishing_lines = []
        legit_summary_count = 0
        for m in user_msgs:
            snippet = m["text"].replace("\n", " ")[:140]
            is_urgent_call = "[PHISHING CALL]" in snippet
            tag = "PHISHING CALL" if is_urgent_call else "PHISHING"
            if m["is_phishing"] or is_urgent_call:
                phishing_lines.append(f"  [{m['ts'].strftime('%Y-%m-%d %H:%M')} {tag}] {snippet}")
            else:
                legit_summary_count += 1

        comms_block = ""
        if phishing_lines:
            comms_block += "PHISHING MESSAGES RECEIVED:\n" + "\n".join(phishing_lines) + "\n"
        comms_block += f"({legit_summary_count} legitimate messages omitted)\n"

        # ── Profile ────────────────────────────────────────────────────
        profile = (
            f"Name: {user['first_name']} {user['last_name']} | "
            f"Job: {user['job']} | Salary: €{user['salary']:,}/year | "
            f"Home: {user.get('residence', {}).get('city', 'unknown')}\n"
            f"Background: {user.get('description', '')[:300]}"
        )

        context = (
            f"=== USER PROFILE ===\n{profile}\n\n"
            f"=== TRANSACTIONS (chronological, {len(txn_lines)} total) ===\n"
            + "\n".join(txn_lines) + "\n\n"
            f"=== COMMUNICATIONS ===\n{comms_block}"
        )

        user_contexts.append({
            "user_id":     biotag,
            "first_name":  first_name,
            "context":     context,
            "transaction_ids": [t["transaction_id"] for t in txn_records],
            "transactions":    txn_records,
        })

    return user_contexts


if __name__ == "__main__":
    import sys, pprint
    data_path = sys.argv[1] if len(sys.argv) > 1 else "The Truman Show - train"
    ctxs = build_user_contexts(data_path)
    for ctx in ctxs:
        print(f"\n{'='*60}")
        print(ctx["context"])