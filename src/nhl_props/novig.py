"""Novig NHL shots-on-goal and saves lines + every trade on them, this season (Novig history starts 2026-08-04).

Usage: uv run python -m nhl_props.novig markets   # signed history scan, newest first, resumable (~25 s a page)
       uv run python -m nhl_props.novig trades    # free daily trade files, one parquet per day, resumable
Signed client + key: ../NBAProps/research/novig_api.py (key in NBAProps/nba-props; never print it).
⚠️ Novig refuses VPNs (451 ANONYMIZED_NETWORK) and the sandbox.
Out: data/novig/events.parquet, data/novig/markets.parquet, data/novig/trades/<date>.parquet
"""

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, "C:/Users/ivan/NBAProps/research")
from novig_api import Novig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "novig"
STATE = OUT / "scan_state.json"
TYPES = {"SHOTS_ON_GOAL", "SAVES"}
FLOOR = pd.Timestamp("2026-09-25").value // 10**6  # before this season's first game (2026-09-29)
TRADES = "https://data.novig.com/reporting/trade-data"


def _row(m):
    return {"marketId": m["marketId"], "eventId": m["eventId"], "description": m["description"],
            "marketType": m["marketType"], "strike": m.get("strike"), "status": m["status"],
            "settledTs": m.get("settledTs"), "outcomes": json.dumps(m.get("outcomes"))}


def _events(n) -> set:
    out, after = [], None
    while True:
        j = n.get("/v3/history/events", limit=5000, **({"after": after} if after else {})).json()
        out += j.get("items", [])
        after = j.get("next")
        if not after:
            break
    d = pd.DataFrame(out)
    d = d[d.league == "NHL"]
    d.to_parquet(OUT / "events.parquet")
    print(f"{len(d)} NHL events", flush=True)
    return set(d.eventId)


def markets(max_pages: int = 1000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = Novig()
    nhl = _events(n)
    st = json.loads(STATE.read_text()) if STATE.exists() else {"after": None, "pages": 0, "old_run": 0}
    path = OUT / "markets.parquet"
    t0 = time.time()
    while st["pages"] < max_pages:
        r = n.get("/v3/history/markets", limit=5000, **({"after": st["after"]} if st["after"] else {}))
        if r.status_code != 200:
            print("stop", r.status_code, r.text[:200], flush=True)
            return
        j = r.json()
        items = j.get("items", [])
        rows = [_row(m) for m in items if m["marketType"] in TYPES and m["eventId"] in nhl]
        if rows:
            new = pd.DataFrame(rows)
            if path.exists():
                new = pd.concat([pd.read_parquet(path), new], ignore_index=True).drop_duplicates("marketId", keep="last")
            new.to_parquet(path)
        settled = [m["settledTs"] for m in items if m.get("settledTs")]
        st["old_run"] = st["old_run"] + 1 if settled and max(settled) < FLOOR else 0
        st["after"], st["pages"] = j.get("next"), st["pages"] + 1
        STATE.write_text(json.dumps(st))
        oldest = pd.Timestamp(min(settled), unit="ms").date() if settled else "-"
        print(f"page {st['pages']} ({time.time() - t0:.0f}s): +{len(rows)} NHL SOG/saves, oldest {oldest}", flush=True)
        if not st["after"] or st["old_run"] >= 3:
            print("done", flush=True)
            return


def _outcomes() -> pd.DataFrame:
    rows = []
    for m in pd.read_parquet(OUT / "markets.parquet").itertuples():
        for o in json.loads(m.outcomes) or []:
            rows.append({"outcomeId": o["outcomeId"], "marketId": m.marketId, "outcome": o["name"],
                         "description": m.description, "marketType": m.marketType, "strike": m.strike,
                         "eventId": m.eventId, "won": o.get("status")})
    return pd.DataFrame(rows)


def trades() -> None:
    # Each trade prints a TAKER and a MAKER row; keep side == "TAKER" to count a trade once. price = cost / qty.
    omap = _outcomes()
    dates = [d for d in requests.get(f"{TRADES}/index.json", timeout=60).json().get("dates", []) if d >= "2026-09-28"]
    (OUT / "trades").mkdir(parents=True, exist_ok=True)
    for date in dates:
        path = OUT / "trades" / f"{date}.parquet"
        if path.exists():
            continue
        try:
            r = requests.get(f"{TRADES}/{date}/trades.csv", timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"{date}: FAILED {e}", flush=True)
            continue
        keep = [c[(c.league == "NHL") & c.marketType.isin(TYPES)]
                for c in pd.read_csv(io.BytesIO(r.content), dtype=str, chunksize=500_000)]
        d = pd.concat(keep, ignore_index=True)
        d["ts"] = pd.to_datetime(d.timestamp, utc=True, format="ISO8601")
        d[["cost", "qty"]] = d[["cost", "qty"]].astype(float)
        d["price"] = d.cost / d.qty
        d = d.drop(columns=["marketId", "marketType", "league"], errors="ignore").merge(omap, on="outcomeId", how="left")
        d.to_parquet(path, index=False)
        print(f"{date}: {len(d):,} rows, {d.marketId.notna().mean():.0%} matched, {len(r.content) / 1e6:.0f} MB", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    {"markets": markets, "trades": trades}[sys.argv[1]]()
