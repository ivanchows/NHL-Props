"""Frozen SOG / saves baseline vs Novig lines, 2026-27 so far. Port of ../MLBProps/src/mlb_props/novig_backtest.py.

Novig has trades, no quotes. Executable price per side at time T = the last TAKER price on that side before T, if
under 60 min old. Snapshots: open (first trade + 1 h) and close (last trades before startsTs).
Fee 0 (Novig pre-game takers pay none). Net CLV = close mid - cost, per bet (the primary metric).
Model rows exist only for skaters who played, so scratched players (Novig voids them) drop out of both sides.

Usage: uv run python -m nhl_props.novig_backtest [pred table] [mu column]
"""

import glob
import gzip
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STALE = pd.Timedelta(minutes=60)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").replace("-", " ").replace("'", "").split())


def full_names() -> pd.DataFrame:
    """player_id -> first / last name from the play-by-play roster of every 2026-27 game."""
    rows = {}
    for f in glob.glob(str(DATA / "raw/2026/*_pbp.json.gz")):
        for r in json.load(gzip.open(f)).get("rosterSpots", []):
            rows[r["playerId"]] = (r["firstName"]["default"], r["lastName"]["default"])
    return pd.DataFrame([(k, *v) for k, v in rows.items()], columns=["player_id", "first", "last"])


def trades(kind: str) -> pd.DataFrame:
    t = pd.concat(pd.read_parquet(f) for f in glob.glob(str(DATA / "novig/trades/*.parquet")))
    t = t[(t.side == "TAKER") & (t.marketType == kind) & (t.legs.astype(int) == 1)].copy()
    ev = pd.read_parquet(DATA / "novig/events.parquet")[["eventId", "startsTs"]]
    t = t.merge(ev, on="eventId")
    t["start"] = pd.to_datetime(pd.to_numeric(t.startsTs), unit="ms")
    t["ts"] = t.ts.dt.tz_convert("UTC").dt.tz_localize(None)
    t = t[t.ts < t.start]
    t["over"] = t.outcome.str.startswith("Over")
    t["player"] = t.description.str.replace(rf"\s+[\d.]+\s+{kind}$", "", regex=True)
    t["line"] = t.strike.astype(float)
    t["game_date"] = t.start.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    t["over_won_nv"] = np.where(t.over, t.won == "WIN", t.won == "LOSS")
    return t.sort_values("ts")


def snapshot(t: pd.DataFrame, when: str) -> pd.DataFrame:
    first = t.groupby("marketId").ts.transform("min")
    cut = (first + pd.Timedelta(hours=1)) if when == "open" else t.start
    s = t[t.ts <= cut].assign(cut=cut[t.ts <= cut])
    last = s.groupby(["marketId", "over"]).tail(1)
    last = last[last.cut - last.ts <= STALE]
    piv = last.pivot_table(index="marketId", columns="over", values="price", aggfunc="last")
    piv = piv.rename(columns={True: "ask_over", False: "ask_under"})
    meta = s.groupby("marketId").agg(player=("player", "first"), line=("line", "first"), game_date=("game_date", "first"),
                                     cut=("cut", "first"), over_won_nv=("over_won_nv", "first"), event=("eventId", "first"))
    return meta.join(piv, how="inner").reset_index()


def model_rows(kind: str, pred: str = "baseline_pred", col: str = "mu") -> pd.DataFrame:
    """One row per player-game: name keys, outcome y, and a function-ready (mu, r)."""
    names = full_names()
    if kind == "SHOTS_ON_GOAL":
        p = pd.read_parquet(DATA / f"{pred}.parquet")
        p = p[p.season == 2026].rename(columns={"sog": "y"})
        p["mu"] = p[col]
    else:
        p = pd.read_parquet(DATA / "saves_pred.parquet")
        p = p[p.season == 2026].rename(columns={"saves": "y"})
    p = p.merge(names, on="player_id", how="left")
    p["key"] = (p["first"] + " " + p["last"]).map(norm)
    p["key2"] = p["first"].str[0].str.lower() + " " + p["last"].map(norm)
    return p[["game_date", "player_id", "key", "key2", "y", "mu", "r"]]


def attach(s: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    s = s.assign(key=s.player.map(norm))
    s["key2"] = s.key.str[0] + " " + s.key.str.split(n=1).str[1]
    m1 = s.merge(p.drop(columns="key2"), on=["game_date", "key"])
    rest = s[~s.marketId.isin(m1.marketId)]
    uniq = p.drop_duplicates(["game_date", "key2"], keep=False).drop(columns="key")
    m2 = rest.drop(columns="key").merge(uniq, on=["game_date", "key2"])
    m = pd.concat([m1, m2], ignore_index=True)
    m["p_over"] = nbinom.sf(np.floor(m.line).astype(int), m.r, m.r / (m.r + m.mu))
    m["over_won"] = (m.y > m.line).astype(float)
    return m


def evaluate(m: pd.DataFrame, close: pd.DataFrame | None, when: str) -> None:
    bad = (m.over_won == 1) != m.over_won_nv
    print(f"\n[{when}] {len(m):,} lines, {m.event.nunique()} games, {m.player_id.nunique()} players; "
          f"settlement mismatches vs box score {bad.sum()}")
    m = m[~bad]
    both = m.dropna(subset=["ask_over", "ask_under"])
    mid = ((both.ask_over + 1 - both.ask_under) / 2).clip(0.01, 0.99).to_numpy()
    y = both.over_won.to_numpy()
    ll = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    print(f"  two-sided lines {len(both):,}; overround {(both.ask_over + both.ask_under - 1).mean():+.3f}; "
          f"log loss model {ll(both.p_over.clip(0.01, 0.99).to_numpy()):.4f} | market mid {ll(mid):.4f} | "
          f"50/50 {ll((both.p_over.clip(0.01, 0.99).to_numpy() + mid) / 2):.4f}")
    if close is not None:
        cm = close.dropna(subset=["ask_over", "ask_under"])
        cm = cm.assign(close_mid=(cm.ask_over + 1 - cm.ask_under) / 2).set_index("marketId").close_mid
        m = m.assign(close_mid=m.marketId.map(cm))
    eo, eu = m.p_over - m.ask_over, (1 - m.p_over) - m.ask_under
    m = m.assign(edge=np.fmax(eo.fillna(-1), eu.fillna(-1)),
                 side=np.where(eo.fillna(-1) >= eu.fillna(-1), 1, -1))
    m["cost"] = np.where(m.side == 1, m.ask_over, m.ask_under)
    m["profit"] = np.where(m.side == 1, m.over_won, 1 - m.over_won) - m.cost
    if close is not None:
        m["clv"] = np.where(m.side == 1, m.close_mid, 1 - m.close_mid) - m.cost
    for label, b0 in (("every line", m), ("best line per player-game", m.sort_values("edge").groupby(["event", "player_id"]).tail(1))):
        print(f"  {label}:")
        for e in (0.0, 0.02, 0.04, 0.06, 0.10):
            b = b0[b0.edge > e]
            if b.empty:
                continue
            g = b.groupby(["event", "player_id"]).agg(profit=("profit", "sum"), cost=("cost", "sum"))
            txt = (f"    edge>={e:4.0%}  bets {len(b):5,} ({len(g):4,} player-games)  ROI {g.profit.sum() / g.cost.sum():+6.1%} "
                   f"± {g.profit.std() * np.sqrt(len(g)) / g.cost.sum():4.1%}  overs {np.mean(b.side == 1):.0%}")
            if "clv" in b and b.clv.notna().any():
                c = b.dropna(subset=["clv"]).groupby(["event", "player_id"]).clv.mean()
                txt += f"  CLV {100 * c.mean():+.2f}c ± {100 * c.std() / np.sqrt(len(c)):.2f} ({len(c)})"
            print(txt)


def main(pred: str = "baseline_pred", col: str = "mu", kinds=("SHOTS_ON_GOAL", "SAVES")) -> None:
    for kind in kinds:
        if kind == "SAVES" and not (DATA / "saves_pred.parquet").exists():
            print("\nSAVES: no saves model yet (data/saves_pred.parquet), skipped")
            continue
        t = trades(kind)
        print(f"\n===== {kind}: {len(t):,} pre-game taker trades, {t.marketId.nunique():,} lines, "
              f"{t.ts.min():%Y-%m-%d} -> {t.ts.max():%Y-%m-%d}")
        p = model_rows(kind, pred, col)
        close = snapshot(t, "close")
        for when in ("open", "close"):
            s = snapshot(t, when)
            m = attach(s, p)
            print(f"matched {m.marketId.nunique():,} / {s.marketId.nunique():,} {when} lines to model rows; "
                  f"line mix {m.line.value_counts().head(5).to_dict()}")
            evaluate(m, close if when == "open" else None, when)


if __name__ == "__main__":
    main(*sys.argv[1:])
