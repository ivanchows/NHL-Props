"""Frozen SOG baseline vs every book's closing line (PTO), 2025-26 + 2026-27 so far. Port of the NBA PTO history backtest.

One line per player-game: the MAIN line (most books priced on both sides). Consensus = median no-vig P(over) across the
two-sided books. Bets take the best price across all books on that line. No opener here: PTO history is the close only.
Staking: plus odds risk 1u, minus odds risk to win 1u; ROI = net / risked. SEs are clustered by game.
Regular season only (the model has no playoff or preseason rows). Scratched players have no model row, so they drop out.

Usage: uv run python -m nhl_props.pto_backtest [pred table, default baseline_pred]
"""

import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from nhl_props.novig_backtest import norm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MIN_ODDS, MAX_ODDS = -250, 250
SPLIT = "2026-02-01"  # walk-forward blend: fit before, test from here


def implied(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o, 100) / (np.abs(o) + 100)


def american(p: float) -> str:
    return f"{-100 * p / (1 - p):+.0f}" if p >= 0.5 else f"{100 * (1 - p) / p:+.0f}"


def payoff(odds, won):
    """(net, risked) in units: plus odds risk 1u, minus odds risk to win 1u."""
    odds = np.asarray(odds, float)
    risk = np.where(odds < 0, -odds / 100, 1.0)
    win = np.where(odds < 0, 1.0, odds / 100)
    return np.where(won, win, -risk), risk


def names() -> pd.DataFrame:
    rows = {}
    for season in (2025, 2026):
        for f in glob.glob(str(DATA / f"raw/{season}/*_pbp.json.gz")):
            for r in json.load(gzip.open(f)).get("rosterSpots", []):
                rows[r["playerId"]] = f"{r['firstName']['default']} {r['lastName']['default']}"
    return pd.DataFrame(list(rows.items()), columns=["player_id", "full"])


def model_rows(pred: str) -> pd.DataFrame:
    p = pd.read_parquet(DATA / f"{pred}.parquet").merge(names(), on="player_id", how="left")
    p["key"] = p.full.map(norm, na_action="ignore")
    p["key2"] = p.key.str[0] + " " + p.key.str.split(n=1).str[1]
    return p[["game_date", "player_id", "team", "key", "key2", "sog", "mu", "r"]].dropna(subset=["key"])


def lines() -> pd.DataFrame:
    """Per game x player x line: book count, consensus, best price per side. Half-point lines only."""
    s = pd.read_parquet(DATA / "pto/SOG.parquet")
    s = s[((s.line * 2) % 2 == 1) & s.player.notna()].copy()
    w = s.pivot_table(index=["game_id", "player_id_pto", "line", "book"], columns="side", values="odds", aggfunc="last")
    two = w.dropna().copy()
    po, pu = implied(two["over"]), implied(two["under"])
    two["p"] = po / (po + pu)
    two["hold"] = po + pu - 1
    a = two.groupby(level=[0, 1, 2]).agg(books=("p", "size"), cons=("p", "median"), hold=("hold", "median"))
    best = s.groupby(["game_id", "player_id_pto", "line", "side"]).odds.max().unstack()
    a = a.join(best.rename(columns={"over": "best_over", "under": "best_under"}))
    meta = s.groupby(["game_id", "player_id_pto"]).agg(start=("start", "first"), player=("player", "first"),
                                                       away=("away", "first"), home=("home", "first"))
    return a.reset_index().merge(meta, on=["game_id", "player_id_pto"])


def attach(a: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    a = a.assign(game_date=a.start.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    a["key"] = a.player.map(norm)
    a["key2"] = a.key.str[0] + " " + a.key.str.split(n=1).str[1]
    m1 = a.merge(p.drop(columns="key2"), on=["game_date", "key"])
    # Two players can share a name (Sebastian Aho): keep the one whose team is in this game.
    n = m1.groupby(["game_id", "player_id_pto", "line"]).player_id.transform("nunique")
    m1 = m1[(n == 1) | (m1.team == m1.away) | (m1.team == m1.home)]
    done = set(zip(m1.game_id, m1.player_id_pto))
    rest = a[[g not in done for g in zip(a.game_id, a.player_id_pto)]]
    uniq = p.drop_duplicates(["game_date", "key2"], keep=False).drop(columns="key")
    m2 = rest.drop(columns="key").merge(uniq, on=["game_date", "key2"])
    m = pd.concat([m1, m2], ignore_index=True)
    m = m[~m.duplicated(["game_id", "player_id_pto", "line"], keep=False)]
    m["p_over"] = nbinom.sf(np.floor(m.line).astype(int), m.r, m.r / (m.r + m.mu)).clip(0.01, 0.99)
    m["y"] = (m.sog > m.line).astype(float)
    m["month"] = m.game_date.str[:7]
    return m


def main_lines(m: pd.DataFrame) -> pd.DataFrame:
    """The line most books price on both sides; ties go to the consensus closest to 50/50.
    Keyed on the NHL player id: PTO can list one player twice (Mike / Michael Benning)."""
    m = m.assign(dist=(m.cons - 0.5).abs()).sort_values(["books", "dist"], ascending=[False, True])
    return m.groupby(["game_id", "player_id"]).head(1)


def logloss(y, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def logit(p):
    return np.log(p / (1 - p))


def fit_logistic(X, y, steps=30):
    b = np.zeros(X.shape[1])
    for _ in range(steps):
        q = 1 / (1 + np.exp(-X @ b))
        W = q * (1 - q)
        b += np.linalg.solve((X.T * W) @ X + 1e-9 * np.eye(X.shape[1]), X.T @ (y - q))
    return b


def bets(m: pd.DataFrame, p, lo=0.03, hi=1.0, odds_over=None, odds_under=None) -> pd.DataFrame:
    """One bet per row: the side with the larger edge at the given prices, inside the odds cap and the edge band."""
    oo = m.best_over if odds_over is None else odds_over
    ou = m.best_under if odds_under is None else odds_under
    eo = pd.Series(p - implied(oo), index=m.index).where(oo.between(MIN_ODDS, MAX_ODDS))
    eu = pd.Series((1 - p) - implied(ou), index=m.index).where(ou.between(MIN_ODDS, MAX_ODDS))
    over = eo.fillna(-1) >= eu.fillna(-1)
    edge = np.where(over, eo.fillna(-1), eu.fillna(-1))
    keep = (edge >= lo) & (edge < hi)
    b = m[keep].assign(side=np.where(over[keep], "over", "under"), edge=edge[keep],
                       odds=np.where(over[keep], oo[keep], ou[keep]))
    b["won"] = np.where(b.side == "over", b.y == 1, b.y == 0)
    b["net"], b["risk"] = payoff(b.odds, b.won)
    return b


def show(b: pd.DataFrame, label: str) -> None:
    if b.empty:
        print(f"  {label:26s} no bets")
        return
    g = b.groupby("game_id").agg(net=("net", "sum"), risk=("risk", "sum"))
    se = g.net.std() * np.sqrt(len(g)) / g.risk.sum()
    print(f"  {label:26s} {len(b):6,} bets {b.won.sum():5,}-{(~b.won).sum():<5,} {b.net.sum():+8.1f}u  "
          f"ROI {b.net.sum() / b.risk.sum():+6.1%} ± {se:4.1%}  overs {np.mean(b.side == 'over'):4.0%}")


def book_prices(m: pd.DataFrame, book: str) -> tuple[pd.Series, pd.Series]:
    s = pd.read_parquet(DATA / "pto/SOG.parquet", columns=["game_id", "player_id_pto", "line", "side", "book", "odds"])
    x = s[s.book == book].pivot_table(index=["game_id", "player_id_pto", "line"], columns="side", values="odds", aggfunc="last")
    j = m[["game_id", "player_id_pto", "line"]].join(x, on=["game_id", "player_id_pto", "line"])
    return j["over"].reindex(m.index), j["under"].reindex(m.index)


def main(pred: str = "baseline_pred") -> None:
    a, p = lines(), model_rows(pred)
    m = attach(a, p)
    print(f"{a.groupby(['game_id', 'player_id_pto']).ngroups:,} PTO player-games -> {m.groupby(['game_id', 'player_id_pto']).ngroups:,} "
          f"matched, {m.game_id.nunique():,} games, {m.game_date.min()} -> {m.game_date.max()}")
    ml = main_lines(m)
    ml = ml[ml.books >= 3].copy()
    print(f"main lines with 3+ two-sided books: {len(ml):,}; books median {ml.books.median():.0f}, hold {ml.hold.median():.1%}; "
          f"line mix {ml.line.value_counts().head(4).to_dict()}")
    print(f"actual over rate {ml.y.mean():.3f}, consensus {ml.cons.mean():.3f}, model {ml.p_over.mean():.3f}")

    print("\nLog loss, main lines:")
    print(f"  all        model {logloss(ml.y, ml.p_over):.4f} | consensus {logloss(ml.y, ml.cons):.4f} | "
          f"50/50 {logloss(ml.y, (ml.p_over + ml.cons) / 2):.4f}")
    for ln, x in ml.groupby("line"):
        if len(x) >= 500:
            print(f"  line {ln}   model {logloss(x.y, x.p_over):.4f} | consensus {logloss(x.y, x.cons):.4f} | n {len(x):,}")
    for mo, x in ml.groupby("month"):
        print(f"  {mo}    model {logloss(x.y, x.p_over):.4f} | consensus {logloss(x.y, x.cons):.4f} | "
              f"over {x.y.mean():.3f} cons {x.cons.mean():.3f} model {x.p_over.mean():.3f} | n {len(x):,}")

    print("\nModel bets at the best price across books:")
    for lo, hi, lab in ((0.0, 0.03, "edge 0-3%"), (0.03, 0.06, "edge 3-6%"), (0.06, 0.12, "edge 6-12%"), (0.12, 1, "edge 12%+"),
                        (0.03, 0.12, "edge 3-12% (NBA band)")):
        show(bets(ml, ml.p_over, lo, hi), lab)
    b = bets(ml, ml.p_over, 0.03, 0.12)
    show(b[b.side == "over"], "  overs")
    show(b[b.side == "under"], "  unders")
    for s0, x in b.groupby(b.game_date >= "2026-09-01"):
        show(x, "  2026-27" if s0 else "  2025-26")

    print("\nControls at the best price (no model):")
    show(bets(ml, np.full(len(ml), 0.0), -9, 9, odds_over=ml.best_over - 100000), "every under")
    show(bets(ml, np.full(len(ml), 1.0), -9, 9, odds_under=ml.best_under - 100000), "every over")
    show(bets(ml, ml.cons.to_numpy(), 0.0, 1), "best price beats consensus")
    show(bets(ml, ml.cons.to_numpy(), 0.03, 1), "beats consensus by 3%+")

    print("\nModel bets (edge 3-12%) at one book only:")
    for book in ("DRAFT_KINGS", "FAN_DUEL", "MGM", "CAESARS", "BET_365", "FANATICS", "THE_SCORE", "NO_VIG", "PROPHET_BETTING", "FLIFF"):
        oo, ou = book_prices(ml, book)
        have = oo.notna() | ou.notna()
        show(bets(ml[have], ml.p_over[have], 0.03, 0.12, oo[have].fillna(-99999), ou[have].fillna(-99999)), book)

    print(f"\nWalk-forward: does the model add to the close? Fit before {SPLIT}, test after")
    fit, test = ml[ml.game_date < SPLIT], ml[ml.game_date >= SPLIT]
    Xb = lambda d: np.c_[logit(d.p_over), logit(d.cons), np.ones(len(d))]
    Xc = lambda d: np.c_[logit(d.cons), np.ones(len(d))]
    bb, bc = fit_logistic(Xb(fit), fit.y.to_numpy()), fit_logistic(Xc(fit), fit.y.to_numpy())
    pb = 1 / (1 + np.exp(-Xb(test) @ bb))
    pc = 1 / (1 + np.exp(-Xc(test) @ bc))
    print(f"  blend weights model {bb[0]:.3f} consensus {bb[1]:.3f} const {bb[2]:+.3f}; control consensus {bc[0]:.3f} const {bc[1]:+.3f}")
    y = test.y.to_numpy()
    dif = (-(y * np.log(pb) + (1 - y) * np.log(1 - pb))) - (-(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    print(f"  fit {len(fit):,} / test {len(test):,}. Log loss blend {logloss(y, pb):.4f} | control {logloss(y, pc):.4f} | "
          f"consensus {logloss(y, test.cons):.4f} | model {logloss(y, test.p_over):.4f}; blend - control {dif.mean():+.5f} ± {dif.std() / np.sqrt(len(y)):.5f}")
    for lo, hi in ((0.0, 0.03), (0.03, 0.06), (0.06, 1)):
        show(bets(test, pb, lo, hi), f"blend edge {lo:.0%}-{hi:.0%}")
        show(bets(test, pc, lo, hi), f"control edge {lo:.0%}-{hi:.0%}")
    print(f"\n(typical price: fair {american(float(ml.cons.median()))})")


if __name__ == "__main__":
    main(*sys.argv[1:])
