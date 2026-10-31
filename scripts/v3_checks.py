"""v3 vs v2 on the same PTO closing lines, at the best, 2nd best and median price, plus a leak check. Usage: uv run python scripts/v3_checks.py"""
import json

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from nhl_props import pto_backtest as B
from nhl_props import v3 as V3
from nhl_props.baseline import DATA, load
from nhl_props.v2 import features

a = B.lines()
extra = pd.read_parquet(DATA / "v3_pred.parquet")[["game_date", "player_id", "mu_v2", "fresh_same"]]
m = B.attach(a, B.model_rows("v3_pred").merge(extra, on=["game_date", "player_id"], how="left"))
ml = B.main_lines(m)
ml = ml[ml.books >= 3].copy()
ml["p_v2"] = nbinom.sf(np.floor(ml.line).astype(int), ml.r, ml.r / (ml.r + ml.mu_v2)).clip(.01, .99)
print(f"{len(ml):,} lines. Log loss v3 {B.logloss(ml.y, ml.p_over):.4f} | v2 {B.logloss(ml.y, ml.p_v2):.4f} | consensus {B.logloss(ml.y, ml.cons):.4f} | "
      f"50/50 v3 {B.logloss(ml.y, (ml.p_over + ml.cons) / 2):.4f}")
k = ml.fresh_same > 0
print(f"teammate newly out ({k.mean():.0%}): v3 {B.logloss(ml.y[k], ml.p_over[k]):.4f} | v2 {B.logloss(ml.y[k], ml.p_v2[k]):.4f} | consensus {B.logloss(ml.y[k], ml.cons[k]):.4f}")

s = pd.read_parquet(DATA / "pto/SOG.parquet", columns=["game_id", "player_id_pto", "line", "side", "odds"])
s["imp"] = B.implied(s.odds)
s = s.sort_values("imp")
g = s.groupby(["game_id", "player_id_pto", "line", "side"])
px = pd.DataFrame({"second": s[g.cumcount() == 1].set_index(["game_id", "player_id_pto", "line", "side"]).odds, "med": g.imp.median()})


def prices(b):
    j = b.join(px, on=["game_id", "player_id_pto", "line", "side"])
    q = j.med.to_numpy()
    return j.second.fillna(j.odds), pd.Series(np.where(q >= .5, -100 * q / (1 - q), 100 * (1 - q) / q), index=j.index)


for lab, p in (("v2", ml.p_v2), ("v3", ml.p_over)):
    print(lab)
    for lo, hi in ((.03, .06), (.06, .12), (.12, 1), (.06, 1)):
        B.show(B.bets(ml, p, lo, hi), f"edge {lo:.0%}-{hi:.0%}")
    b = B.bets(ml, p, .06, 1)
    so, mo = prices(b)
    fair = np.where(b.side == "over", b.cons, 1 - b.cons)
    for t, o in (("2nd best", so), ("median book", mo)):
        n, r = B.payoff(o, b.won)
        print(f"  6%+ at {t}: {n.sum():+.1f}u, ROI {n.sum() / r.sum():+.1%}")
    print(f"  6%+ won {b.won.mean():.3f}, consensus says {fair.mean():.3f}; unders {np.mean(b.side == 'under'):.0%}")
    d = b.game_date.sort_values().iloc[len(b) // 2]
    for nm, x in (("first half", b[b.game_date < d]), ("second half", b[b.game_date >= d])):
        B.show(x, nm)
    B.show(b[b.fresh_same > 0], "teammate newly out")
    B.show(b[b.fresh_same == 0], "no new absence")

print("\nLEAK CHECK: scramble every outcome from a cutoff on; features on or before it must not move", flush=True)
P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
F0 = V3.absences(features(load(), P))
s2 = load()
cut = pd.Timestamp("2026-01-15")
k2 = s2.date >= cut
rng = np.random.default_rng(1)
for c in ("sog", "toi", "pbp_sog", "missed", "blocked_att", "goals", "pp_toi", "pp_att", "shifts", "assists", "pp_goals", "hits", "blocks"):
    s2.loc[k2, c] = rng.permutation(s2.loc[k2, c].to_numpy())
F1 = V3.absences(features(s2, P))
cols = [c for c in V3.TOI_COLS]
day = (F0.date <= cut).to_numpy()
moved = {c: float(np.nanmax(np.abs(F0.loc[day, c].to_numpy(float) - F1.loc[day, c].to_numpy(float)))) for c in cols}
print("  rows", int(day.sum()), "moved:", {c: v for c, v in moved.items() if v > 1e-9} or "none")
later = (F0.date > cut).to_numpy()
print("  control, absence features moved after it:", sum(float(np.nanmax(np.abs(F0.loc[later, c].to_numpy(float) - F1.loc[later, c].to_numpy(float)))) > 1e-9 for c in V3.ABSENT), "of", len(V3.ABSENT))
