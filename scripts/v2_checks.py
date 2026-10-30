import numpy as np, pandas as pd
# Stress checks on v2 vs PTO closes. Usage: uv run python scripts/v2_checks.py
from nhl_props import pto_backtest as B
from scipy.stats import nbinom
pd.set_option("display.width", 200)
a = B.lines(); p = B.model_rows("v2_pred")
extra = pd.read_parquet(B.DATA / "v2_pred.parquet")[["game_id","player_id","mu_nolineup","mu_base"]]
pp = pd.read_parquet(B.DATA / "v2_pred.parquet")[["game_date","player_id","mu_nolineup","mu_base"]]
p = p.merge(pp, on=["game_date","player_id"], how="left")
m = B.attach(a, p); ml = B.main_lines(m); ml = ml[ml.books >= 3].copy()
ml["p_nl"] = nbinom.sf(np.floor(ml.line).astype(int), ml.r, ml.r/(ml.r+ml.mu_nolineup)).clip(.01,.99)
print("log loss no-lineup", round(B.logloss(ml.y, ml.p_nl),4), "full", round(B.logloss(ml.y, ml.p_over),4), "cons", round(B.logloss(ml.y, ml.cons),4), "50/50 nl", round(B.logloss(ml.y,(ml.p_nl+ml.cons)/2),4))
print("\nNO LINEUP at best price"); 
for lo,hi in ((.03,.06),(.06,.12),(.12,1),(.06,1)): B.show(B.bets(ml, ml.p_nl, lo, hi), f"edge {lo:.0%}-{hi:.0%}")
b = B.bets(ml, ml.p_over, .06, 1)
print("\nFULL v2 edge 6%+ splits"); B.show(b,"all")
for k,x in b.groupby(b.game_date>="2026-09-01"): B.show(x,"2026-27" if k else "2025-26")
for k,x in b.groupby("side"): B.show(x,k)
for k,x in b.groupby("line"): 
    if len(x)>50: B.show(x,f"line {k}")
for k,x in b.groupby("month"): B.show(x,k)
print("\nMAINSTREAM best price only (DK FD MGM CZR b365 Fanatics theScore BetRivers)")
s = pd.read_parquet(B.DATA/"pto/SOG.parquet", columns=["game_id","player_id_pto","line","side","book","odds"])
s = s[s.book.isin(["DRAFT_KINGS","FAN_DUEL","MGM","CAESARS","BET_365","FANATICS","THE_SCORE","BET_RIVERS"])]
bst = s.groupby(["game_id","player_id_pto","line","side"]).odds.max().unstack()
j = ml[["game_id","player_id_pto","line"]].join(bst, on=["game_id","player_id_pto","line"]); j.index = ml.index
for lo,hi in ((.03,.06),(.06,.12),(.12,1),(.06,1)):
    B.show(B.bets(ml, ml.p_over, lo, hi, j["over"].fillna(-99999), j["under"].fillna(-99999)), f"edge {lo:.0%}-{hi:.0%}")
print("\nBy model - consensus gap, best price (no edge filter beyond gap)")
gap = ml.p_over - ml.cons
for lo,hi in ((.02,.04),(.04,.06),(.06,.08),(.08,1)):
    k = gap.abs().between(lo,hi,inclusive="left")
    d = ml[k]; g = gap[k]
    won = np.where(g>0,d.y==1,d.y==0); odds = np.where(g>0,d.best_over,d.best_under)
    keep = (odds>=-250)&(odds<=250); net,risk = B.payoff(odds[keep],won[keep])
    print(f"  gap {lo:.0%}-{hi:.0%}: {keep.sum():5d} bets ROI {net.sum()/risk.sum():+.1%}  avg best-price edge vs consensus {np.mean(np.where(g>0,d.cons-B.implied(d.best_over),1-d.cons-B.implied(d.best_under))[keep]):+.3f}")
print("\nPLACEBO: shuffle (model logit - cons logit) across rows, 200 runs, edge 6%+ ROI")
rng = np.random.default_rng(0); dl = (B.logit(ml.p_over)-B.logit(ml.cons)).to_numpy(); c = B.logit(ml.cons).to_numpy(); r=[]
for i in range(200):
    q = pd.Series(1/(1+np.exp(-(c+rng.permutation(dl)))), index=ml.index)
    x = B.bets(ml, q, .06, 1); r.append(x.net.sum()/x.risk.sum())
r=np.array(r); print(f"  mean {r.mean():+.1%} sd {r.std():.1%} max {r.max():+.1%}; real {b.net.sum()/b.risk.sum():+.1%}; share of shuffles >= real {np.mean(r>=b.net.sum()/b.risk.sum()):.3f}")
