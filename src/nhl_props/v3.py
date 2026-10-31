"""SOG v3: v2 + who is missing tonight + a real TOI model. Nothing here touches v2 or data/v2_frozen.joblib (the running paper test).

Usage: uv run python -m nhl_props.v3
Picks settings on 2022-23 -> 2024, fits on 2022-24, scores 2025 and 2026. Writes data/v3_pred.parquet and data/v3_frozen.joblib
(never rerun this during the v3 paper test).

Absences come from who dressed. No injury history exists for NHL, so this is the news as of warmups: fair against a CLOSE, not
an open (same rule as v2's LINEUP group). Every value of an absent or returning player is his own pre-game estimate from his last game.
fresh_* = played the team's last game, out tonight. vac_* = missed up to WINDOW team games. ret_* = back tonight after missing games.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nhl_props.baseline import DATA, fit_r, load, nb_loglik
from nhl_props.v2 import BASE, FIT, GROUPS, TEST, features, fit_model, fit_predict, score

WINDOW = 5  # team games a player can miss and still count as absent
GAP = 30  # days without a team game = new season, start over
ABS_TEAM = ["vac_toi_f", "vac_toi_d", "vac_mu", "vac_pp", "fresh_toi_f", "fresh_toi_d", "fresh_mu", "fresh_pp", "fresh_top",
            "ret_toi", "ret_mu", "ret_pp", "new_n"]
ABS_ROW = ["fresh_same", "fresh_above", "vac_same", "ret_same", "own_missed", "toi_rank", "pp_rank"]
ABS_OPP = ["opp_vac_toi_d", "opp_fresh_toi_d", "opp_vac_mu", "opp_fresh_mu"]
ABSENT = ABS_TEAM + ABS_ROW + ABS_OPP
TOI_COLS = ["toi_hat", "toi_h5", "toi_gap", "is_d", "rest", "games_d", "pp_hat", "home", "rest_t", "b2b_t", "gd_diff",
            "toi_sum", "toi_share", "pp_sum", "pp_share"] + ABSENT


def absences(s: pd.DataFrame) -> pd.DataFrame:
    """Rows must be in load() order. Walks each team's games in date order; state holds each player's last pre-game estimates."""
    d = s[["game_id", "team", "date", "player_id", "toi_hat", "mu0", "pp_hat", "is_d"]]
    tonight = d.groupby("date").player_id.agg(set).to_dict()
    teamof, team_rows, last = {}, [], {}
    row_out = {c: np.zeros(len(s)) for c in ABS_ROW}
    pos = pd.Series(np.arange(len(s)), index=s.index)
    games = d.groupby(["date", "game_id", "team"], sort=True)
    for (date, gid, team), x in games:
        st = last.setdefault(team, {"i": 0, "date": date, "seen": {}})
        if (date - st["date"]).days > GAP:
            st["seen"].clear()
        st["i"] += 1
        st["date"] = date
        i, seen, here = st["i"], st["seen"], set(x.player_id)
        v = dict.fromkeys(ABS_TEAM, 0.0)
        out = []
        for pid, e in list(seen.items()):
            if pid in here:
                continue
            if i - e["i"] > WINDOW + 1 or teamof.get(pid) != team or pid in tonight[date]:
                if i - e["i"] > WINDOW + 1 or pid in tonight[date]:
                    del seen[pid]
                continue
            fresh = e["i"] == i - 1
            k = "d" if e["is_d"] else "f"
            v[f"vac_toi_{k}"] += e["toi"]
            v["vac_mu"] += e["mu"]
            v["vac_pp"] += e["pp"]
            if fresh:
                v[f"fresh_toi_{k}"] += e["toi"]
                v["fresh_mu"] += e["mu"]
                v["fresh_pp"] += e["pp"]
                v["fresh_top"] = max(v["fresh_top"], e["mu"])
            out.append((e["is_d"], e["toi"], fresh))
        rows = x.sort_values("toi_hat", ascending=False)
        ranks = {0.0: 0, 1.0: 0}
        pp = x.pp_hat.rank(ascending=False, method="first")
        back = []
        for r in rows.itertuples():
            ranks[r.is_d] += 1
            p = pos[r.Index]
            row_out["toi_rank"][p] = ranks[r.is_d]
            row_out["pp_rank"][p] = pp[r.Index]
            e = seen.get(r.player_id)
            if e is None:
                v["new_n"] += 1
            elif e["i"] < i - 1:
                row_out["own_missed"][p] = i - 1 - e["i"]
                v["ret_toi"] += e["toi"]
                v["ret_mu"] += e["mu"]
                v["ret_pp"] += e["pp"]
                back.append((e["is_d"], e["toi"]))
            row_out["vac_same"][p] = sum(t for dd, t, _ in out if dd == r.is_d)
            row_out["fresh_same"][p] = sum(t for dd, t, f in out if dd == r.is_d and f)
            row_out["fresh_above"][p] = sum(t for dd, t, f in out if dd == r.is_d and f and t >= r.toi_hat)
            row_out["ret_same"][p] = sum(t for dd, t in back if dd == r.is_d)
        team_rows.append({"game_id": gid, "team": team, **v})
        for r in x.itertuples():
            seen[r.player_id] = {"i": i, "toi": r.toi_hat, "mu": r.mu0, "pp": r.pp_hat, "is_d": r.is_d}
            teamof[r.player_id] = team
    t = pd.DataFrame(team_rows)
    o = t[["game_id", "team", "vac_toi_d", "fresh_toi_d", "vac_mu", "fresh_mu"]].rename(
        columns={"team": "opp", "vac_toi_d": "opp_vac_toi_d", "fresh_toi_d": "opp_fresh_toi_d", "vac_mu": "opp_vac_mu", "fresh_mu": "opp_fresh_mu"})
    j = s[["game_id", "team", "opp"]].merge(t, on=["game_id", "team"], how="left").merge(o, on=["game_id", "opp"], how="left").set_axis(s.index)
    for c in ABS_TEAM + ABS_OPP:
        s[c] = j[c].fillna(0.0)
    for c, a in row_out.items():
        s[c] = a
    return s


def fit_toi(X, y, base):
    m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=300, l2_regularization=10.0, random_state=0)
    return m.fit(X, y - base)


def main() -> None:
    P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
    s = absences(features(load(), P))
    print("features done", flush=True)
    y = s.sog.to_numpy(float)
    ok = (np.isfinite(s.mu0) & (s.mu0 > 0)).to_numpy()
    sea = s.season.to_numpy()
    mu0 = s.mu0.to_numpy()

    # TOI model. Training rows for the SOG model get out-of-fold predictions, or it would trust TOI too much.
    def toi_for(train, pred):
        m = fit_toi(s.loc[train, TOI_COLS], s.toi.to_numpy()[train], s.toi_hat.to_numpy()[train])
        return s.toi_hat.to_numpy()[pred] + m.predict(s.loc[pred, TOI_COLS])

    s["toi_m"] = s.toi_hat
    for season in (2022, 2023, 2024):
        tr = np.isin(sea, [x for x in (2021, *FIT) if x != season]) & ok
        pr = (sea == season) & ok
        s.loc[pr, "toi_m"] = toi_for(tr, pr)
    tr = np.isin(sea, (2021, *FIT)) & ok
    pr = np.isin(sea, TEST) & ok
    toi_final = fit_toi(s.loc[tr, TOI_COLS], s.toi.to_numpy()[tr], s.toi_hat.to_numpy()[tr])
    s.loc[pr, "toi_m"] = s.toi_hat.to_numpy()[pr] + toi_final.predict(s.loc[pr, TOI_COLS])
    for label, k in (("validation folds 2022-24", np.isin(sea, FIT) & ok), ("2025", (sea == 2025) & ok), ("2026", (sea == 2026) & ok)):
        t, h, m = s.toi[k], s.toi_hat[k], s.toi_m[k]
        fr = (s.fresh_same[k] > 0).to_numpy()
        print(f"TOI {label}: MAE average {np.abs(t - h).mean():.3f} -> model {np.abs(t - m).mean():.3f}; with a same-position teammate newly out "
              f"({fr.mean():.0%}) {np.abs(t - h)[fr].mean():.3f} -> {np.abs(t - m)[fr].mean():.3f}, bias {(t - h)[fr].mean():+.3f} -> {(t - m)[fr].mean():+.3f}", flush=True)

    s["toi_ratio"] = s.toi_m / s.toi_hat
    v2_cols = BASE + sum(GROUPS.values(), [])
    sets = {"v2": v2_cols, "+absent": v2_cols + ABSENT, "+toi model": v2_cols + ["toi_m", "toi_ratio"], "v3 all": v2_cols + ABSENT + ["toi_m", "toi_ratio"]}
    cfg = {"iter": 500, "depth": 3, "leaf": 1000}  # v2's choice, not re-tuned
    tr0 = np.isin(sea, (2022, 2023)) & ok
    va = (sea == 2024) & ok
    print(f"2024 validation, SOG, {cfg}:", flush=True)
    best = None
    for name, cols in sets.items():
        f = fit_predict(s.loc[tr0, cols], y[tr0], mu0[tr0], s.loc[va, cols], cfg)
        ll, mae, _ = score(y[va], mu0[va] * f)
        print(f"  {name:11s} ll {ll:.4f} mae {mae:.4f}", flush=True)
        if name != "v2" and (best is None or ll > best[0]):
            best = (ll, name, cols)
    print("chosen", best[1], flush=True)
    cols = best[2]
    r = fit_r(y[va], mu0[va] * fit_predict(s.loc[tr0, cols], y[tr0], mu0[tr0], s.loc[va, cols], cfg))
    fit = np.isin(sea, FIT) & ok
    test = np.isin(sea, TEST) & ok
    d = s[test].copy()
    sog_final = fit_model(s.loc[fit, cols], y[fit], mu0[fit], cfg)
    joblib.dump({"model": sog_final, "cols": cols, "toi": toi_final, "r": r, "cfg": cfg}, DATA / "v3_frozen.joblib")
    d["mu_base"], d["mu"], d["r"] = d.mu0, d.mu0 * sog_final.predict(s.loc[test, cols]), r
    d["mu_v2"] = d.mu0 * fit_predict(s.loc[fit, v2_cols], y[fit], mu0[fit], s.loc[test, v2_cols], cfg)
    d["mu_true_toi"] = np.nan
    for season in TEST:
        x = d[d.season == season]
        yy = x.sog.to_numpy(float)
        print(f"{season}: v2 ll {nb_loglik(yy, x.mu_v2.to_numpy(), r).mean():.4f} mae {np.abs(x.mu_v2 - yy).mean():.4f} | "
              f"v3 ll {nb_loglik(yy, x.mu.to_numpy(), r).mean():.4f} mae {np.abs(x.mu - yy).mean():.4f} bias {x.mu.mean() - yy.mean():+.3f}, r {r:.1f}", flush=True)
    keep = ["game_id", "season", "game_date", "start_utc", "player_id", "name", "team", "opp", "grp", "toi", "sog", "lg", "ratio", "toi_hat", "toi_m",
            "mu", "mu_true_toi", "r", "mu_base", "mu_v2", "fresh_same"]
    d[keep].to_parquet(DATA / "v3_pred.parquet")


if __name__ == "__main__":
    main()
