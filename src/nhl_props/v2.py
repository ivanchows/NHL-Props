"""SOG v2: a gradient-boosted correction on top of the frozen baseline. Baseline mu x exp(f(features)), negative binomial.

Usage: uv run python -m nhl_props.v2
Picks settings on 2022-23 -> 2024, fits on 2022-24, scores 2025 and 2026 (never used for any choice).
Writes data/v2_pred.parquet: same columns as baseline_pred.parquet, plus mu_nolineup, and data/v2_frozen.joblib (the
no-lineup model the live paper test uses: never refit it during the test).

No lookahead: every player, team and opponent feature uses earlier dates only.
The LINEUP group is the exception: it uses who dressed tonight, known at lines and warmups, so it is fair against a CLOSE
and NOT against an open. mu_nolineup leaves that group out: use it for anything priced before lineups are out.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nhl_props.baseline import DATA, decay_sums, fit_r, league_rate, load, nb_loglik, rate_ratio, toi_pred

FIT = (2022, 2023, 2024)
TEST = (2025, 2026)
GROUPS = {
    "attempts": ["att_r30", "att_r120", "att_r480", "acc", "sog_r30", "sog_r120"],
    "pp": ["pp_hat", "pp_rate"],
    "form": ["toi_h5", "toi_gap", "rest", "games_d"],
    "opp": ["opp_allow", "opp_allow_all", "team_own"],
    "game": ["home", "gd", "gd_opp", "gd_diff", "rest_t", "rest_o", "b2b_t", "b2b_o"],
    "lineup": ["toi_sum", "toi_share", "pp_sum", "pp_share", "mu_sum"],
}
BASE = ["log_mu", "ratio", "toi_hat", "is_d"]


def team_features(s: pd.DataFrame, lg: np.ndarray) -> pd.DataFrame:
    """Decayed team and opponent rates, on the rows' game. All from earlier dates."""
    s = s.assign(expg=lg * s.toi, is_d=(s.grp == "D"))
    tg = s.groupby(["team", "opp", "date", "game_id"]).agg(sog=("sog", "sum"), exp=("expg", "sum"), goals=("goals", "sum")).reset_index()
    tg = tg.merge(tg[["team", "game_id", "goals"]].rename(columns={"team": "opp", "goals": "ga"}), on=["opp", "game_id"])
    tg["gd"] = tg.goals - tg.ga
    tg = tg.sort_values(["team", "date"], ignore_index=True)
    tid = tg.team.astype("category").cat.codes.to_numpy()
    t = (tg.date - tg.date.min()).dt.days.to_numpy(float)
    a_s, a_e, a_g, n = decay_sums(tid, t, [tg.sog.to_numpy(float), tg.exp.to_numpy(), tg.gd.to_numpy(float)], 40)
    tg["team_own"] = (a_s + 20) / (a_e + 20)
    tg["gd"] = a_g / (n + 3)
    tg["rest_t"] = tg.groupby("team").date.diff().dt.days.clip(upper=7).fillna(7)
    tg["b2b_t"] = (tg.rest_t == 1).astype(float)

    # Shots the opponent allows, to this position group and overall.
    ag = s.groupby(["opp", "is_d", "date", "game_id"]).agg(sog=("sog", "sum"), exp=("expg", "sum")).reset_index().sort_values(["opp", "is_d", "date"], ignore_index=True)
    oid = (ag.opp.astype("category").cat.codes * 2 + ag.is_d.astype(int)).to_numpy()
    t = (ag.date - ag.date.min()).dt.days.to_numpy(float)
    a_s, a_e, _ = decay_sums(oid, t, [ag.sog.to_numpy(float), ag.exp.to_numpy()], 40)
    ag["opp_allow"] = (a_s + 10) / (a_e + 10)
    al = ag.groupby(["opp", "date", "game_id"]).agg(sog=("sog", "sum"), exp=("exp", "sum")).reset_index().sort_values(["opp", "date"], ignore_index=True)
    a_s, a_e, _ = decay_sums(al.opp.astype("category").cat.codes.to_numpy(), (al.date - al.date.min()).dt.days.to_numpy(float),
                             [al.sog.to_numpy(float), al.exp.to_numpy()], 40)
    al["opp_allow_all"] = (a_s + 20) / (a_e + 20)

    me = tg[["team", "game_id", "team_own", "gd", "rest_t", "b2b_t"]]
    op = tg[["team", "game_id", "gd", "rest_t", "b2b_t"]].rename(columns={"team": "opp", "gd": "gd_opp", "rest_t": "rest_o", "b2b_t": "b2b_o"})
    out = s[["team", "opp", "game_id", "is_d"]].merge(me, on=["team", "game_id"], how="left").merge(op, on=["opp", "game_id"], how="left")
    out = out.merge(ag[["opp", "is_d", "game_id", "opp_allow"]], on=["opp", "is_d", "game_id"], how="left")
    out = out.merge(al[["opp", "game_id", "opp_allow_all"]], on=["opp", "game_id"], how="left")
    assert len(out) == len(s)
    return out.drop(columns=["team", "opp", "game_id", "is_d"]).set_axis(s.index)


def features(s: pd.DataFrame, P: dict) -> pd.DataFrame:
    pid, t = s.player_id.to_numpy(), s.t.to_numpy()
    lg = league_rate(s, P["league_window"])
    toi0 = s.grp.map(P["toi0"]).to_numpy()
    s["lg"] = lg
    s["toi_hat"] = toi_pred(s, pid, t, P["toi_half_life"], P["toi_shrink"], toi0)
    s["ratio"] = rate_ratio(s, pid, t, P["rate_half_life"], P["rate_shrink"], lg)
    s["mu0"] = s.ratio * s.lg * s.toi_hat * P["scale"]
    s["log_mu"] = np.log(s.mu0)
    s["is_d"] = (s.grp == "D").astype(float)
    s["home"] = s.home.astype(float)

    # Attempts settle faster than shots on goal. Same ratio form, against the league's attempt rate.
    att = (s.pbp_sog + s.missed + s.blocked_att).to_numpy(float)
    d = s.assign(att=att).groupby(["grp", "date"])[["att", "toi"]].sum().reset_index().sort_values("date")
    d["lga"] = d.groupby("grp").att.transform(lambda x: x.rolling(P["league_window"], min_periods=1).sum().shift(1))
    d["lgt"] = d.groupby("grp").toi.transform(lambda x: x.rolling(P["league_window"], min_periods=1).sum().shift(1))
    d["lga"] = (d.lga / d.lgt).groupby(d.grp).bfill()
    lga = s[["grp", "date"]].merge(d[["grp", "date", "lga"]], on=["grp", "date"], how="left")["lga"].to_numpy()
    exp_att = s.toi.to_numpy() * lga
    for h in (30, 120, 480):
        a, e, _ = decay_sums(pid, t, [att, exp_att], h)
        s[f"att_r{h}"] = (a + 20) / (e + 20)
    a, sg, _ = decay_sums(pid, t, [att, s.sog.to_numpy(float)], 240)
    s["acc"] = (sg + 10) / (a + 25)
    s["sog_r30"] = rate_ratio(s, pid, t, 30, 5, lg)
    s["sog_r120"] = rate_ratio(s, pid, t, 120, 10, lg)

    # Power-play role. Games without shift charts carry no weight.
    w = s.pp_toi.notna().to_numpy(float)
    pv = s.pp_toi.fillna(0).to_numpy(float)
    a_p, a_w, _ = decay_sums(pid, t, [pv, w], 20)
    s["pp_hat"] = a_p / (a_w + 0.5)
    a_p, a_a, _ = decay_sums(pid, t, [pv, s.pp_att.to_numpy(float) * w], 240)
    s["pp_rate"] = (a_a + 2) / (a_p + 5)

    # Form and availability.
    s["toi_h5"] = toi_pred(s, pid, t, 5, 0.5, toi0)
    s["toi_gap"] = s.toi_h5 - s.toi_hat
    s["rest"] = s.groupby("player_id").date.diff().dt.days.clip(upper=30).fillna(30)
    s["games_d"] = decay_sums(pid, t, [], 60)[0]

    tf = team_features(s, lg)
    for c in tf:
        s[c] = tf[c]
    s["gd_diff"] = s.gd - s.gd_opp

    # Tonight's lineup: everyone who dressed, on what they usually play.
    g = s.groupby(["game_id", "team"])
    s["toi_sum"] = g.toi_hat.transform("sum")
    s["toi_share"] = s.toi_hat * 300 / s.toi_sum
    s["pp_sum"] = g.pp_hat.transform("sum")
    s["pp_share"] = s.pp_hat / s.pp_sum.clip(lower=1)
    s["mu_sum"] = g.mu0.transform("sum")
    return s


def fit_model(X, y, mu0, cfg):
    """Poisson boosting on y / mu0, weighted by mu0: the same as an offset of log mu0."""
    m = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.03, max_iter=cfg["iter"], max_depth=cfg["depth"],
                                      min_samples_leaf=cfg["leaf"], l2_regularization=10.0, random_state=0)
    m.fit(X, y / mu0, sample_weight=mu0)
    return m


def fit_predict(X, y, mu0, Xp, cfg):
    return fit_model(X, y, mu0, cfg).predict(Xp)


def score(y, mu, r=None):
    r = r or fit_r(y, mu)
    return nb_loglik(y, mu, r).mean(), np.abs(mu - y).mean(), r


def main() -> None:
    P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
    s = features(load(), P)
    y = s.sog.to_numpy(float)
    ok = np.isfinite(s.mu0).to_numpy() & (s.mu0 > 0).to_numpy()
    tr0 = s.season.isin((2022, 2023)).to_numpy() & ok
    va = (s.season == 2024).to_numpy() & ok
    fit = s.season.isin(FIT).to_numpy() & ok
    test = s.season.isin(TEST).to_numpy() & ok
    allf = BASE + sum(GROUPS.values(), [])
    mu0 = s.mu0.to_numpy()
    X = lambda cols, mask: s.loc[mask, cols]

    ll0, mae0, r0 = score(y[va], mu0[va])
    print(f"2024 validation: baseline ll {ll0:.4f} mae {mae0:.4f}", flush=True)
    best = None
    for cfg in ({"iter": 150, "depth": 3, "leaf": 500}, {"iter": 300, "depth": 3, "leaf": 500},
                {"iter": 300, "depth": 4, "leaf": 500}, {"iter": 500, "depth": 3, "leaf": 1000}):
        f = fit_predict(X(allf, tr0), y[tr0], mu0[tr0], X(allf, va), cfg)
        ll, mae, _ = score(y[va], mu0[va] * f)
        print(f"  {cfg}: ll {ll:.4f} mae {mae:.4f}", flush=True)
        if best is None or ll > best[0]:
            best = (ll, cfg)
    cfg = best[1]
    print(f"chosen {cfg}")
    for g in GROUPS:
        cols = [c for c in allf if c not in GROUPS[g]]
        f = fit_predict(X(cols, tr0), y[tr0], mu0[tr0], X(cols, va), cfg)
        ll, mae, _ = score(y[va], mu0[va] * f)
        print(f"  without {g:9s} ll {ll:.4f} mae {mae:.4f}", flush=True)

    # Final: settings fixed above. r comes from the in-sample fit, shrunk to the validation r (in-sample mu is too sharp).
    nol = [c for c in allf if c not in GROUPS["lineup"]]
    f_va = fit_predict(X(allf, tr0), y[tr0], mu0[tr0], X(allf, va), cfg)
    r = fit_r(y[va], mu0[va] * f_va)
    f_te = fit_predict(X(allf, fit), y[fit], mu0[fit], X(allf, test), cfg)
    m_nl = fit_model(X(nol, fit), y[fit], mu0[fit], cfg)
    f_nl = m_nl.predict(X(nol, test))
    joblib.dump({"model": m_nl, "cols": nol, "r": r, "cfg": cfg}, DATA / "v2_frozen.joblib")
    print(f"NB r {r:.1f} (baseline {P['nb_r']:.1f})")
    d = s[test].copy()
    d["mu_base"], d["mu"], d["mu_nolineup"], d["r"] = d.mu0, d.mu0 * f_te, d.mu0 * f_nl, r
    d["mu_true_toi"] = np.nan
    for season in TEST:
        x = d[d.season == season]
        yy = x.sog.to_numpy(float)
        print(f"\n{season}: {len(x):,} rows, mean sog {yy.mean():.3f}")
        for lab, col, rr in (("baseline", "mu_base", P["nb_r"]), ("v2 no lineup", "mu_nolineup", r), ("v2", "mu", r)):
            print(f"  {lab:13s} ll {nb_loglik(yy, x[col].to_numpy(), rr).mean():.4f} mae {np.abs(x[col] - yy).mean():.4f} bias {x[col].mean() - yy.mean():+.3f}")
    keep = ["game_id", "season", "game_date", "start_utc", "player_id", "name", "team", "opp", "grp", "toi", "sog", "lg", "ratio",
            "toi_hat", "mu", "mu_true_toi", "r", "mu_base", "mu_nolineup"]
    d[keep].to_parquet(DATA / "v2_pred.parquet")


if __name__ == "__main__":
    main()
