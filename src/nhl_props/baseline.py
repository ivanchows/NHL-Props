"""Frozen SOG baseline: shot rate relative to the league x projected TOI, negative binomial.

Usage: uv run python -m nhl_props.baseline
Tunes on 2022-2024 (2021 = warm-up history only), scores 2025 (2025-26) and 2026 (2026-27 so far).
Writes data/baseline_params.json and data/baseline_pred.parquet (one row per skater-game: mu, r).

No lookahead: every feature uses games on earlier dates only.
- League rate: SOG per minute by position group (F/D) over the previous W game-days, so the falling
  league shot volume (31.6 -> 27.3 team SOG/game, 2021 -> 2026) is tracked instead of lagged.
- Player ratio: decayed SOG / decayed expected SOG (toi x league rate at the time), shrunk to 1 with K
  expected shots.
- TOI: decayed mean TOI, shrunk to the position's debut TOI with M games.
- SOG | mu ~ NegBin(mean mu, size R).
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TUNE = (2022, 2023, 2024)
TEST = (2025, 2026)


def load() -> pd.DataFrame:
    s = pd.read_parquet(DATA / "skater_games.parquet")
    s["grp"] = np.where(s.pos == "D", "D", "F")
    s["date"] = pd.to_datetime(s.game_date)
    s["t"] = (s.date - s.date.min()).dt.days.astype(float)
    return s.sort_values(["player_id", "date", "game_id"], ignore_index=True)


def league_rate(s: pd.DataFrame, w: int) -> pd.Series:
    """SOG per minute by group over the previous w game-days (dates strictly before each row's date)."""
    d = s.groupby(["grp", "date"])[["sog", "toi"]].sum().reset_index().sort_values("date")
    out = []
    for g, dd in d.groupby("grp"):
        roll = dd[["sog", "toi"]].rolling(w, min_periods=1).sum().shift(1)
        # The first date in the data has no history: give it the next date's rate, or its NaN poisons
        # every later decayed sum of the players who played that night.
        out.append(pd.DataFrame({"grp": g, "date": dd.date, "lg": (roll.sog / roll.toi).bfill()}))
    lg = pd.concat(out)
    return s[["grp", "date"]].merge(lg, on=["grp", "date"], how="left")["lg"].to_numpy()


def decay_sums(pid: np.ndarray, t: np.ndarray, xs: list[np.ndarray], h: float) -> list[np.ndarray]:
    """For each row: sum over the same player's EARLIER rows of x * 0.5 ** ((t_now - t_i) / h).
    Rows must be sorted by player, then date. Returns one array per x (plus the decayed count last)."""
    n = len(pid)
    xs = xs + [np.ones(n)]
    acc = [np.zeros(n) for _ in xs]
    run = [0.0] * len(xs)
    for i in range(n):
        if i == 0 or pid[i] != pid[i - 1]:
            run = [0.0] * len(xs)
        else:
            f = 0.5 ** ((t[i] - t[i - 1]) / h)
            run = [(r + x[i - 1]) * f for r, x in zip(run, xs)]
        for a, r in zip(acc, run):
            a[i] = r
    return acc


def nb_loglik(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def fit_r(y, mu) -> float:
    grid = np.exp(np.linspace(np.log(1), np.log(200), 120))
    return float(grid[np.argmax([nb_loglik(y, mu, r).mean() for r in grid])])


def toi_pred(s, pid, t, h, m, toi0):
    sw_toi, sw = decay_sums(pid, t, [s.toi.to_numpy()], h)
    return (sw_toi + m * toi0) / (sw + m)


def rate_ratio(s, pid, t, h, k, lg):
    exp_sog = s.toi.to_numpy() * lg
    s_sog, s_exp, _ = decay_sums(pid, t, [s.sog.to_numpy().astype(float), exp_sog], h)
    return (s_sog + k) / (s_exp + k)


def main() -> None:
    s = load()
    pid, t = s.player_id.to_numpy(), s.t.to_numpy()
    y = s.sog.to_numpy().astype(float)
    tune = s.season.isin(TUNE).to_numpy()
    test = s.season.isin(TEST).to_numpy()

    # Debut TOI per group, from tune-period debuts only.
    first = s.groupby("player_id").cumcount().to_numpy() == 0
    toi0 = s[first & tune].groupby("grp").toi.mean()
    toi0_row = s.grp.map(toi0).to_numpy()

    # 1) TOI: tune half-life (days) and shrink games on TOI squared error.
    best = None
    for h, m in product([10, 20, 40, 80, 160], [0.5, 1, 2, 4]):
        p = toi_pred(s, pid, t, h, m, toi0_row)
        err = np.mean((p[tune] - s.toi.to_numpy()[tune]) ** 2)
        if best is None or err < best[0]:
            best = (err, h, m)
    _, toi_h, toi_m = best
    toi_hat = toi_pred(s, pid, t, toi_h, toi_m, toi0_row)
    print(f"TOI: half-life {toi_h} d, shrink {toi_m} games, tune RMSE {best[0] ** 0.5:.2f} min", flush=True)

    # 2) Rate: league window W, half-life H, shrink K; plus one overall scale, fit on tune.
    best = None
    for w in (10, 20, 40):
        lg = league_rate(s, w)
        for h, k in product([60, 120, 240, 480, 960], [5, 10, 20, 40]):
            ratio = rate_ratio(s, pid, t, h, k, lg)
            mu = ratio * lg * toi_hat
            ok = tune & np.isfinite(mu)
            scale = y[ok].sum() / mu[ok].sum()
            r = fit_r(y[ok], mu[ok] * scale)
            ll = nb_loglik(y[ok], mu[ok] * scale, r).mean()
            if best is None or ll > best[0]:
                best = (ll, w, h, k, scale, r)
    ll, W, H, K, SCALE, R = best
    print(f"rate: W {W} game-days, half-life {H} d, K {K}, scale {SCALE:.3f}, NB r {R:.1f}, tune ll {ll:.4f}", flush=True)

    lg = league_rate(s, W)
    ratio = rate_ratio(s, pid, t, H, K, lg)
    s["lg"], s["ratio"], s["toi_hat"] = lg, ratio, toi_hat
    s["mu"] = ratio * lg * toi_hat * SCALE
    s["mu_true_toi"] = ratio * lg * s.toi * SCALE  # diagnostic only: uses the actual TOI
    s["r"] = R

    params = dict(toi_half_life=toi_h, toi_shrink=toi_m, toi0=toi0.to_dict(), league_window=W,
                  rate_half_life=H, rate_shrink=K, scale=SCALE, nb_r=R, tune=TUNE)
    (DATA / "baseline_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    keep = ["game_id", "season", "game_date", "start_utc", "player_id", "name", "team", "opp", "grp",
            "toi", "sog", "lg", "ratio", "toi_hat", "mu", "mu_true_toi", "r"]
    s.loc[test, keep].to_parquet(DATA / "baseline_pred.parquet")
    report(s)


def report(s: pd.DataFrame) -> None:
    s = s.sort_values(["player_id", "date", "game_id"])
    # Naive: the player's season-to-date mean SOG (league mean of the group before his first game).
    g = s.groupby(["player_id", "season"]).sog
    s["naive"] = (g.cumsum() - s.sog) / g.cumcount().replace(0, np.nan)
    s["naive"] = s.naive.fillna(s.groupby("grp").sog.transform("mean"))
    for season in TEST:
        d = s[s.season == season]
        y, mu, r = d.sog.to_numpy(), d.mu.to_numpy(), d.r.iloc[0]
        print(f"\n{season}: {len(d)} skater-games, {d.game_id.nunique()} games")
        print(f"  mean SOG {y.mean():.3f}, mean mu {mu.mean():.3f} (bias {mu.mean() - y.mean():+.3f})")
        print(f"  MAE model {np.abs(mu - y).mean():.3f} | true-TOI {np.abs(d.mu_true_toi - y).mean():.3f}"
              f" | season-to-date {np.abs(d.naive - y).mean():.3f} | constant {np.abs(y.mean() - y).mean():.3f}")
        print(f"  NB log-lik model {nb_loglik(y, mu, r).mean():.4f}"
              f" | season-to-date {nb_loglik(y, np.maximum(d.naive, 0.05).to_numpy(), r).mean():.4f}")
        print(f"  TOI MAE {np.abs(d.toi_hat - d.toi).mean():.2f} min, bias {(d.toi_hat - d.toi).mean():+.2f}")
        from scipy.stats import nbinom
        p = r / (r + mu)
        for line in (0.5, 1.5, 2.5, 3.5):
            pov = nbinom.sf(int(line), r, p)
            bins = pd.cut(pov, [0, .2, .35, .5, .65, .8, 1])
            cal = pd.DataFrame({"p": pov, "hit": y > line}).groupby(bins, observed=True).agg(
                n=("hit", "size"), pred=("p", "mean"), actual=("hit", "mean"))
            print(f"  over {line}: pred {pov.mean():.3f} actual {(y > line).mean():.3f}")
            print("   " + cal.round(3).to_string().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
