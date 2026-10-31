# NHL player props: shots on goal

Projects skater shots on goal (SOG) from free NHL data and tests the numbers against real sportsbook and exchange
prices. Every feature uses only earlier games. **No real bets so far.** The projections alone do not beat the closing
market; the best result needs the best price across books and is still in paper testing. Goalie saves data is
collected, but no saves model exists yet.

## Results (2025-26 held out, settings chosen on 2022-24)

| Model | MAE (SOG) | NegBin log-lik |
|---|---|---|
| Constant | 1.172 | |
| Season-to-date mean | 1.045 | -1.583 |
| Baseline (rate x projected TOI) | 1.015 | -1.536 |
| v2 (boosted correction on baseline) | 1.011 | -1.5308 |
| v3 (v2 + TOI model) | not recorded | -1.5306 |

v3 also cuts TOI MAE from 1.927 to 1.814 minutes (2025-26) and from 1.976 to 1.870 (2026-27 so far).

**Closing lines** (PTO history of every book, 2025-26 + 2026-27 through Oct 29, 2026, main line per player-game).
Log loss: consensus .6828, baseline .6863, v2 .6833, v3 .6831, 50/50 v3 + consensus .6819.

| Bets at edge 6%+ | Bets | Best price | 2nd best price | Median book |
|---|---|---|---|---|
| v2 | 1,849 | +5.8% | +3.2% | +0.2% |
| v3 | 1,820 | +6.8% ± 2.2 | +4.2% | +1.3% |
| Control: every under | | -2.3% ± 0.7 | | |
| Control: every over | | -3.6% ± 0.7 | | |

ROI = net / risked (plus odds risk 1 unit, minus odds risk to win 1 unit), odds -250..+250, SEs clustered by game.

**Novig open** (2026-27 only, through Oct 28, 2026; v2 without lineup features): edge 6%+, 472 bets, ROI +11.7% ± 4.4,
net CLV +1.18 ± 0.28 pts of implied probability. Log loss model .6798, market .6789, 50/50 .6762. At the close +6.6%
± 5.1. Prices are last trades, not quotes: in a stricter test 239 of 462 bets had only that side traded.

**Does it beat the market?**

- Log loss: no. The model alone loses at the open and the close. A 50/50 blend beats both.
- CLV: positive at the Novig open, on about one month of data. PTO history has closes only.
- ROI: mostly line shopping. At the median book v2 is about break-even and v3 is +1.3%.
- Stability: v2 by month runs from Feb -20.1% to Apr +17.7%. v3 first half +3.8% ± 3.3, second half +9.6% ± 3.1.
- Part of the edge is books lagging falling shot volume, which may fade. Signal at about 2.4 SE: not proven.

## Setup

uv project, Python 3.12. `data/` and `logs/` are gitignored, so raw data and the frozen models are not in git.

```bash
uv sync
```

## Run everything

Public data and models:

```bash
uv run python -m nhl_props.fetch 2021 2026   # box, play-by-play, shifts; resumes; slow, the API throttles
uv run python -m nhl_props.data              # data/skater_games.parquet, data/goalie_games.parquet
uv run python -m nhl_props.baseline          # frozen baseline, scores 2025 + 2026
uv run python -m nhl_props.v2                # data/v2_pred.parquet, data/v2_frozen.joblib
uv run python -m nhl_props.v3                # data/v3_pred.parquet, data/v3_frozen.joblib
```

Do not rerun `v2` or `v3` during a paper test: they overwrite the frozen models. Market data needs access that is not
in this repo: `nhl_props.novig` imports a signed Novig client and key; PTO needs a logged-in session and a browser extension.

```bash
uv run python -m nhl_props.novig markets     # then `novig trades`
uv run python -m nhl_props.pto serve         # receiver for scripts/pto_pull.js, run in a PTO tab
uv run python -m nhl_props.pto build         # data/pto/SOG.parquet, data/pto/SAVES.parquet
uv run python -m nhl_props.novig_backtest v2_pred mu_nolineup
uv run python -m nhl_props.pto_backtest v2_pred
uv run python scripts/v2_checks.py           # no-lineup, mainstream books, placebo, splits
uv run python scripts/v3_checks.py           # v3 vs v2 on the same closes, leak check
```

## Daily live commands (paper only)

```bash
uv run python -m nhl_props.live decide           # v2 at Novig, every 30 min; --dry-run to test
uv run python -m nhl_props.live_v3 decide        # v3 at Novig, every 30 min
uv run python -m nhl_props.pto_live decide       # v3 at PTO's best price; needs Chrome and the extension
uv run python -m nhl_props.fetch recent          # nightly, then `nhl_props.data`
uv run python -m nhl_props.live settle           # nightly, also `live_v3 settle` and `pto_live settle`
```

`scripts/*.cmd` are Task Scheduler wrappers for these, tied to one machine.

**State:** all three tests started Oct 30, 2026 with rules fixed first, and have no results yet. Edge 6%+, one side per
player-game, judged at about 500 bets per stage, net CLV first. Tests are never pooled. v2 logs and grades only. The
first real PTO scan had 0 rows because every game had started. `v3.py`, `live_v3.py`, `pto_live.py` and
`scripts/v3_checks.py` run locally and were not committed when this was written.

## How it works

- **Baseline:** mu = player ratio x league rate x projected TOI, SOG ~ NegBin(mu, r = 17.3). League rate tracks the fall
  from 31.6 to 27.3 team SOG/game.
- **v2:** boosted Poisson correction on the baseline (attempts, form, opponent, team tendency, rest, home, lineup).
- **v3:** adds a TOI model that sees who is missing. Live lineups = DailyFaceoff minus ESPN's out list.
- **Protocol:** tune on 2022-24, score 2025 and 2026 with no choice made on them. Props settle on the box score.

## Tested and rejected

- **Baseline at the close:** +0.5% ± 1.2 at the best price, and every single book loses.
- **Better TOI alone:** even the true TOI improves MAE by only 0.018. Most error is shot noise.
- **Power-play features:** add nothing beyond the rest. Absence features work only through TOI.
- **Injury news at the close:** already priced. Teammate newly out +2.0% ± 3.3, no new absence +10.9% ± 3.1.
- **Skipped:** moneyline, puck line, totals (sharpest markets); goals, points, assists (low counts). Kalshi has no SOG series.
