# CLAUDE.md: NHL player props (shots on goal, goalie saves)

Not a Roblox project: the Roblox rules in `../CLAUDE.md` do not apply. Started 2026-10-29.
Same discipline as `../NBAProps/nba-props/CLAUDE.md` rules 1-10 (frozen baseline, no lookahead,
price what you can get, judge by CLV, only info known at quote time).
🔴 **The user makes every git commit.** Never commit or push. Remote: github.com/ivanchows/NHL-Props.

## Why props, why SOG (decided 2026-10-29)
- Mainlines (ML, puck line, totals) are the sharpest NHL markets and goals are low-count noise: skip.
- Targets: **skater shots on goal** first, **goalie saves** second. Skip goals/points/assists props (low counts, heavy vig).
- Lesson from NBA and MLB: the raw projection lost to Kalshi's mid both times. Expect the edge, if any, in
  **news timing** (morning-skate line combos, PP units, confirmed goalies) and soft books, not in the baseline.

## Setup
- uv project `nhl-props`, Python 3.12 (same interpreter flag as nba-props / mlb-props):
  `uv run --python C:\Users\ivan\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe ...`
- `data/` and `logs/` are gitignored.

## Data
- `nhl_props.fetch 2021 2026` (season 2021 = 2021-22): regular season, game ids `<season>02NNNN`, 1..1312.
  Per game: `data/raw/<season>/<gid>_{box,pbp,shifts}.json.gz`. ~8 req/s, 6 workers. Resumes; reruns only
  retry missing shifts and unplayed games.
  - `api-web.nhle.com/v1/gamecenter/<gid>/boxscore`: per-player sog, toi, shifts; goalies saves/shotsAgainst/starter.
  - `.../play-by-play`: every shot (`shot-on-goal`, `missed-shot`, `blocked-shot`, `goal`) with shooter, goalie in net,
    x/y, `situationCode` (strength), period clock. Also `startTimeUTC`.
  - `api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<gid>`: every shift (who was on ice when).
- 🔴 Shift charts are MISSING for 2026020090-151 (Oct 13-20, 2026) and 2024021235-1291 (Apr 2025), though the
  games are final. Box TOI covers it.
- DONE 2026-10-29: every regular-season game 2021-22 -> 2025-26 (1,312 each) + 202 of 2026-27 so far; ~6,760 games.
- 🔴 **League shot volume falls every year**: team SOG/game 31.6 (2021) 31.2, 30.3, 28.3, 27.8, 27.3 (2026 so far).
  Same trap as MLB K%: a history-weighted rate will over-project. Expect a scale or a season trend term.
- ⚠️ 2026-27 game 2026020001 is dated 2026-09-29 and 2024020001 2024-10-04 (Prague): use `gameDate`, not id order.
  Preseason is gameType 1, so it never appears under `02` ids.
- ⚠️ The API throttles: ~35 min a season, not 8. Retries back off up to 60 s; a game that still fails logs
  `gave up:` and the next run retries it. Paused/killed runs lose nothing.
- `nhl_props.data`: box + pbp -> `data/skater_games.parquet` (toi in minutes, sog, goals, pbp shot attempts:
  `pbp_sog` incl. goals, `missed`, `blocked_att`) and `data/goalie_games.parquet` (starter, shots_against, saves).
  Rerun after every fetch. Shootout attempts are excluded.
  - Checked 2026-10-29: box sog = pbp sog in all but 55 / 132k rows (scorer disagreement; props settle on the box).
    Exactly 2 starting goalies every game. saves = SA - GA 99.8%.
  - Team sog > opposing goalies' shots_against by exactly the **empty-net goals**: a skater's EN goal is a SOG
    no goalie faced. Model saves from goalie shots_against, not from team SOG.
## Pipeline
`nhl_props.fetch` -> `nhl_props.data` -> `nhl_props.baseline` (tunes on 2022-24, 2021 = warm-up history; scores
2025 = 2025-26 and 2026 = 2026-27 so far; writes `data/baseline_params.json`, `data/baseline_pred.parquet`, ~4 min).

## SOG baseline (2026-10-29) = the frozen baseline to beat
mu = player ratio x league rate x projected TOI; SOG ~ NegBin(mu, r = 17.3).
- League rate: SOG/min by group (F/D) over the previous 20 game-days. This tracks the falling shot volume:
  the fitted overall scale came out 1.001, so no fudge factor was needed (unlike MLB's 0.975).
- Player ratio: decayed SOG / decayed expected SOG (half-life 240 d), shrunk to 1 with 10 expected shots.
- TOI: decayed mean (half-life 40 d), shrunk to the debut TOI with 0.5 games (edge of the grid).
- 🔴 The first date in the data has no league rate: it must be backfilled, or the NaN poisons every later
  decayed sum of the players in that game (cost 4,140 test rows before the fix).
- 2025-26 test: MAE 1.015 (season-to-date mean 1.045, constant 1.172; given the true TOI 0.997), bias +0.007,
  NB log-lik -1.536 (season-to-date -1.583). Over 0.5/1.5/2.5/3.5 calibrated within ~1-2 pts per bin.
  Slight under-projection of the top bins (P over 2.5 0.565 predicted vs 0.587 actual).
- 2026-27 so far (202 games): MAE 0.996, bias -0.033; TOI projected 0.47 min short early in the season.
- Even the TRUE TOI only improves MAE by 0.018: better TOI alone won't make an edge. Most error is shot noise.
- Next: goalie saves baseline; then market data (Kalshi NHL series if any, sportsbook SOG lines) to test vs price.

## Market data (checked 2026-10-29)
- **Kalshi: NO shots-on-goal series.** Has `KXNHLSAVE` (goalie saves; 211 events since 2026-09-29 = this season
  only, mostly ONE rung per goalie like MLB outs), `KXNHLGOAL`, `KXNHLPTS`, `KXNHLAST`, game/spread/total.
  (`KXNHLSAVES` exists but is empty.) Reuse `../MLBProps/src/mlb_props/kalshi.py` for candles.
- **Novig: SHOTS_ON_GOAL and SAVES** (also PLAYER_GOALS, POINTS, ASSISTS, POWER_PLAY_POINTS). Novig history starts
  2026-08-04, so NHL = this season only. `nhl_props.novig markets` (signed scan, each 5,000-market page is only ~6 h
  of all sports, ~25 s a page) then `nhl_props.novig trades` (free daily CSVs, ~160 MB a day, filtered to NHL).
  Prices are trades, not quotes (see MLB's novig_backtest for the method).
  - PULLED 2026-10-29 (through 2026-10-28): 213 events, 15,713 markets (14,796 SOG, 917 saves);
    154,922 taker trades, 100% matched to markets: SOG 140k trades / $5.7M on 4,993 markets, saves 15k / $0.74M on 589.
    The scan took ~48 min (111 pages); `trades` skips days already saved.

## Baseline vs Novig SOG (2026-10-29, `nhl_props.novig_backtest`, log `logs/novig_backtest.log`)
MLB method: price per side = last TAKER trade < 60 min old; open = first trade + 1 h, close = last before start; fee 0;
net CLV = close mid - cost. Names: Novig "First Last" vs pbp `rosterSpots` (fallback first initial + last, unique per date).
- 4,879 / 4,964 lines matched, 202 games. Lines mostly 1.5 / 2.5. 23 settlement mismatches (0.5%, all off by one SOG:
  Novig's source vs the box score): dropped.
- OPEN: log loss model 0.6813, market mid 0.6789, 50/50 blend **0.6766** (beats both). Best line per player-game,
  ROI / net CLV: edge >= 0% +2.0% / +0.39c ± 0.13; >= 4% +6.0% ± 3.4 / +0.84c ± 0.20; >= 10% +13.8% ± 6.6 / +1.68c ± 0.55.
  Bets are ~80% unders (model p_over 0.435 vs actual 0.462: early-season TOI under-projection).
- Control: every under at the open CLV -0.85c (ROI +0.5%), every over -1.32c (ROI -5.0%): the model's picks beat a
  blind under by ~1.2-2.5c. CLV only on lines two-sided at the close (~1/3).
- CLOSE: market clearly better (0.6797 vs model 0.6887), ROI -2% to +1%: no edge. Same pattern as MLB: the open is soft.
- One month, many cuts, raw frozen baseline: promising, not proven.

- **PTO (picktheodds.app)**: closing lines of every book for finished games via the page's own GraphQL
  `completedBetMarket` (method + scripts in `../NBAProps/CLAUDE.md`, "PTO CLOSING ODDS VIA THE PAGE'S OWN API").
  Runs in the user's Chrome.
- **PTO NHL pull (started 2026-10-29 17:06 ET)**: `scripts/pto_pull.js` (run in a PTO tab via the Chrome tools) +
  receiver `nhl_props.pto serve <blackouts>` (detached via Start-Process, log `logs/pto_receiver.log`) ->
  `data/pto/raw/` -> `nhl_props.pto build` -> `data/pto/SOG.parquet`, `SAVES.parquet` (every line, the user's 20 books).
  - Markets: `PLAYER_PROP_SHOTS_ON_GOAL`, `PLAYER_PROP_SHOTS_SAVED` (also _NO_OT, period variants, BLOCKED_SHOTS, GOALS_AGAINST).
    Game type `IceHockeyGameType`; game list = `games(league: NHL, request: {afterDateTime, take: 50, orderBy: ASC})`
    (`cursor` needs both cursor + cursorDateTime). Players: `players(ids, league: NHL)`.
  - Probe: ~19 skaters / 190 SOG markets per game, 117-153 site ids (most are DFS / sweepstakes apps; we keep 20).
    History reaches back to at least Oct 2025 (incl. preseason Sep 30). 1,660 finished games Oct 7 2025 -> now.
  - 🔴 Requiring every webpack module to find `gql` FROZE the tab (12k modules); require only modules whose source
    has `"Document"` and `fragment` (2 candidates). A frozen tab: close it, open a new one.
  - 🔴 **The PTO page reloads itself** (seen ~40 min in, 2026-10-29) and the reload kills the pull silently (`window.__pull`
    undefined). Resume: re-inject with `PTO_PULL.skip` = every `<market>__<gameId>` already in `raw/odds/`. Watch the
    file count.
  - 🔴 **Chrome throttles the tab once it loses focus**: ~2 game-markets/s while focused, ~4-20/min in the background
    (a 1.2 s `sleep` became 20-30 s).
  - 🔴 **Rate limit**: unpaced (~46/min) got HTTP 429 after ~1,000 requests. The page's link then refuses everything with
    "Network is still offline. Request blocked." (its health probe treats 429 as offline). Stop, wait, reload the tab,
    resume with `PTO_PULL.pace` (default 2.5 s). A silent AudioContext keep-alive does not work:
    it stays suspended without a click (and awaiting `resume()` hangs the eval).
  - 🔴 Port 8766 is the NBA snapshot's: both the page and the receiver pause in blackout windows around the NBA
    early / late / open snapshots (unix-second pairs passed to both). Messages carry `nhl: true`.
