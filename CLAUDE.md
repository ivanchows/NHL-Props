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
  Since 2026-10-30 `.venv` is on 3.12.15 and plain `uv run` works, scheduled tasks included. uv.lock pins the packages.
- `data/` and `logs/` are gitignored, so `data/v2_frozen.joblib` is not in git: rebuild it with `nhl_props.v2` (seeded, same model).

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

## Baseline vs PTO closes (2026-10-30, `nhl_props.pto_backtest`, log `logs/pto_backtest.log`)
Regular season 2025-26 + 2026-27 so far, frozen baseline. Main line per player-game (most two-sided books, 3+): 19,884 lines,
1,391 games, 9 books, hold 6.9%. Consensus = median no-vig. Bets at the best price, -250..+250, plus odds risk 1u, minus odds
risk to win 1u, ROI = net / risked, SE clustered by game. PTO `isWinner` is always null: settled on the box score.
- Log loss: model 0.6863, consensus 0.6828 (50/50 0.6829). The close wins at every line and in every month except Apr 2026
  and the Sept 2026 handful. Over rate 0.501, consensus 0.507, model 0.484 (model leans low on players with props).
- Model bets, edge 3-12% at the best price: 7,078, 3,609-3,469, +45.0u, **+0.5% ± 1.2** (2025-26 +0.3%, 2026-27 +1.3%;
  unders +1.3% ± 1.3, overs -2.1% ± 2.5). Edge does not scale: 3-6% +0.7%, 6-12% +0.3%, 12%+ +5.1% ± 4.4; 0-3% -2.6%.
- No-model controls at the best price: every under -2.3% ± 0.7, every over -3.6% ± 0.7, best price beats consensus -0.6%.
- One book only (edge 3-12%): all lose. theScore -0.3%, bet365 -1.2%, NoVig -1.5%, FD -3.3%, DK -3.6%, MGM -4.5%, Caesars -5.8%.
- Walk-forward blend (fit before Feb 2026, test after, 8,419 lines): logit weights model 0.17, consensus 0.91. Log loss
  0.6849 vs consensus-only control 0.6855: **-0.00063 ± 0.00023**, the model adds a little the close lacks, more than the NBA
  model did (-0.0003 ± 0.0003). Blend bets are few: 3-6% edge 218, +7.3% ± 6.8 (control 344, -6.1% ± 5.5).
- Verdict: break-even at the best price, a loss at any one book. Not bettable at the close as is. The small blend signal + the
  soft Novig open are the leads; the close itself is efficient.

## SOG v2 (2026-10-30, `nhl_props.v2`, writes `data/v2_pred.parquet`; rerun after every `nhl_props.data`)
Boosted Poisson correction on the frozen baseline (fit y / mu0 weighted by mu0 = offset log mu0). Settings picked on 2022-23 ->
2024, fit 2022-24, scored 2025 + 2026 with no choice made on them. NB r stays 17.3. `data.py` now adds `pp_att` and `pp_toi`
(pbp situationCode windows x shift charts; 1.8% of rows have no charts).
- Features: attempt rates (30/120/480 d), shot accuracy, short-window SOG rate, PP minutes + PP shot rate, short-window TOI,
  rest, opponent SOG allowed (by F/D and overall), team shot tendency, decayed goal difference, home, back-to-backs, and the
  LINEUP group (who dressed: team TOI sum, TOI share, PP share). `mu_nolineup` drops that group.
- 🔴 LINEUP is fair against a close, NOT an open: use `mu_nolineup` for anything priced before lineups are out.
- 2024 validation, ablation: opp and attempts and form matter most; PP group adds nothing beyond the rest (log-lik unchanged).
- Held out: 2025-26 log-lik -1.5360 -> -1.5308, MAE 1.015 -> 1.011; 2026-27 -1.5237 -> -1.5202, MAE 0.996 -> 0.997.
- PTO closes (`pto_backtest v2_pred`, same lines as the baseline test): log loss v2 0.6833, consensus 0.6828, **50/50 0.6820**
  (baseline was 0.6863, no gain from blending). Mean P(over) 0.500 vs actual 0.501: the low lean is gone.
  Best price: edge 3-12% 5,307 bets +1.1% ± 1.4; 3-6% -0.3% ± 1.6; 6-12% +4.1% ± 2.4; 12%+ +16.6% ± 6.5 (206).
  **Edge 6%+ (the MLB rule, set before this test): 1,838 bets, 970-868, +117.2u, +5.5% ± 2.3.**
- Checks (`scripts/v2_checks.py`, logs/v2_checks.log): no-lineup 6%+ +7.2% ± 2.2 (2,075), so no lineup leak. Mainstream 8 books only
  (DK FD MGM CZR b365 Fanatics theScore BetRivers) 6%+ +7.4% ± 2.5 (1,437). Placebo (shuffle model-vs-consensus gaps, 200 runs):
  mean -1.4%, max +3.1%, real +5.5%: 0/200 reach it. 2025-26 +3.7% ± 3.0, 2026-27 +8.1% ± 3.4; unders +6.0% ± 2.7, overs +4.4% ± 4.4;
  line 1.5 +8.1%, 2.5 +2.6%. ⚠️ Not stable by month: Dec -14.8%, Feb -20.1%, Mar +16.8%, Apr +17.7%, Oct 2026 +7.3%. ⚠️ Non-monotone in
  the raw model-consensus gap (6-8% -2.5%, 8%+ +5.3%).
- Walk-forward blend (fit before Feb 2026): weights model 0.47, consensus 0.62; log loss -0.00111 ± 0.00057 vs consensus-only control.
  Blend bets: 3-6% +7.9% ± 3.6 (643), 6%+ +11.4% ± 8.0 (137).
- Novig (`novig_backtest v2_pred mu_nolineup`, 2026-27 only): OPEN edge >= 6%, best line per player-game: 472 bets, +11.7% ± 4.4, net CLV
  +1.18c ± 0.28; >= 4% +10.0% ± 3.6, CLV +0.85c ± 0.21. Close 6%+ +6.6% ± 5.1. Open log loss 0.6798 vs market 0.6789, 50/50 0.6762.
- Part of 2026-27's edge is the books lagging falling shot volume (Oct 2026 over rate 0.471, consensus 0.489, v2 0.466). May fade.
- Verdict: first real signal, and it passes every check so far, but it is ~2.4 SE on history and the months swing. Paper-track before staking.
- Audit (2026-10-30, on the actual bets):
  - Clean: 1 duplicate (PTO lists Mike / Michael Benning twice, fixed in `main_lines`, now 1,837 bets, +116.2u, +5.5%). No wrong-game matches
    (PTO's LA NJ SJ TB UTAH = LAK NJD SJS TBL UTA). No leakage: scrambling every outcome from 2026-01-15 on moved no feature on or before that
    date (215,798 rows, 24 of 32 features moved after it). Baseline params tuned on 2022-24 only.
  - 🔴 The ROI is mostly line shopping: 6%+ is +5.5% at the best price, +3.0% at the 2nd best, 0.0% at the median book. The signal itself: bets won
    52.8% vs consensus 49.3% (+3.5 pts ± 1.1). No-lineup 54.0% vs 49.5%.
  - Weak by period: first half (to 2026-01-16) 6%+ +1.8% ± 4.0, second half +7.3% ± 2.8. Bootstrap 95% by date +0.8% to +10.0%.
  - One-way lean: 71% unders, 21% overs share of bets in the second half, so a market catch-up on shot volume hits most bets at once.
  - Novig at the paper-test rules (odds cap, one bet per player-game): 462 bets, +60.6u, +10.7% ± 4.5, 29 days. Prices are last trades, and 239 of the
    462 had only that side traded, so they may not have been available. Not a fair comparison with the live quotes.
- Not done: confirmed PP units as a model feature (DailyFaceoff has them live, no history).

## SOG v3 (2026-10-30, `nhl_props.v3`, writes `data/v3_pred.parquet`, `data/v3_frozen.joblib`; checks `scripts/v3_checks.py`, logs/v3_checks.log)
v2 + TOI model (`toi_m`, gradient boosting on toi_hat residual, sees absences). No NHL injury history exists: absences = who dressed, so fair at a CLOSE only.
- TOI MAE 1.927 -> 1.814 (2025), 1.976 -> 1.870 (2026). With a same-position teammate newly out, bias +0.26 -> -0.01 (2025).
- Chosen on 2024: `+toi model`. The raw absence features added nothing to SOG once tested alone (ll -1.5408 vs v2 -1.5408); they work through TOI.
  SOG: 2025 ll -1.5308 -> -1.5306, 2026 -1.5191 -> -1.5184.
- PTO closes, same lines (data through 2026-10-29): 6%+ v3 1,820 bets +141.8u +6.8% ± 2.2, 2nd best +4.2%, median book +1.3%; v2 1,849 +123.8u +5.8%, +3.2%, +0.2%.
  Log loss v3 .6831, v2 .6833, consensus .6828, 50/50 .6819. First half +3.8% ± 3.3, second +9.6% ± 3.1.
- 🔴 Injuries are not where the money is at the close: teammate newly out +2.0% ± 3.3, no new absence +10.9% ± 3.1 (same as NFL inactives: priced by then).
  Before the market moves is untested and can't be backtested.
- Leak check: scrambling outcomes from 2026-01-15 moved nothing on or before it (215,798 rows, every TOI model input).

## Paper test of v3 (`nhl_props.live_v3`, started 2026-10-30 6:19 PM, no real bets)
🔴 Same rule as v2's test, frozen `data/v3_frozen.joblib`. Do NOT rerun `nhl_props.v3`. Never pool with v2. `\NHL Props\decide3` every 30 min; `nightly` settles it too.
- Lineup = DailyFaceoff `ev` f1-4 + d1-3, minus ESPN Out / IR / suspended. Under 16 matched: last game's lineup minus that list. Projections rebuild only when a lineup changes.
- `data/live_v3/news/{espn,dfo}_<date>.parquet` keeps every run's raw rows: the only NHL injury history there will be. Includes PP units, game-time decisions.
- Owns #shots-on-goal. Oct 30 is partial: game 2026020214 had started, 2026020215 got early and late in one run.

## Paper test of v3 at PTO's best price (`nhl_props.pto_live`, built 2026-10-30, no real bets)
🔴 Same frozen v3, separate test: never pool with Novig. `\NHL Props\pto` at :05 and :35; `nightly` settles it. Posts picks with the book named.
- Reads PTO with the NBA extension (v0.3.1 adds the NHL shots screen; needs Reload on chrome://extensions after any change). 🔴 PTO strips `#ptoscan` before document_idle on the NHL screen: `start.js` reads it at document_start. Chrome opens only when a stage is due; the extension closes its tab.
- 🔴 Port 8766 is shared with NBAProps-PTO (10-minute slots), so this runs on the 5s with a 3.5 min limit and skips if busy. Needs Chrome open and logged in to PTO.
- Started games have no "AAA @ BBB · Shots on Goal" label on the screen, so an NHL scan scrolls past them until the first labelled row.
- Sportsbooks show no size, so the $25 floor only applies to exchanges. Saved: `data/live_pto/raw/`, `lines/` (every book's price, every scan).
- Task on since 2026-10-30. A real scan ran and posted back (42 s, 20 books) but with 0 rows, every game had started: the first real prices are the next slate.

## Paper test of v2 (`nhl_props.live`, started 2026-10-30, no real bets)
🔴 Rule fixed before the first decision. Do NOT change it or refit mid-test; a new model = a new test.
- Discord: #shots-on-goal (1565469010248867971) through the EV bot's token (`../NBAProps/ev-bot/config.toml`, never print it), no webhook.
  Posting belongs to v3 now (`live.POST` off): v2 logs and grades without posting.
- Model: `data/v2_frozen.joblib` (no-lineup v2, r 17.3). `live check` proves it reproduces `v2_pred.parquet` (max diff 0.000000).
  Rerunning `nhl_props.v2` refits to the same model (seeded) but would overwrite the file: don't.
- Price: Novig public book, no key, no Chrome (`/v3/public/catalog/{events,markets}`). A book side's bid = the OTHER side's offer:
  buy over = 1 - best under bid; depth $ = that bid's qty / 100. Pre-game takers pay no fee.
- One side per player-game (largest edge on any line), odds -250..+250, depth >= $25. **tier bet = edge >= 6%** (the test);
  tier watch = 3-6% (logged only). Plus odds risk 1u, minus odds risk to win 1u, ROI = net / risked.
- Two stages, judged apart: early = first run with 10+ lines on game day from 09:00 ET; late = under 50 min to puck drop.
- Scratch = void. Settles on the box score. Close = last logged quote before the start (~0-30 min out).
- Files: `data/live/bets.csv`, `decided.csv` (one row per game x stage, so no double logging), `quotes/<date>.parquet` (every line,
  every run), `proj/<date>.parquet` (first run of the day builds it, ~1 min), `results.csv`, `report.md`, `logs/`.
- Task Scheduler `\NHL Props\` (user OK 2026-10-30, ends 2027-04-20): `decide` every 30 min 09:00-23:30, `nightly` 05:00 =
  `fetch recent` + `data` + `live settle`. Only while the user is logged on and the PC is awake; StartWhenAvailable catches up.
- 🔴 `fetch 2026` probes all 1,312 ids, ~14 min, mostly throttled "unplayed": nightly uses `fetch recent` (schedule, last 4 days).
- 🔴 Rosters are the current roster, so a player traded today projects with his new team and the same history.
- Not tracked live: best price across books (PTO needs Chrome). Compare later with a PTO close pull; the backtest's +5.5% is best-price.
- Judge at ~500 bets per stage: net CLV first, ROI second. Regular season only.

- **PTO (picktheodds.app)**: closing lines of every book for finished games via the page's own GraphQL
  `completedBetMarket` (method + scripts in `../NBAProps/CLAUDE.md`, "PTO CLOSING ODDS VIA THE PAGE'S OWN API").
  Runs in the user's Chrome.
- **PTO NHL pull (started 2026-10-29 17:06 ET)**: `scripts/pto_pull.js` (run in a PTO tab via the Chrome tools) +
  receiver `nhl_props.pto serve <blackouts>` (detached via Start-Process, log `logs/pto_receiver.log`) ->
  `data/pto/raw/` -> `nhl_props.pto build` -> `data/pto/SOG.parquet`, `SAVES.parquet` (every line, the user's 20 books).
  - **DONE 2026-10-30 00:46** (1,668 finished games Oct 8 2025 -> Oct 29 2026, incl. playoffs + preseason; 0 missing):
    SOG 1,509 games / 26,270 player-games / 1.19M prices (0.9% unnamed); SAVES 1,369 games / 2,817 goalie-games / 102k.
    Gaps are PTO's, not the pull's: Mar 2026 only 181/247 games with SOG (108 saves), Sep 2026 preseason 7/72.
    SOG books: BET_365, DK, PROP_BUILDER, HARD_ROCK_IN, ONYX, FANATICS, FD, FLIFF, NO_VIG, CAESARS, MGM, ... **no Pinnacle,
    no BetOnline**; Kalshi only on saves (370 prices). SOG lines 0.5-5.5+ (alt ladders); saves mostly 21.5-26.5.
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
