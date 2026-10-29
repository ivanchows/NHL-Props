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
- 🔴 Shift charts are MISSING for 2026020090-151 (Oct 13-20, 2026) though the games are final. Box TOI covers it.
- ⚠️ 2026-27 game 2026020001 is dated 2026-09-29 and 2024020001 2024-10-04 (Prague): use `gameDate`, not id order.
  Preseason is gameType 1, so it never appears under `02` ids.
- Market data: not fetched yet. Next: Kalshi NHL series (if any) and sportsbook SOG/saves lines.
