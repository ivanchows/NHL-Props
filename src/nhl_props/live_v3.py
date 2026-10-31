"""Forward paper test of SOG v3 at Novig's live prices: tonight's projected lineups + the TOI model. No real bets.

    uv run python -m nhl_props.live_v3 decide [--dry-run]   # every 30 min
    uv run python -m nhl_props.live_v3 settle               # nightly, after fetch + data

Rule, fixed before the first decision (2026-10-30): same as nhl_props.live, with data/v3_frozen.joblib (never refit). Lineup = DailyFaceoff's
lines minus ESPN's Out / IR list; a team with no usable DailyFaceoff page uses its last game's lineup minus that list. Every run keeps the
raw DailyFaceoff and ESPN rows in data/live_v3/news/, which is the only way to get an NHL injury history. Separate test from v2: never pool them.
"""

import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
import requests

from nhl_props import live
from nhl_props.baseline import DATA, load
from nhl_props.novig_backtest import norm
from nhl_props.v2 import features
from nhl_props.v3 import TOI_COLS, absences

LIVE = DATA / "live_v3"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
DFO = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
OUT = {"Out", "Injured Reserve", "Long Term Injured Reserve", "Suspension", "Suspended"}
SCALE = "For scale: PTO close 6%+ +6.8% best price, +4.2% 2nd best, +1.3% median book (who dressed known, 1,820 bets)."


def teams(date: str) -> dict:
    """Abbreviation -> DailyFaceoff slug for tonight's teams."""
    out = {}
    for day in live._json(live.SCHEDULE.format(date=date)).get("gameWeek", []):
        if day["date"] == date:
            for g in day["games"]:
                for t in (g["awayTeam"], g["homeTeam"]):
                    out[t["abbrev"]] = norm(f"{t['placeName']['default']} {t['commonName']['default']}").replace(" ", "-")
    return out


def espn_rows() -> pd.DataFrame:
    rows = []
    for t in live._json(ESPN).get("injuries", []):
        for i in t.get("injuries", []):
            rows.append({"team_name": t.get("displayName"), "name": i.get("athlete", {}).get("displayName"), "status": i.get("status"),
                         "updated": i.get("date"), "detail": i.get("shortComment")})
    return pd.DataFrame(rows)


def dfo_rows(team: str, slug: str) -> pd.DataFrame:
    r = requests.get(DFO.format(slug=slug), headers=UA, timeout=25)
    r.raise_for_status()
    c = json.loads(re.search(r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S).group(1))["props"]["pageProps"]["combinations"]
    return pd.DataFrame([{"team": team, "name": p["name"], "cat": p.get("categoryIdentifier"), "group": p.get("groupIdentifier"), "injury": p.get("injuryStatus"),
                          "gtd": p.get("gameTimeDecision"), "updated": c.get("updatedAt"), "source": c.get("sourceName")} for p in c["players"]])


def lineups(rows: pd.DataFrame, date: str, base=LIVE) -> tuple[pd.DataFrame, dict]:
    """rows = everyone rostered tonight. Keeps the projected dressed players; logs the raw news."""
    now = pd.Timestamp.now(tz=live.ET)
    e = espn_rows()
    frames, note = [], {}
    for team, slug in teams(date).items():
        try:
            frames.append(dfo_rows(team, slug))
        except (requests.RequestException, AttributeError, KeyError, ValueError) as x:
            print(f"dailyfaceoff {team}: {type(x).__name__}", flush=True)
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["team", "name", "cat", "group", "injury", "gtd", "updated", "source"])
    news = base / "news"
    news.mkdir(parents=True, exist_ok=True)
    for name, x in (("espn", e), ("dfo", d)):
        f = news / f"{name}_{date}.parquet"
        x = x.assign(ts=now.isoformat())
        (pd.concat([pd.read_parquet(f), x], ignore_index=True) if f.exists() else x).to_parquet(f)
    out_names = {norm(n) for n in e[e.status.isin(OUT)].name.dropna()}
    last = pd.read_parquet(DATA / "skater_games.parquet", columns=["game_date", "game_id", "team", "player_id"])
    keep = []
    for team, x in rows.groupby("team"):
        dressed = d[(d.team == team) & (d.cat == "ev") & d.group.astype(str).str.match(r"^(f|d)\d")]
        ids = x[x.name.map(norm).isin(set(dressed.name.map(norm)))]
        src = "dfo"
        if len(ids) < 16:
            lg = last[(last.team == team) & (last.game_date < date)]
            ids = x[x.player_id.isin(lg[lg.game_id == lg.sort_values("game_date").game_id.iloc[-1]].player_id)] if len(lg) else x
            src = "last game"
        y = ids[~ids.name.map(norm).isin(out_names)]
        note[team] = f"{src} {len(y)}"
        keep.append(y)
    return pd.concat(keep, ignore_index=True), note


def project(date: str, games: pd.DataFrame, base=LIVE) -> tuple[pd.DataFrame, dict]:
    """Rebuilt only when a projected lineup changes: features take a couple of minutes."""
    rows, note = lineups(live.roster_rows(games, date), date, base)
    sig = str(sorted(map(int, rows.player_id)))
    f, fs = base / "proj" / f"{date}.parquet", base / "proj" / f"{date}.sig"
    if f.exists() and fs.exists() and fs.read_text(encoding="utf-8") == sig:
        return pd.read_parquet(f), note
    f.parent.mkdir(parents=True, exist_ok=True)
    P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
    F = joblib.load(DATA / "v3_frozen.joblib")
    s = absences(features(load(rows), P))
    s = s[s.game_id.isin(games.game_id)].copy()
    s["toi_m"] = s.toi_hat + F["toi"].predict(s[TOI_COLS])
    s["toi_ratio"] = s.toi_m / s.toi_hat
    s["mu"], s["r"] = s.mu0 * F["model"].predict(s[F["cols"]]), F["r"]
    p = s[["game_id", "player_id", "name", "team", "mu", "r", "toi_hat", "toi_m", "fresh_same"]].copy()
    p = p[np.isfinite(p.mu)]
    p["key"] = p.name.map(norm)
    p["key2"] = p.key.str[0] + " " + p.key.str.split(n=1).str[1]
    p.to_parquet(f)
    fs.write_text(sig, encoding="utf-8")
    return p, note


def decide(dry: bool = False) -> None:
    base = (DATA / "live_v3_dryrun") if dry else LIVE
    base.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=live.ET)
    date = now.strftime("%Y-%m-%d")
    if now.hour < 9 and not dry:
        return
    games = live.slate(date)
    if games.empty:
        print(f"{now:%H:%M} no games")
        return
    df = base / "decided.csv"
    done = set(map(tuple, pd.read_csv(df)[["game_id", "stage"]].to_numpy())) if df.exists() else set()
    due = []
    for g in games.itertuples():
        mins = (g.start - now).total_seconds() / 60
        if mins <= 0 or g.state not in ("FUT", "PRE"):
            continue
        due += [(g, s) for s in ("early", "late") if (g.game_id, s) not in done and (s == "early" or mins <= live.LATE_MIN)]
    if not due:
        print(f"{now:%H:%M} nothing due", flush=True)
        return
    q = live.quotes(games)
    if q.empty:
        print(f"{now:%H:%M} no Novig lines yet")
        return
    p, note = project(date, games, base)
    new, marks = [], []
    for g, stage in due:
        x = q[q.game_id == g.game_id]
        if len(x) < 10:
            continue
        b = live.decide_game(x, p[p.game_id == g.game_id])
        marks.append({"game_id": g.game_id, "stage": stage, "ts": now.isoformat(), "lines": len(x), "picks": len(b)})
        if len(b):
            new.append(b.assign(ts=now.isoformat(), date=date, game_id=g.game_id, start_utc=g.start_utc, stage=stage,
                                tier=np.where(b.edge >= live.MIN_EDGE, "bet", "watch")))
    if marks:
        pd.DataFrame(marks).to_csv(df, mode="a", header=not df.exists(), index=False)
    if new:
        f = base / "bets.csv"
        old = pd.read_csv(f) if f.exists() else None
        out = pd.concat(new, ignore_index=True)[live.BET_COLS]
        out.to_csv(f, mode="a", header=not f.exists(), index=False)
        if not dry:
            live.post_bets(out, old)
    print(f"{now:%H:%M} {len(q)} lines, lineups {note}, stages {[(m['game_id'], m['stage'], m['picks']) for m in marks]}", flush=True)


def settle() -> None:
    """Quotes for the closing price come from v2's log: both tests see the same Novig lines."""
    live.LIVE, live.POST, live.TITLE, live.SCALE = LIVE, True, "v3 Novig", SCALE
    q = LIVE / "quotes"
    q.mkdir(parents=True, exist_ok=True)
    for f in (DATA / "live" / "quotes").glob("*.parquet"):
        (q / f.name).write_bytes(f.read_bytes())
    live.settle()


if __name__ == "__main__":
    {"decide": lambda: decide("--dry-run" in sys.argv), "settle": settle}[sys.argv[1]]()
