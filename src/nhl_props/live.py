"""Forward paper test of SOG v2 (no-lineup model) at Novig's live prices. No real bets.
v3 (nhl_props.live_v3) owns #shots-on-goal: with POST on, tier bet picks and nightly results go there through the EV bot
(token in ../NBAProps/ev-bot/config.toml). v2 keeps logging without posting.

    uv run python -m nhl_props.live decide [--dry-run]   # every 30 min: log quotes, decide each game at two stages
    uv run python -m nhl_props.live settle               # after nhl_props.fetch + data: grade bets, write the report
    uv run python -m nhl_props.live check                # frozen model reproduces v2_pred.parquet

Rule, fixed before the first decision (2026-10-30): data/v2_frozen.joblib is never refit. One side per player-game: the
largest edge on any line. Edge = model P - price to buy at Novig's public book. Odds -250..+250, depth >= $25.
tier bet = edge >= 6%; tier watch = 3-6% (logged, not the test). Stakes: plus odds risk 1u, minus odds risk to win 1u.
Stages: early = first run with lines on game day (from 09:00 ET); late = under 50 min to puck drop. Judge each stage alone.
A player who does not dress = void. Settles on the box score. Closing price = last logged quote before the start.
"""

import json
import re
import sys
import tomllib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom

from nhl_props.baseline import DATA, load
from nhl_props.novig_backtest import norm
from nhl_props.pto_backtest import american, payoff
from nhl_props.v2 import features

LIVE = DATA / "live"
POST, TITLE = False, "v2"  # live_v3 flips these
SCALE = ("For scale: Novig open at these rules +10.7% ± 4.5 (462 bets, one month, last-trade prices), PTO close 6%+ +5.5% best price, "
         "+3.0% 2nd best, 0% median book.")
NOVIG = "https://api.novig.com/v3/public/catalog"
SCHEDULE = "https://api-web.nhle.com/v1/schedule/{date}"
ROSTER = "https://api-web.nhle.com/v1/roster/{team}/current"
ET = "America/New_York"
BOT_CONFIG = Path(r"C:\Users\ivan\NBAProps\ev-bot\config.toml")
CHANNEL = "1565469010248867971"  # #shots-on-goal
MIN_EDGE, WATCH_EDGE, MIN_DEPTH, MIN_ODDS, MAX_ODDS, LATE_MIN = 0.06, 0.03, 25, -250, 250, 50
BET_COLS = ["ts", "date", "game_id", "start_utc", "stage", "tier", "player_id", "name", "team", "line", "side", "price", "odds",
            "p_model", "mu", "edge", "depth", "market_id"]


def _json(url, **params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _paged(path, **params):
    out, after = [], None
    while True:
        j = _json(NOVIG + path, limit=500, **params, **({"after": after} if after else {}))
        out += j.get("items", [])
        after = j.get("next")
        if not after:
            return out


def bot_token() -> str | None:
    """The EV bot's token, read from disk and never printed."""
    try:
        return tomllib.loads(BOT_CONFIG.read_text(encoding="utf-8"))["discord"]["bot_token"] or None
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def post(lines: list[str]) -> bool:
    """Send lines to #shots-on-goal, 30 to a message (2,000 character limit). Log only the error's type."""
    token = bot_token()
    if not token:
        return False
    try:
        for i in range(0, len(lines), 30):
            r = requests.post(f"https://discord.com/api/v10/channels/{CHANNEL}/messages", headers={"Authorization": f"Bot {token}"},
                              json={"content": "\n".join(lines[i:i + 30]), "allowed_mentions": {"parse": []}}, timeout=20)
            r.raise_for_status()
    except requests.RequestException as e:
        print(f"discord post failed: {type(e).__name__} {getattr(e.response, 'status_code', '')}", flush=True)
        return False
    return True


def post_bets(b: pd.DataFrame, old: pd.DataFrame | None) -> None:
    """Tier bet picks, each once: a late pick the early run already posted stays quiet."""
    keys = ["game_id", "player_id", "side", "line"]
    b = b[b.tier == "bet"].drop_duplicates(keys)
    if old is not None:
        b = b[~b.set_index(keys).index.isin(old[old.tier == "bet"].set_index(keys).index)]
    if b.empty:
        return
    post(["**NHL shots on goal** (paper, Novig)"] + [f"{x.name} ({x.team}) {x.side} {x.line:g} {x.odds:+d}, edge {x.edge:+.1%}" for x in b.itertuples()])


def post_results(b: pd.DataFrame) -> None:
    """One post per finished date, once. Totals cover every graded tier bet."""
    f = LIVE / "posted_results.txt"
    posted = set(f.read_text(encoding="utf-8").split()) if f.exists() else set()
    x = b[b.tier == "bet"]
    d = x[x.status == "done"]
    open_dates = set(x[x.status == "pending"].date)
    line = lambda y, lab: f"{lab}: {y.won.sum()}-{(~y.won).sum()}, {y.net.sum():+.1f}u, ROI {y.net.sum() / y.risk.sum():+.1%}"
    for date, y in d.groupby("date"):
        if date in posted or date in open_dates:
            continue
        L = [f"**NHL shots on goal results, {date}** (paper, {TITLE})"] + [line(z, stage.capitalize()) for stage, z in y.groupby("stage")]
        L += [line(z, f"Total {stage}") for stage, z in d.groupby("stage")]
        if post(L):
            with open(f, "a", encoding="utf-8") as h:
                h.write(date + "\n")


def slate(date: str) -> pd.DataFrame:
    """Regular-season games on this ET date."""
    rows = []
    for day in _json(SCHEDULE.format(date=date)).get("gameWeek", []):
        if day["date"] != date:
            continue
        for g in day["games"]:
            if g.get("gameType") == 2:
                rows.append({"game_id": g["id"], "start_utc": g["startTimeUTC"], "state": g.get("gameState"),
                             "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
                             "away_name": g["awayTeam"]["commonName"]["default"], "home_name": g["homeTeam"]["commonName"]["default"]})
    d = pd.DataFrame(rows)
    if len(d):
        d["start"] = pd.to_datetime(d.start_utc, utc=True)
    return d


def roster_rows(games: pd.DataFrame, date: str) -> pd.DataFrame:
    """One unplayed row per rostered skater per game. toi/sog are 0: every feature reads earlier dates only."""
    rows = []
    for g in games.itertuples():
        for team, opp, home in ((g.away, g.home, False), (g.home, g.away, True)):
            r = _json(ROSTER.format(team=team))
            for p in r.get("forwards", []) + r.get("defensemen", []):
                first, last = p["firstName"]["default"], p["lastName"]["default"]
                rows.append({"game_id": g.game_id, "season": g.game_id // 1000000, "game_date": date, "start_utc": g.start_utc,
                             "team": team, "opp": opp, "home": home, "player_id": p["id"], "name": f"{first} {last}",
                             "pos": p["positionCode"], "toi": 0.0, "shifts": 0, "sog": 0, "goals": 0, "assists": 0, "pp_goals": 0,
                             "hits": 0, "blocks": 0, "pbp_sog": 0, "missed": 0, "blocked_att": 0, "pp_att": 0, "pp_toi": np.nan})
    return pd.DataFrame(rows)


def score_rows(s: pd.DataFrame, F: dict) -> pd.DataFrame:
    return s.assign(mu=s.mu0 * F["model"].predict(s[F["cols"]]), r=F["r"])


def project(date: str, games: pd.DataFrame) -> pd.DataFrame:
    """Frozen projections, cached per date: the features take about a minute, so only the first run of a day builds them."""
    f = LIVE / "proj" / f"{date}.parquet"
    if f.exists():
        p = pd.read_parquet(f)
        if set(games.game_id) <= set(p.game_id):
            return p
    f.parent.mkdir(parents=True, exist_ok=True)
    P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
    s = features(load(roster_rows(games, date)), P)
    p = score_rows(s[s.game_id.isin(games.game_id)], joblib.load(DATA / "v2_frozen.joblib"))
    p = p[["game_id", "player_id", "name", "team", "mu", "r"]].copy()
    p = p[np.isfinite(p.mu)]
    p["key"] = p.name.map(norm)
    p["key2"] = p.key.str[0] + " " + p.key.str.split(n=1).str[1]
    p.to_parquet(f)
    return p


def quotes(games: pd.DataFrame) -> pd.DataFrame:
    """Every open SOG line on this slate: the price to BUY each side. A book side's bid is the other side's offer."""
    ev = pd.DataFrame(_paged("/events", league="NHL"))
    ids = {}
    for g in games.itertuples():
        for e in ev.itertuples():
            a, _, h = str(e.description).partition("@")
            near = abs(pd.Timestamp(e.startsTs, unit="ms", tz="UTC") - g.start) < pd.Timedelta(hours=3)
            if near and a.strip().endswith(g.away_name) and h.strip().endswith(g.home_name):
                ids[e.eventId] = g.game_id
    rows = []
    for m in _paged("/markets", league="NHL"):
        if m.get("marketType") != "SHOTS_ON_GOAL" or m.get("status") != "OPEN" or m["eventId"] not in ids or m.get("strike") is None:
            continue
        outs = {o["name"].split()[0].lower(): o for o in m["outcomes"]}
        if "over" not in outs or "under" not in outs:
            continue
        bid = lambda o: (float(o["bid"]["price"]), float(o["bid"]["qty"]) / 100) if o.get("bid") else (np.nan, 0.0)
        bo, qo = bid(outs["over"])
        bu, qu = bid(outs["under"])
        rows.append({"game_id": ids[m["eventId"]], "market_id": m["marketId"], "player": re.sub(r"\s+[\d.]+\s+\w+$", "", m["description"]).strip(),
                     "line": float(m["strike"]), "price_over": 1 - bu, "price_under": 1 - bo, "depth_over": qu, "depth_under": qo})
    return pd.DataFrame(rows)


def decide_game(q: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game with an edge of at least WATCH_EDGE: the best side and line."""
    q = q.assign(key=q.player.map(norm))
    q["key2"] = q.key.str[0] + " " + q.key.str.split(n=1).str[1]
    m1 = q.merge(p.drop(columns=["key2", "game_id"]), on="key")
    rest = q[~q.market_id.isin(m1.market_id)]
    m2 = rest.drop(columns="key").merge(p.drop_duplicates("key2", keep=False).drop(columns=["key", "game_id"]), on="key2")
    m = pd.concat([m1, m2], ignore_index=True)
    if m.empty:
        return m
    m["p_over"] = nbinom.sf(np.floor(m.line).astype(int), m.r, m.r / (m.r + m.mu)).clip(0.01, 0.99)
    rows = []
    for side, pm, price, depth in (("over", m.p_over, m.price_over, m.depth_over), ("under", 1 - m.p_over, m.price_under, m.depth_under)):
        rows.append(m.assign(side=side, p_model=pm, price=price, depth=depth, edge=pm - price))
    b = pd.concat(rows, ignore_index=True).dropna(subset=["price"])
    b = b[(b.price > 0.01) & (b.price < 0.99) & (b.depth >= MIN_DEPTH) & (b.edge >= WATCH_EDGE)]
    if b.empty:
        return b
    b["odds"] = b.price.map(lambda x: int(float(american(x))))
    b = b[b.odds.between(MIN_ODDS, MAX_ODDS)]
    return b.sort_values("edge").groupby("player_id").tail(1)


def decide(dry: bool = False) -> None:
    base = (DATA / "live_dryrun") if dry else LIVE
    base.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=ET)
    date = now.strftime("%Y-%m-%d")
    if now.hour < 9 and not dry:
        return
    games = slate(date)
    if games.empty:
        print(f"{now:%H:%M} no games")
        return
    q = quotes(games)
    if q.empty:
        print(f"{now:%H:%M} no Novig lines yet")
        return
    (base / "quotes").mkdir(exist_ok=True)
    qf = base / "quotes" / f"{date}.parquet"
    snap = q.assign(ts=now.tz_convert("UTC"))
    (pd.concat([pd.read_parquet(qf), snap]) if qf.exists() else snap).to_parquet(qf)
    p = project(date, games)
    df = base / "decided.csv"
    done = set(map(tuple, pd.read_csv(df)[["game_id", "stage"]].to_numpy())) if df.exists() else set()
    new, marks = [], []
    for g in games.itertuples():
        mins = (g.start - now).total_seconds() / 60
        x = q[q.game_id == g.game_id]
        if mins <= 0 or g.state not in ("FUT", "PRE") or len(x) < 10:
            continue
        for stage in ("early", "late"):
            if (g.game_id, stage) in done or (stage == "late" and mins > LATE_MIN):
                continue
            b = decide_game(x, p[p.game_id == g.game_id])
            marks.append({"game_id": g.game_id, "stage": stage, "ts": now.isoformat(), "lines": len(x), "picks": len(b)})
            if len(b):
                new.append(b.assign(ts=now.isoformat(), date=date, game_id=g.game_id, start_utc=g.start_utc, stage=stage,
                                    tier=np.where(b.edge >= MIN_EDGE, "bet", "watch")))
    if marks:
        pd.DataFrame(marks).to_csv(df, mode="a", header=not df.exists(), index=False)
    if new:
        f = base / "bets.csv"
        old = pd.read_csv(f) if f.exists() else None
        out = pd.concat(new, ignore_index=True)[BET_COLS]
        out.to_csv(f, mode="a", header=not f.exists(), index=False)
        if POST and not dry:
            post_bets(out, old)
    print(f"{now:%H:%M} {len(games)} games, {len(q)} lines, stages {[(m['game_id'], m['stage'], m['picks']) for m in marks]}", flush=True)


def settle() -> None:
    f = LIVE / "bets.csv"
    if not f.exists():
        print("no bets yet")
        return
    b = pd.read_csv(f)
    sk = pd.read_parquet(DATA / "skater_games.parquet", columns=["game_id", "player_id", "sog"])
    b = b.merge(sk, on=["game_id", "player_id"], how="left")
    played = set(sk.game_id)
    b["status"] = np.where(b.sog.notna(), "done", np.where(b.game_id.isin(played), "void", "pending"))
    b["won"] = np.where(b.side == "over", b.sog > b.line, b.sog < b.line)
    b["net"], b["risk"] = payoff(b.odds, b.won)
    b.loc[b.status != "done", ["net", "risk"]] = 0.0

    # Closing price = the last logged quote of that line before the start.
    q = pd.concat([pd.read_parquet(x) for x in sorted((LIVE / "quotes").glob("*.parquet"))], ignore_index=True)
    q = q.dropna(subset=["price_over", "price_under"]).sort_values("ts").groupby("market_id").tail(1)
    q["close_mid"] = (q.price_over + 1 - q.price_under) / 2
    b = b.merge(q[["market_id", "close_mid"]], on="market_id", how="left")
    b["close_side"] = np.where(b.side == "over", b.close_mid, 1 - b.close_mid)
    b.to_csv(LIVE / "results.csv", index=False)

    d = b[b.status == "done"]
    L = [f"# NHL SOG {TITLE} paper test, through {d.date.max() if len(d) else '-'}", "",
         f"Paper only, Novig prices. tier bet = edge 6%+, watch = 3-6%. Stages are separate tests. {SCALE}", "",
         "| Tier | Stage | Bets | W-L | Units | ROI | ± | Avg got | Avg close |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (tier, stage), x in d.groupby(["tier", "stage"]):
        g = x.groupby("game_id").agg(net=("net", "sum"), risk=("risk", "sum"))
        se = g.net.std() * np.sqrt(len(g)) / g.risk.sum() if len(g) > 1 else np.nan
        c = x.dropna(subset=["close_side"])
        got, cl = (american(float(c.price.mean())), american(float(c.close_side.mean()))) if len(c) else ("-", "-")
        L.append(f"| {tier} | {stage} | {len(x)} | {x.won.sum()}-{(~x.won).sum()} | {x.net.sum():+.1f}u | {x.net.sum() / x.risk.sum():+.1%} | {se:.1%} | {got} | {cl} |")
    L += ["", f"Pending {(b.status == 'pending').sum()}, void {(b.status == 'void').sum()}."]
    (LIVE / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if POST:
        post_results(b)


def check() -> None:
    """The saved model, on the same features as the backtest, must reproduce v2_pred.parquet."""
    v = pd.read_parquet(DATA / "v2_pred.parquet")
    P = json.loads((DATA / "baseline_params.json").read_text(encoding="utf-8"))
    F = joblib.load(DATA / "v2_frozen.joblib")
    s = score_rows(features(load(), P).query("season == 2026"), F)
    j = v.merge(s[["game_id", "player_id", "mu"]], on=["game_id", "player_id"], suffixes=("", "_live"))
    print(f"{len(j):,} rows, max |mu_nolineup - frozen| {(j.mu_nolineup - j.mu_live).abs().max():.6f}, frozen r {F['r']:.1f}")


if __name__ == "__main__":
    {"decide": lambda: decide("--dry-run" in sys.argv), "settle": settle, "check": check}[sys.argv[1]]()
