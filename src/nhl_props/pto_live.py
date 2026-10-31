"""Forward paper test of SOG v3 at the BEST price across PTO's books. No real bets. Separate test from Novig: never pool them.

    uv run python -m nhl_props.pto_live decide [--dry-run] [--file scan.json]   # task \\NHL Props\\pto, :05 and :35
    uv run python -m nhl_props.pto_live settle                                  # nightly

Reads PTO with the NBA scanner extension (NBAProps/pto-scanner/extension, v0.3.1+): opens the NHL shots screen with #ptoscan in the user's
Chrome, the extension reads it, posts to 127.0.0.1:8766 and closes its own tab. Chrome only opens when a game has a stage due.
🔴 Port 8766 is shared with NBAProps-PTO (:00, :10, ...): this runs at :05 and :35 with a 3.5 min limit so it ends first, and skips if the port is busy.

Rule, fixed before the first decision (2026-10-30): data/v3_frozen.joblib, lineups as in live_v3. One side per player-game at the screen's main
line. Edge = model P - implied chance of the best price with odds -250..+250 (exchanges that show a size need $25+; books show none, so depth is
unknown there). tier bet = 6%+, watch = 3-6%. Stages: early = first scan with lines from 10:05 ET, late = under 50 min to puck drop.
Close = median no-vig across two-sided books in the last logged scan.
"""

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from nhl_props import live, live_v3
from nhl_props.baseline import DATA
from nhl_props.novig_backtest import norm

LIVE = DATA / "live_pto"
PORT = 8766
START = "https://picktheodds.app/en/odds-screen/NHL?group=PLAYER_PROP_SHOTS_ON_GOAL&time=PLAYER_PROP_SHOTS_ON_GOAL&betGroup=PLAYER_PROP#ptoscan"
CODE = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA"}  # PTO -> NHL
BOOKS = {"FAN_DUEL": "FanDuel", "MGM": "BetMGM", "DRAFT_KINGS": "DraftKings", "CAESARS": "Caesars", "PINNACLE": "Pinnacle", "BET_ONLINE_AG": "BetOnline",
         "BET_365": "bet365", "PROPHET_BETTING": "ProphetX", "FLIFF": "Fliff", "THE_SCORE": "theScore", "BET_RIVERS": "BetRivers", "BALLY_BET": "Bally Bet",
         "BOOKMAKER_EU": "Bookmaker", "MY_BOOKIE": "MyBookie", "FANATICS": "Fanatics", "NO_VIG": "Novig", "ONYX_ODDS": "Onyx", "PROP_BUILDER": "PropBuilder",
         "KALSHI": "Kalshi", "HARD_ROCK_IN": "Hard Rock", "CIRCA": "Circa", "REBET": "Rebet", "BET_105": "Bet105"}
CHROME = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"), Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")]
SCALE = "For scale: PTO close 6%+ +6.8% ± 2.2 at the best price (who dressed known, 1,820 bets)."


def scan(timeout_s: float = 210) -> dict | None:
    """The extension's one screen, or None if the port is busy or nothing came back."""
    got, done = {}, threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.update(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            done.set()
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    except OSError:
        print("port busy, skipped", flush=True)
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    chrome = next((str(p) for p in CHROME if p.exists()), shutil.which("chrome"))
    subprocess.Popen([chrome, START])
    done.wait(timeout_s)
    srv.shutdown()
    srv.server_close()
    if not got:
        print(f"no scan after {timeout_s:.0f} s (Chrome closed, or the extension needs Reload)", flush=True)
    return got or None


def odds(x) -> float:
    try:
        o = float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan
    return o if abs(o) >= 100 else np.nan


def implied(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o, 100) / (np.abs(o) + 100)


def long_rows(screen: dict) -> pd.DataFrame:
    """One row per player x book x side."""
    rows = []
    for r in screen.get("rows", []):
        away, home = r["label"].split(" · ")[0].replace(" vs ", " @ ").split(" @ ")
        for book, cell in r["prices"].items():
            for side in ("over", "under"):
                q = (cell or {}).get(side) or {}
                o = odds(q.get("price"))
                if np.isfinite(o) and r.get("line") is not None:
                    rows.append({"away": CODE.get(away.strip(), away.strip()), "home": CODE.get(home.strip(), home.strip()), "player": r["name"],
                                 "line": float(r["line"]), "book": book, "side": side, "odds": o, "size": q.get("size")})
    return pd.DataFrame(rows, columns=["away", "home", "player", "line", "book", "side", "odds", "size"])


def price_game(x: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """One row per priced player: consensus, model chance, best side at the best price."""
    p = p.assign(k=p.key).set_index("key")
    p2 = p.drop_duplicates("key2", keep=False).set_index("key2")
    out = []
    for (player, line), y in x.groupby(["player", "line"]):
        k = norm(player)
        k2 = k[0] + " " + k.split(" ", 1)[1] if " " in k else k
        m = p.loc[k] if k in p.index else p2.loc[k2] if k2 in p2.index else None
        if not isinstance(m, pd.Series):
            continue
        w = y.pivot_table(index="book", columns="side", values="odds", aggfunc="last").reindex(columns=["over", "under"])
        two = w.dropna()
        if len(two):
            po, pu = implied(two.over), implied(two.under)
            cons = float(np.median(po / (po + pu)))
        else:
            cons = np.nan
        p_over = float(np.clip(nbinom.sf(int(np.floor(line)), m.r, m.r / (m.r + m.mu)), .01, .99))
        best = None
        for side, pm in (("over", p_over), ("under", 1 - p_over)):
            o = y[(y.side == side) & y.odds.between(live.MIN_ODDS, live.MAX_ODDS) & ~(y["size"].astype(float) < live.MIN_DEPTH)]
            if o.empty:
                continue
            o = o.iloc[int(np.argmin(implied(o.odds)))]
            edge = pm - float(implied(o.odds))
            if best is None or edge > best["edge"]:
                best = {"side": side, "p_model": pm, "odds": int(o.odds), "price": float(implied(o.odds)), "edge": edge, "book": o.book, "depth": o["size"]}
        out.append({"player_id": m.player_id, "name": m["name"], "team": m.team, "line": line, "mu": m.mu, "cons": cons, "books": len(two), **(best or {})})
    return pd.DataFrame(out)


def post_picks(b: pd.DataFrame, old: pd.DataFrame | None) -> None:
    keys = ["game_id", "player_id", "side", "line"]
    b = b[b.tier == "bet"].drop_duplicates(keys)
    if old is not None:
        b = b[~b.set_index(keys).index.isin(old[old.tier == "bet"].set_index(keys).index)]
    if len(b):
        live.post(["**NHL shots on goal** (paper, best price)"] + [f"{x.name} ({x.team}) {x.side} {x.line:g} {x.odds:+d} {BOOKS.get(x.book, x.book)}, edge {x.edge:+.1%}"
                                                                  for x in b.itertuples()])


def decide(dry: bool = False, file: str | None = None) -> None:
    base = (DATA / "live_pto_dryrun") if dry else LIVE
    base.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=live.ET)
    date = now.strftime("%Y-%m-%d")
    if now.hour < 10 and not dry:
        return
    games = live.slate(date)
    df = base / "decided.csv"
    done = set(map(tuple, pd.read_csv(df)[["game_id", "stage"]].to_numpy())) if df.exists() else set()
    due = []
    for g in games.itertuples():
        mins = (g.start - now).total_seconds() / 60
        if mins > 0 and g.state in ("FUT", "PRE"):
            due += [(g, s) for s in ("early", "late") if (g.game_id, s) not in done and (s == "early" or mins <= live.LATE_MIN)]
    if not due:
        print(f"{now:%H:%M} nothing due", flush=True)
        return
    screen = json.loads(Path(file).read_text(encoding="utf-8")) if file else scan()
    if not screen:
        return
    (base / "raw").mkdir(exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M")
    (base / "raw" / f"{stamp}.json").write_text(json.dumps(screen), encoding="utf-8")
    d = long_rows(screen)
    if d.empty:
        print(f"{now:%H:%M} empty scan", flush=True)
        return
    (base / "lines").mkdir(exist_ok=True)
    d.assign(date=date, ts=now.isoformat()).to_parquet(base / "lines" / f"{stamp}.parquet")
    p, note = live_v3.project(date, games, live_v3.LIVE)
    new, marks, quotes, seen = [], [], [], {}
    for g, stage in due:
        x = d[(d.away == g.away) & (d.home == g.home)]
        if x.player.nunique() < 10:
            continue
        if g.game_id not in seen:
            seen[g.game_id] = price_game(x, p[p.game_id == g.game_id])
            r = seen[g.game_id]
            if len(r):
                quotes.append(pd.DataFrame({"market_id": [f"{g.game_id}_{a}_{b}" for a, b in zip(r.player_id, r.line)], "price_over": r.cons, "price_under": 1 - r.cons, "ts": now.tz_convert("UTC")}))
        r = seen[g.game_id]
        b = r[r.edge >= live.WATCH_EDGE] if "edge" in r else r.iloc[0:0]
        marks.append({"game_id": g.game_id, "stage": stage, "ts": now.isoformat(), "players": x.player.nunique(), "priced": len(r), "picks": len(b)})
        if len(b):
            new.append(b.assign(ts=now.isoformat(), date=date, game_id=g.game_id, start_utc=g.start_utc, stage=stage, tier=np.where(b.edge >= live.MIN_EDGE, "bet", "watch"),
                                market_id=[f"{g.game_id}_{a}_{c}" for a, c in zip(b.player_id, b.line)]))
    if quotes:
        (base / "quotes").mkdir(exist_ok=True)
        q = pd.concat(quotes, ignore_index=True)
        f = base / "quotes" / f"{date}.parquet"
        (pd.concat([pd.read_parquet(f), q], ignore_index=True) if f.exists() else q).to_parquet(f)
    if marks:
        pd.DataFrame(marks).to_csv(df, mode="a", header=not df.exists(), index=False)
    if new:
        f = base / "bets.csv"
        old = pd.read_csv(f) if f.exists() else None
        out = pd.concat(new, ignore_index=True)[live.BET_COLS + ["book", "books", "cons"]]
        out["odds"] = out.odds.astype(int)
        out.to_csv(f, mode="a", header=not f.exists(), index=False)
        if not dry:
            post_picks(out, old)
    print(f"{now:%H:%M} {len(d):,} prices, {d.player.nunique()} players, {note}, {[(m['game_id'], m['stage'], m['priced'], m['picks']) for m in marks]}", flush=True)


def settle() -> None:
    live.LIVE, live.POST, live.TITLE, live.SCALE = LIVE, True, "v3 PTO best price", SCALE
    live.settle()


if __name__ == "__main__":
    a = sys.argv
    {"decide": lambda: decide("--dry-run" in a, a[a.index("--file") + 1] if "--file" in a else None), "settle": settle}[a[1]]()
