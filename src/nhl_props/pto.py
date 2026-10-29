"""PTO (picktheodds.app) closing lines of every book for finished NHL games: shots on goal + goalie saves.

    uv run python -m nhl_props.pto serve    # receiver on 127.0.0.1:8766 (the PTO extension's fixed port)
    uv run python -m nhl_props.pto build    # raw -> data/pto/<SOG|SAVES>.parquet (every line, the 20 books)

The page script `scripts/pto_pull.js` (paste into a PTO tab via the Chrome tools) calls the page's own Apollo client
(it signs each request; replays get 401) for completedBetMarket(gameId, market), one game x market at a time, and
posts rows through the NBA PTO extension (../NBAProps/pto-scanner/extension -> port 8766). Method and traps:
../NBAProps/CLAUDE.md, "PTO CLOSING ODDS VIA THE PAGE'S OWN API".
🔴 Port 8766 is shared with the NBA daily snapshots: the receiver closes the port during BLACKOUT windows and the page
script pauses during the same windows (both read the list passed on the command line / in PTO_PULL).
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "pto" / "raw"
OUT = ROOT / "data" / "pto"
PORT = 8766
MARKETS = {"PLAYER_PROP_SHOTS_ON_GOAL": "SOG", "PLAYER_PROP_SHOTS_SAVED": "SAVES"}
COLS = ["player_id_pto", "side", "line", "book", "odds", "size", "is_winner", "is_push"]


def _blocked(blackout, now=None) -> bool:
    now = now or time.time()
    return any(a <= now < b for a, b in blackout)


def serve(blackout=()) -> None:
    (RAW / "odds").mkdir(parents=True, exist_ok=True)
    done, lock, n = threading.Event(), threading.Lock(), {"gm": 0}

    def save_table(name, rows):
        f = RAW / f"{name}.parquet"
        new = pd.DataFrame(rows)
        old = pd.read_parquet(f) if f.exists() else None
        (pd.concat([old, new]) if old is not None else new).drop_duplicates("id", keep="last").to_parquet(f)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if msg.get("raw") and msg.get("nhl"):
                with lock:
                    k = msg["kind"]
                    if k in ("games", "players"):
                        save_table(k, msg[k])
                        print(f"{k}: +{len(msg[k])}", flush=True)
                    elif k == "odds":
                        d = pd.DataFrame(msg["rows"], columns=COLS).assign(game_id=msg["game_id"], group=msg["market"])
                        d.to_parquet(RAW / "odds" / f"{msg['market']}__{msg['game_id']}.parquet")
                        n["gm"] += 1
                        if n["gm"] % 50 == 0:
                            print(f"odds: {n['gm']} game-markets this run ({msg.get('note', '')})", flush=True)
                    elif k == "done":
                        print("done", flush=True)
                        done.set()
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = None
    t_end = time.time() + 12 * 3600
    while not done.is_set() and time.time() < t_end:
        if _blocked(blackout) and srv:
            srv.shutdown()
            srv.server_close()
            srv = None
            print(f"{time.strftime('%H:%M')} blackout: port {PORT} released for the NBA snapshot", flush=True)
        elif not _blocked(blackout) and not srv:
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
            except OSError as e:  # the NBA snapshot holds it: try again shortly
                print(f"{time.strftime('%H:%M')} port busy ({e}), retrying", flush=True)
                time.sleep(30)
                continue
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            print(f"{time.strftime('%H:%M')} listening on {PORT}", flush=True)
        time.sleep(10)
    if srv:
        srv.shutdown()


def build() -> None:
    """Every book's closing price on every line, plus game start / teams and player names."""
    games = pd.read_parquet(RAW / "games.parquet").set_index("id")
    names = pd.read_parquet(RAW / "players.parquet").set_index("id")["name"]
    d = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "odds").glob("*.parquet"))], ignore_index=True)
    d = d[d.odds.notna()]
    d["odds"] = d.odds.astype(int)
    d = d.join(games, on="game_id")
    d["player"] = d.player_id_pto.map(names)
    d["market"] = d.group.map(MARKETS)
    d["start"] = pd.to_datetime(d.start, unit="s")
    for mk, x in d.groupby("market"):
        x = x.drop(columns="group").drop_duplicates(["game_id", "player_id_pto", "line", "book", "side"], keep="last")
        x.to_parquet(OUT / f"{mk}.parquet")
        print(f"{mk}: {x.game_id.nunique()} games ({x.start.min():%Y-%m-%d} -> {x.start.max():%Y-%m-%d}), "
              f"{x.groupby(['game_id', 'player_id_pto']).ngroups:,} player-games, {len(x):,} prices, "
              f"{x.player.isna().mean():.1%} unnamed")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:  # serve [start_unix:end_unix ...]
        serve([tuple(map(float, w.split(":"))) for w in sys.argv[2:]])
