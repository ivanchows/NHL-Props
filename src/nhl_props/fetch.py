"""Raw NHL game data (api-web.nhle.com + api.nhle.com shift charts), one gzipped JSON per game and feed.

Usage: uv run python -m nhl_props.fetch 2021 2026
Season 2021 = 2021-22. Regular season only (game ids <season>02NNNN).
Writes data/raw/<season>/<gameId>_{pbp,box,shifts}.json.gz. Resumes: a game with all three files is skipped.
Unfinished games (gameState not OFF/FINAL) are not saved, so rerunning fills them in.
"""

import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WEB = "https://api-web.nhle.com/v1/gamecenter/{gid}/{feed}"
SHIFTS = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
FEEDS = ("pbp", "box", "shifts")
MAX_GAMES = 1312  # 32 teams x 82 / 2
WORKERS = 6
MIN_GAP = 1 / 8  # <= 8 requests/s overall

_lock = threading.Lock()
_last = [0.0]
_local = threading.local()


def _get(url: str):
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    for attempt in range(9):
        with _lock:
            wait = _last[0] + MIN_GAP - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = s.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(min(2**attempt, 60))
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(2**attempt, 60))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up: {url}")


def _path(season: int, gid: str, feed: str) -> Path:
    return RAW / str(season) / f"{gid}_{feed}.json.gz"


def _save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    tmp.replace(path)


def fetch_game(season: int, n: int) -> str:
    gid = f"{season}02{n:04d}"
    if all(_path(season, gid, f).exists() for f in FEEDS):
        return "have"
    if not (_path(season, gid, "box").exists() and _path(season, gid, "pbp").exists()):
        box = _get(WEB.format(gid=gid, feed="boxscore"))
        if box is None:
            return "missing"
        if box.get("gameState") not in ("OFF", "FINAL"):
            return "unplayed"
        pbp = _get(WEB.format(gid=gid, feed="play-by-play"))
        if pbp is None:
            return "missing"
        _save(_path(season, gid, "box"), box)
        _save(_path(season, gid, "pbp"), pbp)
    shifts = _get(SHIFTS.format(gid=gid))
    if not shifts or not shifts.get("data"):
        # Shift charts lag the final horn, and some games never get them (2026020090-151).
        # Box TOI covers those; each rerun retries only the shifts.
        return "noshifts"
    _save(_path(season, gid, "shifts"), shifts)
    return "saved"


def _try_game(season: int, n: int) -> str:
    # One game that keeps failing must not kill the season; a rerun retries it.
    try:
        return fetch_game(season, n)
    except RuntimeError as e:
        print(e, flush=True)
        return "error"


def fetch_season(season: int) -> None:
    (RAW / str(season)).mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, status in enumerate(ex.map(lambda n: _try_game(season, n), range(1, MAX_GAMES + 1)), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0:
                print(f"{season}: {i}/{MAX_GAMES} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"{season} done: {counts} {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str]) -> None:
    first, last = int(argv[0]), int(argv[1] if len(argv) > 1 else argv[0])
    for season in range(first, last + 1):
        fetch_season(season)


if __name__ == "__main__":
    main(sys.argv[1:])
