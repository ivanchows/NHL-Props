"""Raw game JSON -> one row per skater-game and per goalie-game.

Usage: uv run python -m nhl_props.data
Writes data/skater_games.parquet and data/goalie_games.parquet from every game with box + pbp on disk.
Shot attempts come from play-by-play; sog/toi/saves come from the box score (what props settle on).
"""

import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
ATTEMPTS = {"shot-on-goal": "pbp_sog", "goal": "pbp_goals", "missed-shot": "missed", "blocked-shot": "blocked"}


def _load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _mmss(s: str) -> float:
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def parse_game(box_path: Path):
    gid = box_path.name[:10]
    pbp_path = box_path.with_name(f"{gid}_pbp.json.gz")
    if not pbp_path.exists():
        return [], []
    box, pbp = _load(box_path), _load(pbp_path)

    # Shot attempts per shooter. A blocked shot's shooter is shootingPlayerId; blockingPlayerId is the defender.
    att: dict[int, Counter] = {}
    for p in pbp["plays"]:
        kind = ATTEMPTS.get(p["typeDescKey"])
        if not kind or p.get("periodDescriptor", {}).get("periodType") == "SO":
            continue
        d = p.get("details", {})
        pid = d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if pid:
            att.setdefault(pid, Counter())[kind] += 1

    common = {
        "game_id": int(gid),
        "season": int(gid[:4]),
        "game_date": box["gameDate"],
        "start_utc": box.get("startTimeUTC"),
    }
    skaters, goalies = [], []
    for side, opp in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        team, opp_team = box[side]["abbrev"], box[opp]["abbrev"]
        stats = box["playerByGameStats"][side]
        base = {**common, "team": team, "opp": opp_team, "home": side == "homeTeam"}
        for group in ("forwards", "defense"):
            for s in stats.get(group, []):
                a = att.get(s["playerId"], Counter())
                skaters.append({
                    **base,
                    "player_id": s["playerId"],
                    "name": s["name"]["default"],
                    "pos": s["position"],
                    "toi": _mmss(s["toi"]),
                    "shifts": s.get("shifts"),
                    "sog": s["sog"],
                    "goals": s["goals"],
                    "assists": s["assists"],
                    "pp_goals": s.get("powerPlayGoals", 0),
                    "hits": s.get("hits", 0),
                    "blocks": s.get("blockedShots", 0),
                    "pbp_sog": a["pbp_sog"] + a["pbp_goals"],
                    "missed": a["missed"],
                    "blocked_att": a["blocked"],
                })
        for g in stats.get("goalies", []):
            goalies.append({
                **base,
                "player_id": g["playerId"],
                "name": g["name"]["default"],
                "toi": _mmss(g["toi"]),
                "starter": bool(g.get("starter")),
                "shots_against": g.get("shotsAgainst", 0),
                "saves": g.get("saves", 0),
                "goals_against": g.get("goalsAgainst", 0),
                "decision": g.get("decision"),
            })
    return skaters, goalies


def build() -> None:
    boxes = sorted(RAW.glob("*/*_box.json.gz"))
    sk, gl = [], []
    with ProcessPoolExecutor() as ex:
        for s, g in ex.map(parse_game, boxes, chunksize=64):
            sk += s
            gl += g
    skaters = pd.DataFrame(sk).sort_values(["game_date", "game_id", "player_id"], ignore_index=True)
    goalies = pd.DataFrame(gl).sort_values(["game_date", "game_id", "player_id"], ignore_index=True)
    skaters.to_parquet(OUT / "skater_games.parquet")
    goalies.to_parquet(OUT / "goalie_games.parquet")
    print(f"{skaters.game_id.nunique()} games, {len(skaters)} skater rows, {len(goalies)} goalie rows")


if __name__ == "__main__":
    build()
