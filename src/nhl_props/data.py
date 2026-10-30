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


def _sit(code, home: bool):
    """Own skaters, opposing skaters for one side, or None on an empty net. situationCode = away goalie, away skaters,
    home skaters, home goalie."""
    if not code or len(code) != 4 or code[0] != "1" or code[3] != "1":
        return None
    away, hme = int(code[1]), int(code[2])
    return (hme, away) if home else (away, hme)


def pp_windows(pbp, home: bool) -> dict[int, list[tuple[float, float]]]:
    """period -> [(start, end) in seconds] of this side's man advantage. The state between two plays is the later
    play's, so a penalty's time runs from the faceoff after it. Expiry is only seen at the next play."""
    out: dict[int, list[tuple[float, float]]] = {}
    prev = None
    for p in pbp["plays"]:
        per = p["periodDescriptor"]["number"]
        if p["periodDescriptor"].get("periodType") == "SO":
            continue
        t = _mmss(p["timeInPeriod"]) * 60
        if prev and prev[0] == per and t > prev[1]:
            st = _sit(p.get("situationCode"), home)
            if st and st[0] > st[1]:
                out.setdefault(per, []).append((prev[1], t))
        prev = (per, t)
    return out


def pp_toi(shifts_path: Path, pbp, team_ids: dict[bool, int]):
    """player_id -> power-play minutes, from shift charts. Empty when the charts are missing for this game."""
    if not shifts_path.exists():
        return None
    rows = _load(shifts_path)
    rows = rows["data"] if isinstance(rows, dict) else rows
    if not rows:
        return None
    win = {tid: pp_windows(pbp, home) for home, tid in team_ids.items()}
    out: Counter = Counter()
    for r in rows:
        if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
            continue
        a, b = _mmss(r["startTime"]) * 60, _mmss(r["endTime"]) * 60
        for x, y in win.get(r["teamId"], {}).get(r["period"], ()):
            lap = min(b, y) - max(a, x)
            if lap > 0:
                out[r["playerId"]] += lap / 60
    return out


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
            st = _sit(p.get("situationCode"), d.get("eventOwnerTeamId") == box["homeTeam"]["id"])
            if st and st[0] > st[1] and kind != "blocked":  # a blocked shot's owner is the blocking team
                att[pid]["pp_att"] += 1

    common = {
        "game_id": int(gid),
        "season": int(gid[:4]),
        "game_date": box["gameDate"],
        "start_utc": box.get("startTimeUTC"),
    }
    skaters, goalies = [], []
    pp = pp_toi(box_path.with_name(f"{gid}_shifts.json.gz"), pbp, {True: box["homeTeam"]["id"], False: box["awayTeam"]["id"]})
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
                    "pp_att": a["pp_att"],
                    "pp_toi": None if pp is None else pp[s["playerId"]],
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
