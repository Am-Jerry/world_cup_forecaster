"""
bracket_data.py
----------------
Builds the full, real tournament picture — every group's final standings
and every knockout round's actual results — from data/results_2026.csv.
This is what powers the Groups page and the Path to the Final bracket view.

Like everything else in this app, this is READ-ONLY: it reads the two
static CSVs (results_2026.csv, simulation_results.csv) plus the constant
GROUPS/QUARTERFINALISTS_2026/QF_BRACKET_2026 from src/tournament.py. No
model inference, no simulation, happens at runtime.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.tournament import GROUPS, QUARTERFINALISTS_2026, QF_BRACKET_2026

RESULTS_2026_PATH = Path("data/results_2026.csv")

ROUND_ORDER = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABELS = {
    "R32": "Round of 32", "R16": "Round of 16",
    "QF": "Quarterfinals", "SF": "Semifinals", "Final": "Final",
}


@st.cache_data
def load_2026_results() -> pd.DataFrame:
    if not RESULTS_2026_PATH.exists():
        st.error("data/results_2026.csv not found.")
        st.stop()
    return pd.read_csv(RESULTS_2026_PATH)


@st.cache_data
def group_standings() -> dict:
    """
    Compute REAL final group-stage standings (W/D/L/GF/GA/Pts) for every
    group, from the actual played matches — not a forecast, an accounting
    of results that already happened.

    Returns
    -------
    dict: group_letter -> list of team-stat dicts, sorted by points (desc,
    with goal difference as the tiebreaker — a simplification of the full
    FIFA rule set, sufficient for DISPLAY since the real official
    standings are already baked into who actually advanced).
    """
    df = load_2026_results()
    group_df = df[df["stage"] == "Group"]

    standings = {}
    for letter, teams in GROUPS.items():
        stats = {t: {"team": t, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for t in teams}
        matches = group_df[group_df["home_team"].isin(teams) & group_df["away_team"].isin(teams)]
        for _, row in matches.iterrows():
            h, a = row["home_team"], row["away_team"]
            hs, as_ = row["home_score"], row["away_score"]
            stats[h]["gf"] += hs
            stats[h]["ga"] += as_
            stats[a]["gf"] += as_
            stats[a]["ga"] += hs
            if hs > as_:
                stats[h]["w"] += 1; stats[h]["pts"] += 3; stats[a]["l"] += 1
            elif hs < as_:
                stats[a]["w"] += 1; stats[a]["pts"] += 3; stats[h]["l"] += 1
            else:
                stats[h]["d"] += 1; stats[h]["pts"] += 1
                stats[a]["d"] += 1; stats[a]["pts"] += 1
        ranked = sorted(
            stats.values(),
            key=lambda s: (s["pts"], s["gf"] - s["ga"], s["gf"]),
            reverse=True,
        )
        standings[letter] = ranked
    return standings


def _match_winner(df: pd.DataFrame, stage: str, a: str, b: str):
    """Return (winner, loser, score_a, score_b) for a completed match, or None if not played."""
    rows = df[df["stage"] == stage]
    match = rows[
        ((rows["home_team"] == a) & (rows["away_team"] == b)) |
        ((rows["home_team"] == b) & (rows["away_team"] == a))
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    h, aw = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    winner = h if hs > as_ else aw
    loser = aw if winner == h else h
    return {"winner": winner, "loser": loser, "score": (int(hs), int(as_)) if winner == h else (int(as_), int(hs))}


@st.cache_data
def build_full_bracket() -> dict:
    """
    Build every round of the real knockout bracket, from Round of 32
    through the Final, using actual results where played and leaving a
    match "pending" where it hasn't happened yet.

    Returns
    -------
    dict: round_key -> list of match dicts:
        {t1, t2, score (or None), winner (or None), status}
    status is one of "FT" (played) or "UPCOMING".
    """
    df = load_2026_results()

    # --- Round of 32: derive winners/losers directly from the data ---
    r32_rows = df[df["stage"] == "R32"]
    r32_matches = []
    for _, row in r32_rows.iterrows():
        h, a, hs, as_ = row["home_team"], row["away_team"], row["home_score"], row["away_score"]
        winner = h if hs > as_ else a
        r32_matches.append({
            "t1": h, "t2": a, "score": (int(hs), int(as_)), "winner": winner, "status": "FT",
        })

    # --- Round of 16: same approach ---
    r16_rows = df[df["stage"] == "R16"]
    r16_matches = []
    for _, row in r16_rows.iterrows():
        h, a, hs, as_ = row["home_team"], row["away_team"], row["home_score"], row["away_score"]
        winner = h if hs > as_ else a
        r16_matches.append({
            "t1": h, "t2": a, "score": (int(hs), int(as_)), "winner": winner, "status": "FT",
        })

    # --- Quarterfinals: fixed bracket from tournament.py, all 4 decided ---
    qf_matches = []
    for a, b in QF_BRACKET_2026:
        result = _match_winner(df, "QF", a, b)
        if result:
            qf_matches.append({
                "t1": a, "t2": b, "score": result["score"], "winner": result["winner"], "status": "FT",
            })
        else:
            qf_matches.append({"t1": a, "t2": b, "score": None, "winner": None, "status": "UPCOMING"})

    # --- Semifinals: derived from QF winners ---
    qf_winners = [m["winner"] for m in qf_matches]
    sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_matches = []
    for a, b in sf_pairs:
        if a is None or b is None:
            sf_matches.append({"t1": a, "t2": b, "score": None, "winner": None, "status": "UPCOMING"})
            continue
        result = _match_winner(df, "SF", a, b)
        if result:
            sf_matches.append({
                "t1": a, "t2": b, "score": result["score"], "winner": result["winner"], "status": "FT",
            })
        else:
            sf_matches.append({"t1": a, "t2": b, "score": None, "winner": None, "status": "UPCOMING"})

    # --- Final ---
    sf_winners = [m["winner"] for m in sf_matches]
    final_matches = [{
        "t1": sf_winners[0], "t2": sf_winners[1], "score": None, "winner": None, "status": "UPCOMING",
    }]

    return {
        "R32": r32_matches, "R16": r16_matches, "QF": qf_matches,
        "SF": sf_matches, "Final": final_matches,
    }
