"""
elo.py
------
Implements a custom Elo rating system for international football.

Mathematical Background
-----------------------
Elo is a zero-sum rating system originally designed for chess (Arpad Elo,
1960s) and widely adopted for football by FiveThirtyEight, Club Elo, etc.

Core formula:
    R_new = R_old + K * (S - E)

Where:
    R_old  = current rating of the team
    K      = sensitivity constant (how much a single match moves the needle)
    S      = actual result  (1 = win, 0.5 = draw, 0 = loss)
    E      = expected result, derived from the rating difference:
             E = 1 / (1 + 10^((R_opponent - R_team) / 400))

Key design decisions
--------------------
1.  Starting rating: 1500 (universal default; new/unknown teams begin here).
2.  K-factor by match importance:
        World Cup final / semi       → K = 60
        World Cup group / QF         → K = 50
        Other confederation tourneys → K = 40
        Qualifiers                   → K = 30
        (Friendlies excluded upstream, but K=20 if included)
    Higher K for bigger matches means form in tournaments updates
    ratings faster — which is exactly what we want for a World Cup model.
3.  Home advantage: +100 Elo points added to the home team's effective
    rating when computing E (not when neutral=True).
4.  Margin of Victory multiplier (MOV): borrowed from FiveThirtyEight.
    Winning by more goals should shift ratings more.
        MOV_mult = ln(|goal_diff| + 1) * (2.2 / (Δ_Elo * 0.001 + 2.2))
    This prevents already-strong teams from gaining too much from
    thrashing weaker opponents (autocorrelation correction).
"""

import math
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RATING: float = 1500.0
HOME_ADVANTAGE: float = 100.0  # Elo points added to home team

# K-factor lookup — matched against the 'tournament' column string
K_FACTOR_RULES: list[Tuple[int, list[str]]] = [
    (60, ["FIFA World Cup"]),
    (50, ["UEFA Euro", "Copa América", "Africa Cup of Nations",
          "AFC Asian Cup", "CONCACAF Gold Cup"]),
    (40, ["UEFA Nations League", "AFC Challenge Cup",
          "COSAFA Cup", "CECAFA Cup"]),
    (30, ["FIFA World Cup qualification"]),
]
DEFAULT_K: int = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_k(tournament: str) -> int:
    """Return the K-factor for a given tournament name."""
    for k, names in K_FACTOR_RULES:
        if any(name.lower() in tournament.lower() for name in names):
            return k
    return DEFAULT_K


def _expected_score(rating_a: float, rating_b: float) -> float:
    """
    Compute the expected score for team A against team B.

    Returns a probability in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier(goal_diff: int, elo_diff: float) -> float:
    """
    Margin-of-Victory multiplier (FiveThirtyEight methodology).

    Larger winning margins increase the Elo shift, but with diminishing
    returns and a correction for lopsided matchups.

    Parameters
    ----------
    goal_diff : int
        Absolute goal difference (always positive).
    elo_diff : float
        Elo difference of winner minus loser BEFORE the match.
    """
    return math.log(goal_diff + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


# ---------------------------------------------------------------------------
# Main Elo engine
# ---------------------------------------------------------------------------

def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through every match chronologically and compute rolling Elo ratings.

    For each match we:
      1. Look up both teams' current ratings (defaulting to 1500).
      2. Apply home-field advantage to the effective ratings.
      3. Compute each team's expected score E.
      4. Determine the actual result S.
      5. Apply the MOV multiplier.
      6. Update both teams' ratings in the registry.
      7. Record the PRE-MATCH ratings as features on the row
         (important: we use ratings BEFORE the match to avoid leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned match dataframe from data_loader.clean().

    Returns
    -------
    pd.DataFrame
        Original dataframe with four new columns appended:
          - home_elo_before  : home team Elo at kick-off
          - away_elo_before  : away team Elo at kick-off
          - home_elo_after   : home team Elo post-match (for audit trail)
          - away_elo_after   : away team Elo post-match (for audit trail)
    """
    # Mutable registry: team_name → current Elo
    ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_RATING)

    home_elo_before, away_elo_before = [], []
    home_elo_after, away_elo_after = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", ""))
        k = _get_k(tournament)

        # --- Pre-match ratings (what we'll use as features) ---
        r_home = ratings[home]
        r_away = ratings[away]
        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        # --- Effective ratings for E calculation ---
        eff_home = r_home + (0.0 if neutral else HOME_ADVANTAGE)
        eff_away = r_away  # no away advantage

        e_home = _expected_score(eff_home, eff_away)
        e_away = 1.0 - e_home

        # --- Actual result ---
        h_goals = row["home_score"]
        a_goals = row["away_score"]

        if h_goals > a_goals:
            s_home, s_away = 1.0, 0.0
        elif h_goals < a_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # --- Margin of Victory multiplier ---
        goal_diff = abs(h_goals - a_goals)
        if goal_diff == 0:
            mov = 1.0
        else:
            winner_elo = r_home if s_home == 1.0 else r_away
            loser_elo  = r_away if s_home == 1.0 else r_home
            mov = _mov_multiplier(goal_diff, winner_elo - loser_elo)

        # --- Rating update ---
        ratings[home] = r_home + k * mov * (s_home - e_home)
        ratings[away] = r_away + k * mov * (s_away - e_away)

        home_elo_after.append(ratings[home])
        away_elo_after.append(ratings[away])

    df = df.copy()
    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    df["home_elo_after"]  = home_elo_after
    df["away_elo_after"]  = away_elo_after

    print(f"[elo] Elo ratings computed for {len(ratings)} teams.")
    return df, dict(ratings)
