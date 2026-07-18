"""
features.py
-----------
Builds the final feature matrix used to train the xG regression model
in Phase 2.

Features engineered per match-team observation
----------------------------------------------
For each match we create TWO rows: one from the home team's perspective
and one from the away team's perspective.  This gives the model symmetry
and doubles our training sample.

Feature                  Description
-----------------------  -------------------------------------------------
elo_for                  Elo rating of the team in question (pre-match)
elo_against              Elo rating of the opponent (pre-match)
elo_diff                 elo_for - elo_against  (signed strength gap)
is_home                  1 if the team is playing at home, 0 otherwise
rolling_scored_5         Rolling mean goals scored over last 5 matches
rolling_conceded_5       Rolling mean goals conceded over last 5 matches
rolling_scored_10        Rolling mean goals scored over last 10 matches
rolling_conceded_10      Rolling mean goals conceded over last 10 matches
goals_scored             TARGET — actual goals scored in this match
"""

import pandas as pd
import numpy as np
from typing import Dict


def _compute_rolling_stats(
    df: pd.DataFrame,
    windows: list[int] = [5, 10],
) -> pd.DataFrame:
    """
    For every team and every match, calculate rolling average goals
    scored and conceded over the N most recent matches.

    We compute this in 'long form': reshape the match dataframe so each
    row represents one team's experience in one match, sort chronologically
    per team, then use pandas .rolling() on the shifted series.

    The critical detail is .shift(1) BEFORE .rolling() — this ensures we
    only use PAST matches, never the current one (no data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Match dataframe with Elo columns attached.
    windows : list[int]
        Rolling window sizes in number of matches.

    Returns
    -------
    pd.DataFrame
        Long-form dataframe: one row per team per match.
    """
    records = []

    for _, row in df.iterrows():
        # Home team perspective
        records.append({
            "date":       row["date"],
            "team":       row["home_team"],
            "opponent":   row["away_team"],
            "is_home":    1,
            "elo_for":    row["home_elo_before"],
            "elo_against":row["away_elo_before"],
            "goals_scored":   row["home_score"],
            "goals_conceded": row["away_score"],
            "neutral":    row.get("neutral", False),
        })
        # Away team perspective
        records.append({
            "date":       row["date"],
            "team":       row["away_team"],
            "opponent":   row["home_team"],
            "is_home":    0,
            "elo_for":    row["away_elo_before"],
            "elo_against":row["home_elo_before"],
            "goals_scored":   row["away_score"],
            "goals_conceded": row["home_score"],
            "neutral":    row.get("neutral", False),
        })

    long_df = pd.DataFrame(records).sort_values(["team", "date"])
    long_df.reset_index(drop=True, inplace=True)

    # --- Rolling stats (shift first to prevent leakage) ---
    for w in windows:
        long_df[f"rolling_scored_{w}"] = (
            long_df.groupby("team")["goals_scored"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        long_df[f"rolling_conceded_{w}"] = (
            long_df.groupby("team")["goals_conceded"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )

    return long_df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate feature engineering and return the final feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Match dataframe with Elo columns (output of elo.compute_elo_ratings).

    Returns
    -------
    pd.DataFrame
        Clean feature matrix ready for model training in Phase 2.
    """
    long_df = _compute_rolling_stats(df)

    long_df["elo_diff"] = long_df["elo_for"] - long_df["elo_against"]

    # Drop rows where rolling stats couldn't be computed (very early rows)
    long_df.dropna(inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    feature_cols = [
        "date", "team", "opponent", "is_home",
        "elo_for", "elo_against", "elo_diff",
        "rolling_scored_5", "rolling_conceded_5",
        "rolling_scored_10", "rolling_conceded_10",
        "goals_scored",   # TARGET
    ]

    print(f"[features] Feature matrix shape: {long_df[feature_cols].shape}")
    return long_df[feature_cols]
