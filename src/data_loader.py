"""
data_loader.py
--------------
Responsibilities:
  - Load the raw historical CSV from disk, plus the live 2026 World Cup
    results supplement, and merge them into a single chronological dataset.
  - Normalise team name spellings so the two files refer to the same team
    with the same string (see TEAM_NAME_MAP below — this is what makes
    Elo continuity actually work).
  - Parse dates and enforce correct dtypes.
  - Filter to a configurable cutoff year (default: 2000).
  - Remove low-signal Friendly matches.

IMPORTANT — Team name normalisation, and which direction it runs:
  src/tournament.py's GROUPS dict and data/results_2026.csv both use one
  fixed spelling per team (e.g. "United States", "South Korea", "Czech
  Republic", "Curacao"). That spelling is the standard the rest of the
  pipeline is built around — it's what load_teams() in simulator.py looks
  up in the Elo dictionary.

  So normalisation only ever needs to run ONE way: any team name in your
  historical results.csv that's spelled differently gets rewritten to
  match OUR standard spelling. Never the other way around — renaming a
  2026 team away from what GROUPS/results_2026.csv already use just
  breaks the lookup instead of fixing it.

  TEAM_NAME_MAP keys = the spelling found in results.csv.
  TEAM_NAME_MAP values = our standard spelling (must match GROUPS exactly).

  Run check_team_names.py first to get the real list for your file — the
  entries below are common candidates, not a verified list for your
  specific results.csv. Confirm/adjust against that script's output
  before trusting this.
"""

import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "results.csv"
DATA_PATH_2026 = DATA_DIR / "results_2026.csv"

FRIENDLY_LABELS = {"Friendly"}

# Keys: spelling as it appears in your historical results.csv.
# Values: our standard spelling, matching src/tournament.py's GROUPS and
# data/results_2026.csv exactly.
#
# VERIFIED against Jerry's actual results.csv (47 of 48 real 2026 teams
# already matched exactly — only Curaçao/Curacao needed normalising).
# If you add or update data/results.csv later, rerun check_team_names.py
# and extend this map if it finds anything new.
TEAM_NAME_MAP = {
    "Curaçao": "Curacao",
}


def _normalise_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rewrite any team name in df that matches a TEAM_NAME_MAP key into our
    standard spelling. Applied to the HISTORICAL dataframe only — the 2026
    supplement is already written in our standard spelling by construction.
    """
    df = df.copy()
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
    return df


def load_raw(path: Path = DATA_PATH, path_2026: Path = DATA_PATH_2026) -> pd.DataFrame:
    """
    Load the historical CSV and the 2026 World Cup supplement, normalise
    team names, merge them, and sort chronologically.

    Parameters
    ----------
    path : Path
        Location of the historical results.csv on disk.
    path_2026 : Path
        Location of the 2026 World Cup results supplement. If this file
        doesn't exist yet, it's silently skipped.

    Returns
    -------
    pd.DataFrame
        Combined raw dataframe with a parsed datetime column, sorted
        chronologically, with team names normalised to one consistent
        spelling per team.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = _normalise_team_names(df)  # <- fixes historical spellings, not 2026's

    if path_2026.exists():
        df_2026 = pd.read_csv(path_2026, parse_dates=["date"])
        before = len(df)
        df = pd.concat([df, df_2026], ignore_index=True)
        print(f"[data_loader] Merged {len(df_2026)} rows from "
              f"{path_2026.name} ({before:,} -> {len(df):,} total matches)")
    else:
        print(f"[data_loader] No {path_2026.name} found — skipping 2026 supplement")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean(
    df: pd.DataFrame,
    start_year: int = 2000,
    remove_friendlies: bool = True,
) -> pd.DataFrame:
    """
    Filter and clean the raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw().
    start_year : int
        Discard matches played before this calendar year.
    remove_friendlies : bool
        If True, drop rows whose tournament type is in FRIENDLY_LABELS.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = df[df["date"].dt.year >= start_year].copy()

    critical_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    df.dropna(subset=critical_cols, inplace=True)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if remove_friendlies:
        df = df[~df["tournament"].isin(FRIENDLY_LABELS)]

    df.reset_index(drop=True, inplace=True)

    print(f"[data_loader] Loaded {len(df):,} matches after cleaning "
          f"(start_year={start_year}, remove_friendlies={remove_friendlies})")
    return df
