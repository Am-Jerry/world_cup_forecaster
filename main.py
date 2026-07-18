"""
main.py  —  Phase 1 + Phase 2 + Phase 3 entry point
-----------------------------------------------------
Run this script to execute the full pipeline:
  1. Ingest historical + 2026 tournament data, compute Elo, build features.
  2. Train the xG model.
  3. Run a LIVE forecast from the tournament's real current position —
     every completed 2026 match (group stage through the two played
     quarterfinals) is locked in as-is; only the genuinely undecided
     remainder (2 pending QFs, both SFs, the Final) is Monte Carlo
     simulated, 10,000 times.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import load_raw, clean
from src.elo import compute_elo_ratings
from src.features import build_feature_matrix
from src.model import train_xg_model, predict_lambda
from src.poisson import score_matrix, match_outcome_probs, sample_score
from src.simulator import run_monte_carlo_2026


OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
FEATURES_PATH = OUTPUT_DIR / "features.csv"


def run_phase1() -> pd.Series:
    """Ingest historical + 2026 data, compute Elo, build feature matrix."""
    print("=== Phase 1: Data Ingestion & Feature Engineering ===\n")

    raw_df   = load_raw()  # merges data/results.csv + data/results_2026.csv
    clean_df = clean(raw_df, start_year=2000, remove_friendlies=True)

    elo_df, final_ratings = compute_elo_ratings(clean_df)

    elo_series = (
        pd.Series(final_ratings, name="elo")
        .rename_axis("team")
        .reset_index()
        .sort_values("elo", ascending=False)
    )
    elo_series.to_csv(OUTPUT_DIR / "final_elo_ratings.csv", index=False)

    features_df = build_feature_matrix(elo_df)
    features_df.to_csv(FEATURES_PATH, index=False)

    print(f"\n[main] Phase 1 outputs saved to {OUTPUT_DIR}/")
    print("\n--- Top 10 teams by Elo (post-2026 form) ---")
    print(elo_series.head(10).to_string(index=False))
    return elo_series


def run_phase2() -> None:
    """Train xG model and run sanity check."""
    print("\n=== Phase 2: xG Model + Poisson Engine ===\n")

    model = train_xg_model(FEATURES_PATH)

    features_df = pd.read_csv(FEATURES_PATH)

    def get_team_latest(team: str) -> dict:
        rows = features_df[features_df["team"] == team]
        if rows.empty:
            raise ValueError(f"Team '{team}' not found in feature matrix.")
        return rows.iloc[-1]

    try:
        france = get_team_latest("France")
        spain  = get_team_latest("Spain")
    except ValueError as e:
        print(f"[main] WARNING: {e} — skipping sanity check.")
        return

    lambda_france = predict_lambda(
        model,
        elo_for=france["elo_for"], elo_against=spain["elo_for"], is_home=0,
        rolling_scored_5=france["rolling_scored_5"],
        rolling_conceded_5=france["rolling_conceded_5"],
        rolling_scored_10=france["rolling_scored_10"],
        rolling_conceded_10=france["rolling_conceded_10"],
    )
    lambda_spain = predict_lambda(
        model,
        elo_for=spain["elo_for"], elo_against=france["elo_for"], is_home=0,
        rolling_scored_5=spain["rolling_scored_5"],
        rolling_conceded_5=spain["rolling_conceded_5"],
        rolling_scored_10=spain["rolling_scored_10"],
        rolling_conceded_10=spain["rolling_conceded_10"],
    )

    matrix = score_matrix(lambda_france, lambda_spain)
    p_home, p_draw, p_away = match_outcome_probs(matrix)

    print(f"\n[sanity] France λ = {lambda_france:.3f}")
    print(f"[sanity] Spain  λ = {lambda_spain:.3f}")
    print(f"\n[sanity] France win: {p_home*100:.1f}% | "
          f"Draw: {p_draw*100:.1f}% | "
          f"Spain win: {p_away*100:.1f}%")

    scorelines = [(h, a, matrix[h][a]) for h in range(6) for a in range(6)]
    scorelines.sort(key=lambda x: x[2], reverse=True)
    print("\n[sanity] Top 5 scorelines:")
    for h, a, prob in scorelines[:5]:
        print(f"  {h}-{a}  →  {prob*100:.2f}%")


def run_phase3() -> None:
    """Run the live real-state-aware Monte Carlo forecast."""
    print("\n=== Phase 3: Live Monte Carlo Forecast (from real Quarterfinal stage) ===\n")

    results_df = run_monte_carlo_2026(n_simulations=10_000)

    print("\n--- Remaining contenders (the 8 real quarterfinalists) ---")
    alive = results_df[results_df["Winner%"] > 0].sort_values("Winner%", ascending=False)
    print(
        alive[["team", "QF%", "SF%", "Final%", "Winner%"]].to_string(index=False)
    )
    print("\n=== Phase 3 complete. ===")


if __name__ == "__main__":
    run_phase1()
    run_phase2()
    run_phase3()
