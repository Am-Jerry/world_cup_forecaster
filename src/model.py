"""
model.py
--------
Trains an XGBoost regression model to predict Expected Goals (xG) —
the λ parameter of the Poisson distribution — for a given team in a
given matchup.

Mathematical Framing
--------------------
We treat goal-scoring as a Poisson process:
    Goals ~ Poisson(λ)

where λ (the rate parameter) is the expected number of goals a team
will score.  Our job here is to estimate λ from observable pre-match
features using a supervised regression model.

Why XGBoost?
------------
- Football goal data is low-count, right-skewed, and non-linear.
  Tree-based models handle this naturally without needing
  log-transforms or distributional assumptions.
- XGBoost with 'reg:squarederror' minimises MSE, which works well
  for count-like targets in the 0–5 range.
- An alternative would be a Poisson GLM (which directly models count
  data), but XGBoost typically captures interaction effects better
  (e.g. a high-Elo team WITH strong recent form vs. just either alone).

We train ONE model that predicts goals scored by the 'team' column.
To get λ for BOTH teams in a fixture, we call the model twice —
once per team — passing that team's features as inputs.

Feature set (from features.csv)
--------------------------------
  elo_for            : team's own Elo
  elo_against        : opponent's Elo
  elo_diff           : signed gap (elo_for - elo_against)
  is_home            : home advantage flag
  rolling_scored_5   : attack form (short window)
  rolling_conceded_5 : defensive form (short window)
  rolling_scored_10  : attack form (long window)
  rolling_conceded_10: defensive form (long window)

Target
------
  goals_scored : actual goals scored in the match
"""

from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "elo_for",
    "elo_against",
    "elo_diff",
    "is_home",
    "rolling_scored_5",
    "rolling_conceded_5",
    "rolling_scored_10",
    "rolling_conceded_10",
]
TARGET_COL = "goals_scored"

# Clamp predicted λ to a sensible range — a model should never predict
# negative goals or an absurd number like 8.
LAMBDA_MIN = 0.1
LAMBDA_MAX = 5.0

MODEL_PATH = Path(__file__).resolve().parents[1] / "outputs" / "xg_model.json"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xg_model(features_path: Path) -> XGBRegressor:
    """
    Load the feature matrix, train an XGBoost xG model, evaluate it,
    and persist it to disk.

    Parameters
    ----------
    features_path : Path
        Path to outputs/features.csv produced in Phase 1.

    Returns
    -------
    XGBRegressor
        The fitted model instance (also saved to outputs/xg_model.json).
    """
    df = pd.read_csv(features_path, parse_dates=["date"])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ------------------------------------------------------------------ #
    # Train / test split                                                   #
    # We split temporally — train on older matches, test on recent ones.  #
    # Random splits would leak future form into past predictions.         #
    # ------------------------------------------------------------------ #
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[model] Training samples : {len(X_train):,}")
    print(f"[model] Test samples     : {len(X_test):,}")

    # ------------------------------------------------------------------ #
    # XGBoost hyperparameters                                              #
    # These are sensible defaults for low-count regression.               #
    # n_estimators   : number of boosting rounds                          #
    # max_depth       : tree depth — keep shallow to avoid overfitting     #
    # learning_rate   : shrinkage per round                                #
    # subsample       : row sampling per tree (reduces variance)           #
    # colsample_bytree: feature sampling per tree (reduces variance)       #
    # ------------------------------------------------------------------ #
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,       # L1 regularisation — sparsity
        reg_lambda=1.0,      # L2 regularisation — weight decay
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"[model] Test MAE  : {mae:.4f} goals")
    print(f"[model] Test RMSE : {rmse:.4f} goals")
    print(f"[model] Baseline MAE (predict mean): "
          f"{mean_absolute_error(y_test, [y_train.mean()] * len(y_test)):.4f} goals")

    # ------------------------------------------------------------------ #
    # Feature importance (useful for debugging)                            #
    # ------------------------------------------------------------------ #
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    importance = dict(sorted(importance.items(),
                              key=lambda x: x[1], reverse=True))
    print("\n[model] Feature importances:")
    for feat, score in importance.items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<25} {score:.4f}  {bar}")

    # ------------------------------------------------------------------ #
    # Persist                                                              #
    # ------------------------------------------------------------------ #
    model.save_model(str(MODEL_PATH))
    print(f"\n[model] Model saved → {MODEL_PATH}")

    return model


def load_xg_model() -> XGBRegressor:
    """
    Load a previously trained model from disk.

    Returns
    -------
    XGBRegressor
        The loaded model, ready for inference.
    """
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_lambda(
    model: XGBRegressor,
    elo_for: float,
    elo_against: float,
    is_home: int,
    rolling_scored_5: float,
    rolling_conceded_5: float,
    rolling_scored_10: float,
    rolling_conceded_10: float,
) -> float:
    """
    Predict the Poisson λ (expected goals) for a single team in a fixture.

    Call this TWICE per match — once for each team — swapping the
    elo_for / elo_against and is_home values accordingly.

    Parameters
    ----------
    model            : fitted XGBRegressor
    elo_for          : Elo rating of the team we're predicting for
    elo_against      : Elo rating of the opponent
    is_home          : 1 if this team is the 'home' side, else 0
    rolling_scored_5 : team's average goals scored over last 5 matches
    rolling_conceded_5 : team's average goals conceded over last 5
    rolling_scored_10  : team's average goals scored over last 10
    rolling_conceded_10: team's average goals conceded over last 10

    Returns
    -------
    float
        Clamped λ value in [LAMBDA_MIN, LAMBDA_MAX].
    """
    features = pd.DataFrame([{
        "elo_for":             elo_for,
        "elo_against":         elo_against,
        "elo_diff":            elo_for - elo_against,
        "is_home":             is_home,
        "rolling_scored_5":    rolling_scored_5,
        "rolling_conceded_5":  rolling_conceded_5,
        "rolling_scored_10":   rolling_scored_10,
        "rolling_conceded_10": rolling_conceded_10,
    }])

    raw_lambda = float(model.predict(features)[0])
    return float(np.clip(raw_lambda, LAMBDA_MIN, LAMBDA_MAX))
