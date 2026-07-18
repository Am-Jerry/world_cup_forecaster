"""
poisson.py
----------
Implements the Poisson scoreline probability matrix and derives
match outcome probabilities from it.

Mathematical Background
-----------------------
If we model goals as independent Poisson processes:

    Home goals ~ Poisson(λ_home)
    Away goals ~ Poisson(λ_away)

Then the probability of a SPECIFIC scoreline (h, a) is:

    P(H=h, A=a) = P(H=h) * P(A=a)
                = [(λ_home^h * e^−λ_home) / h!]
                * [(λ_away^a * e^−λ_away) / a!]

This gives us a matrix where:
    rows    = home goals (0, 1, 2, ... MAX_GOALS)
    columns = away goals (0, 1, 2, ... MAX_GOALS)
    cell    = exact probability of that scoreline

From this matrix we can derive:
    P(home win) = sum of cells where row > col
    P(draw)     = sum of cells on the diagonal
    P(away win) = sum of cells where col > row

Independence assumption
-----------------------
We assume home and away goals are independent. This is a known
simplification — in reality, teams adjust tactics based on the
score. However, it is standard in football modelling (Dixon-Coles
1997 introduced a correlation correction, which we intentionally
omit here for clarity; it has minimal impact on match-winner
probabilities).

MAX_GOALS
---------
We truncate the distribution at 10 goals per team. The probability
mass beyond this point is negligible (< 0.001% for typical λ values)
but we normalise the matrix to ensure probabilities sum to 1.0.
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_GOALS = 10  # Maximum goals per team considered in the matrix


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    """
    Build the (MAX_GOALS+1) x (MAX_GOALS+1) scoreline probability matrix.

    Parameters
    ----------
    lambda_home : float
        Expected goals for the home team (predicted by xG model).
    lambda_away : float
        Expected goals for the away team (predicted by xG model).

    Returns
    -------
    np.ndarray  shape (MAX_GOALS+1, MAX_GOALS+1)
        matrix[h][a] = P(home scores h, away scores a)
        Rows  = home goals (0 … MAX_GOALS)
        Cols  = away goals (0 … MAX_GOALS)
        Matrix is normalised to sum to 1.0.
    """
    # PMF vector for each team: P(goals = k) for k in 0..MAX_GOALS
    home_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_away)

    # Outer product gives the joint probability matrix
    # (valid because we assume independence)
    matrix = np.outer(home_probs, away_probs)

    # Normalise to correct for truncation at MAX_GOALS
    matrix /= matrix.sum()

    return matrix


def match_outcome_probs(
    matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Derive Win / Draw / Loss probabilities from the score matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Output of score_matrix().

    Returns
    -------
    Tuple (p_home_win, p_draw, p_away_win)
        Three floats that sum to ~1.0.
    """
    # Upper triangle (row > col) → home win
    p_home_win = float(np.sum(np.tril(matrix, k=-1)))

    # Main diagonal (row == col) → draw
    p_draw = float(np.sum(np.diag(matrix)))

    # Lower triangle (col > row) → away win
    p_away_win = float(np.sum(np.triu(matrix, k=1)))

    return p_home_win, p_draw, p_away_win


def sample_score(matrix: np.ndarray) -> Tuple[int, int]:
    """
    Randomly sample a single scoreline from the probability matrix.

    This is the function the Monte Carlo simulator calls on every
    match in Phase 3.  We flatten the 2D matrix into a 1D probability
    vector, draw one index, then convert back to (home_goals, away_goals).

    Parameters
    ----------
    matrix : np.ndarray
        Output of score_matrix().

    Returns
    -------
    Tuple (home_goals, away_goals)
    """
    flat_probs = matrix.flatten()

    # np.random.choice over indices 0..(n-1)
    idx = np.random.choice(len(flat_probs), p=flat_probs)

    home_goals = idx // (MAX_GOALS + 1)
    away_goals = idx  % (MAX_GOALS + 1)

    return int(home_goals), int(away_goals)


def simulate_penalty_shootout() -> Tuple[int, int]:
    """
    Simulate a penalty shootout result.

    We don't model individual penalties — we simply apply empirical
    win probabilities from historical World Cup shootouts:
        Home / higher-Elo team wins ≈ 50% (shootouts are near-random).

    Returns a result code rather than a score, since the scoreline
    is officially recorded as the 90+30 min score in tournament records.

    Returns
    -------
    Tuple (home_pens, away_pens)
        Symbolic: (1, 0) = home wins shootout, (0, 1) = away wins.
        The simulator interprets these as advancement tokens, not goals.
    """
    if np.random.random() < 0.5:
        return (1, 0)   # home wins
    else:
        return (0, 1)   # away wins
