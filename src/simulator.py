"""
simulator.py
------------
The Monte Carlo simulation engine for the 2026 FIFA World Cup.

Two simulation modes live in this file:

1. LEGACY / FROM-SCRATCH — simulate_tournament() + run_monte_carlo()
   Simulates the entire tournament from the group stage onward. This was
   the original Phase 3 design, written before the real tournament had
   been played. It's kept for reference and for reuse on a future
   tournament, but is NOT what main.py calls for the 2026 forecast anymore.

2. LIVE / REAL-STATE-AWARE — derive_tournament_state() +
   simulate_tournament_2026() + run_monte_carlo_2026()
   Reads the real results in data/results_2026.csv, locks in every
   completed match (group stage, R32, R16, and the two quarterfinals
   already played), and Monte Carlo simulates ONLY what's genuinely still
   undecided: the two remaining quarterfinals, both semifinals, and the
   final. This is what main.py actually runs.

Architecture (live mode)
-------------------------
run_monte_carlo_2026()
    └── derive_tournament_state()      ← reads real results, once
    └── pre-seeds guaranteed stage counts (100% for already-real outcomes)
    └── loop n_simulations times:
            simulate_tournament_2026() ← only QF(undecided)/SF/Final
                └── simulate_knockout_match()   ← Poisson score sampler
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from src.tournament import (
    Team, GroupStanding, GROUPS, build_r32_bracket,
    QUARTERFINALISTS_2026, QF_BRACKET_2026,
)
from src.model import load_xg_model, predict_lambda
from src.poisson import score_matrix, sample_score, simulate_penalty_shootout


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ELO_PATH      = Path("outputs/final_elo_ratings.csv")
FEATURES_PATH = Path("outputs/features.csv")
RESULTS_2026_PATH = Path("data/results_2026.csv")
DEFAULT_ELO   = 1500.0
N_SIMULATIONS = 10_000

# Stages tracked for output probability tables
STAGES = ["R32", "R16", "QF", "SF", "Final", "Winner"]

# Ordinal rank for each stage — used both by the legacy and live paths to
# credit a team for every stage at-or-below the furthest one they reached.
STAGE_ORDER = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "Final": 4, "Winner": 5}

# Ordinal rank for the three REAL, already-decided elimination points that
# can show up in results_2026.csv before the quarterfinal stage.
GUARANTEED_STAGE_RANK = {"Group": -1, "R32": 0, "R16": 1, "QF": 2}


# ---------------------------------------------------------------------------
# Team loader (shared by both simulation modes)
# ---------------------------------------------------------------------------

def load_teams(
    elo_path: Path = ELO_PATH,
    features_path: Path = FEATURES_PATH,
) -> Dict[str, Team]:
    """
    Build a Team object for every team in the tournament by merging:
      - Final Elo ratings (from Phase 1 output — now includes 2026 form,
        since data_loader.py merges data/results_2026.csv into training data)
      - Latest rolling stats (from Phase 1 feature matrix)

    Parameters
    ----------
    elo_path      : Path to outputs/final_elo_ratings.csv
    features_path : Path to outputs/features.csv

    Returns
    -------
    Dict mapping team name → Team instance.
    """
    elo_df = pd.read_csv(elo_path)
    elo_map = dict(zip(elo_df["team"], elo_df["elo"]))

    feat_df = pd.read_csv(features_path)
    latest  = (
        feat_df.sort_values("date")
               .groupby("team")
               .last()
               .reset_index()
    )
    stats_map = {}
    for _, row in latest.iterrows():
        stats_map[row["team"]] = {
            "rolling_scored_5":    row["rolling_scored_5"],
            "rolling_conceded_5":  row["rolling_conceded_5"],
            "rolling_scored_10":   row["rolling_scored_10"],
            "rolling_conceded_10": row["rolling_conceded_10"],
        }

    all_team_names = {t for group in GROUPS.values() for t in group}
    teams = {}

    for name in all_team_names:
        elo   = elo_map.get(name, DEFAULT_ELO)
        stats = stats_map.get(name, {
            "rolling_scored_5":    1.2,
            "rolling_conceded_5":  1.2,
            "rolling_scored_10":   1.2,
            "rolling_conceded_10": 1.2,
        })
        teams[name] = Team(name=name, elo=elo, **stats)

    print(f"[simulator] Loaded {len(teams)} teams.")
    return teams


# ---------------------------------------------------------------------------
# Match simulation (shared by both simulation modes)
# ---------------------------------------------------------------------------

def simulate_match(
    team_a: Team,
    team_b: Team,
    model,
    neutral: bool = True,
) -> Tuple[int, int]:
    """Simulate a single match and return the scoreline."""
    is_home_a = 0 if neutral else 1
    is_home_b = 0

    lambda_a = predict_lambda(
        model,
        elo_for=team_a.elo, elo_against=team_b.elo, is_home=is_home_a,
        rolling_scored_5=team_a.rolling_scored_5,
        rolling_conceded_5=team_a.rolling_conceded_5,
        rolling_scored_10=team_a.rolling_scored_10,
        rolling_conceded_10=team_a.rolling_conceded_10,
    )
    lambda_b = predict_lambda(
        model,
        elo_for=team_b.elo, elo_against=team_a.elo, is_home=is_home_b,
        rolling_scored_5=team_b.rolling_scored_5,
        rolling_conceded_5=team_b.rolling_conceded_5,
        rolling_scored_10=team_b.rolling_scored_10,
        rolling_conceded_10=team_b.rolling_conceded_10,
    )

    matrix = score_matrix(lambda_a, lambda_b)
    return sample_score(matrix)


def simulate_knockout_match(team_a: Team, team_b: Team, model) -> Team:
    """
    Simulate a knockout match. If drawn after 90 minutes, simulate
    extra time (reduced λ), then penalties if still level.
    """
    goals_a, goals_b = simulate_match(team_a, team_b, model, neutral=True)
    if goals_a != goals_b:
        return team_a if goals_a > goals_b else team_b

    lambda_a = predict_lambda(
        model, elo_for=team_a.elo, elo_against=team_b.elo, is_home=0,
        rolling_scored_5=team_a.rolling_scored_5,
        rolling_conceded_5=team_a.rolling_conceded_5,
        rolling_scored_10=team_a.rolling_scored_10,
        rolling_conceded_10=team_a.rolling_conceded_10,
    ) * 0.65
    lambda_b = predict_lambda(
        model, elo_for=team_b.elo, elo_against=team_a.elo, is_home=0,
        rolling_scored_5=team_b.rolling_scored_5,
        rolling_conceded_5=team_b.rolling_conceded_5,
        rolling_scored_10=team_b.rolling_scored_10,
        rolling_conceded_10=team_b.rolling_conceded_10,
    ) * 0.65

    et_matrix  = score_matrix(lambda_a, lambda_b)
    et_a, et_b = sample_score(et_matrix)
    if et_a != et_b:
        return team_a if et_a > et_b else team_b

    pen_a, pen_b = simulate_penalty_shootout()
    return team_a if pen_a > pen_b else team_b


# =============================================================================
# LIVE / REAL-STATE-AWARE PATH — this is what main.py actually uses
# =============================================================================

def derive_tournament_state(results_path: Path = RESULTS_2026_PATH) -> dict:
    """
    Read data/results_2026.csv and work out, for every real team, the
    furthest stage they've genuinely reached so far — plus which knockout
    matches (quarterfinal and semifinal) are already decided vs. still to
    be played.

    Why not just compare scores everywhere? A few knockout matches were
    decided on penalties, which are recorded as a drawn scoreline (e.g.
    Switzerland beat Colombia 0-0 on penalties in the Round of 16). Score
    comparison alone can't tell you who advanced in that case. Instead we
    determine advancement by checking which stage's match list each team
    actually appears in — a team that played an R16 match but never
    appears in a QF-stage row didn't advance, regardless of how their R16
    match finished. The one gap this can't close on its own is telling
    apart the two still-undecided quarterfinals from the two that are
    already played, which is why QUARTERFINALISTS_2026 is recorded
    explicitly in tournament.py rather than inferred.

    Semifinal pairings, unlike the quarterfinals, aren't fixed in advance
    in tournament.py — they're derived here from the actual quarterfinal
    winners (SF1 = winner(QF1) vs winner(QF2), SF2 = winner(QF3) vs
    winner(QF4)), then checked against results_2026.csv the same way.

    Returns
    -------
    dict with:
      'eliminated'  : {team_name: furthest_real_stage}  — for the 40 teams
                       out before the quarterfinal stage. Stage is one of
                       'Group', 'R32', 'R16'.
      'qf_decided'  : {(team_a, team_b): winner_name} — completed QFs.
      'qf_pending'  : [(team_a, team_b), ...] — QFs still to simulate.
      'sf_pairs'    : [(team_a, team_b), (team_a, team_b)] — the two real
                       semifinal pairings, derived from qf_decided/pending
                       winners (a pairing may reference a team name that
                       isn't known yet if its QF is still pending).
      'sf_decided'  : {(team_a, team_b): winner_name} — completed SFs.
      'sf_pending'  : [(team_a, team_b), ...] — SFs still to simulate.
    """
    df = pd.read_csv(results_path)

    def teams_in_stage(stage: str) -> set:
        rows = df[df["stage"] == stage]
        return set(rows["home_team"]) | set(rows["away_team"])

    group_teams = teams_in_stage("Group")
    r32_teams   = teams_in_stage("R32")
    r16_teams   = teams_in_stage("R16")

    eliminated: Dict[str, str] = {}
    for t in group_teams:
        if t not in r32_teams:
            eliminated[t] = "Group"
    for t in r32_teams:
        if t not in r16_teams:
            eliminated[t] = "R32"
    for t in r16_teams:
        if t not in QUARTERFINALISTS_2026:
            eliminated[t] = "R16"

    def resolve_matches(pairs, stage_label):
        """Given a list of (team_a, team_b) pairs, check results_2026.csv
        for a completed match at the given stage and split into decided
        (with winner) vs pending."""
        rows = df[df["stage"] == stage_label]
        decided, pending = {}, []
        for a, b in pairs:
            match = rows[
                ((rows["home_team"] == a) & (rows["away_team"] == b)) |
                ((rows["home_team"] == b) & (rows["away_team"] == a))
            ]
            if match.empty:
                pending.append((a, b))
            else:
                row = match.iloc[0]
                winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
                decided[(a, b)] = winner
        return decided, pending

    qf_decided, qf_pending = resolve_matches(QF_BRACKET_2026, "QF")

    # Derive the two real semifinal pairings from quarterfinal winners.
    # A pairing may include a team we don't know yet (QF still pending) —
    # resolve_matches simply won't find a match for that pair, correctly
    # leaving it in sf_pending.
    def qf_winner_or_none(a, b):
        return qf_decided.get((a, b))

    sf_pairs = [
        (qf_winner_or_none(*QF_BRACKET_2026[0]), qf_winner_or_none(*QF_BRACKET_2026[1])),
        (qf_winner_or_none(*QF_BRACKET_2026[2]), qf_winner_or_none(*QF_BRACKET_2026[3])),
    ]
    # Only look up real SF results for pairings where both teams are known.
    resolvable_sf_pairs = [(a, b) for a, b in sf_pairs if a is not None and b is not None]
    sf_decided, sf_pending = resolve_matches(resolvable_sf_pairs, "SF")

    print(f"[simulator] Real state: {len(eliminated)} teams already eliminated "
          f"before QF, {len(qf_decided)} QFs decided, {len(qf_pending)} QFs pending, "
          f"{len(sf_decided)} SFs decided, {len(sf_pending) + (2 - len(resolvable_sf_pairs))} SFs pending.")

    return {
        "eliminated": eliminated,
        "qf_decided": qf_decided,
        "qf_pending": qf_pending,
        "sf_pairs": sf_pairs,
        "sf_decided": sf_decided,
        "sf_pending": sf_pending,
    }


def simulate_tournament_2026(teams_map: Dict[str, Team], model, state: dict) -> Dict[str, str]:
    """
    Simulate ONE possible completion of the real 2026 tournament from its
    current position. Every match with a real result — quarterfinal or
    semifinal — is used as-is; only genuinely undecided matches are
    Monte Carlo simulated.

    Returns
    -------
    Dict mapping team_name → furthest stage reached IN THIS SIMULATION RUN,
    for teams that reach at least the semifinal. (Teams eliminated at Group/
    R32/R16/QF already have those stages credited deterministically outside
    this function, in run_monte_carlo_2026 — see the docstring there.)
    """
    results: Dict[str, str] = {}

    # Resolve all four quarterfinals — real result where we have one,
    # simulated where we don't — in fixed bracket order.
    qf_winners: List[Team] = []
    for a, b in QF_BRACKET_2026:
        if (a, b) in state["qf_decided"]:
            winner = teams_map[state["qf_decided"][(a, b)]]
        else:
            winner = simulate_knockout_match(teams_map[a], teams_map[b], model)
        results[winner.name] = "SF"  # winning the QF means reaching the SF
        qf_winners.append(winner)

    # Semifinals: SF1 = QF1 winner vs QF2 winner, SF2 = QF3 winner vs QF4
    # winner — again, real result where we have one (e.g. France/Spain,
    # already played), simulated otherwise (e.g. England/Argentina).
    sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
    sf_winners: List[Team] = []
    for team_a, team_b in sf_pairs:
        fwd, rev = (team_a.name, team_b.name), (team_b.name, team_a.name)
        if fwd in state["sf_decided"]:
            winner = teams_map[state["sf_decided"][fwd]]
        elif rev in state["sf_decided"]:
            winner = teams_map[state["sf_decided"][rev]]
        else:
            winner = simulate_knockout_match(team_a, team_b, model)
        results[winner.name] = "Final"  # winning the SF means reaching the Final
        sf_winners.append(winner)

    # Final — always simulated; the real final hasn't been played yet.
    champion = simulate_knockout_match(sf_winners[0], sf_winners[1], model)
    results[champion.name] = "Winner"

    return results


def run_monte_carlo_2026(n_simulations: int = N_SIMULATIONS) -> pd.DataFrame:
    """
    Run the live, real-state-aware Monte Carlo simulation.

    Every team's probability table is built from two sources:
      1. GUARANTEED counts — stages already achieved in reality (Group
         exit, R32 exit, R16 exit, or reaching QF) are set to 100% directly,
         with zero simulation needed, since they already happened.
      2. SIMULATED counts — SF / Final / Winner probabilities for the 8
         real quarterfinalists, built by running simulate_tournament_2026()
         n_simulations times. France and Spain additionally get SF
         pre-seeded to 100%, since they already won their quarterfinal.

    Parameters
    ----------
    n_simulations : Number of tournament completions to simulate (default
                    10,000).

    Returns
    -------
    pd.DataFrame
        Columns: team, R32%, R16%, QF%, SF%, Final%, Winner%
        Sorted by Winner% descending. All 48 real teams included.
    """
    print(f"[simulator] Loading model and team data...")
    model     = load_xg_model()
    teams_map = load_teams()
    state     = derive_tournament_state()

    stage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # --- 1. Guaranteed counts for the 40 teams eliminated before QF ---
    for team_name, stage in state["eliminated"].items():
        reached_rank = GUARANTEED_STAGE_RANK[stage]
        for s, idx in STAGE_ORDER.items():
            if idx <= reached_rank:
                stage_counts[team_name][s] = n_simulations

    # --- 2. Guaranteed R32/R16/QF for all 8 real quarterfinalists ---
    # This is always true regardless of whether a given team's QF or SF
    # match has actually been played yet, so it's set once here rather
    # than left for the simulation loop to (re-)derive.
    for team_name in QUARTERFINALISTS_2026:
        for s in ("R32", "R16", "QF"):
            stage_counts[team_name][s] = n_simulations

    # --- 3. Simulate the genuinely undecided remainder ---
    # IMPORTANT: only SF / Final / Winner are accumulated from the loop
    # below. R32/R16/QF are already guaranteed in step 2 above — even for
    # a real, already-decided match, simulate_tournament_2026() still
    # reports that team's result on every iteration (deterministically,
    # since there's nothing to randomise), so letting the cascade below
    # also touch R32/R16/QF would double-count them on top of step 2 and
    # push those percentages to 200%. Restricting the cascade to
    # SF/Final/Winner keeps each stage counted exactly once.
    LIVE_STAGES = ("SF", "Final", "Winner")
    print(f"[simulator] Simulating remaining bracket {n_simulations:,} times...\n")
    for _ in tqdm(range(n_simulations), desc="Simulating", unit="sim"):
        sim_result = simulate_tournament_2026(teams_map, model, state)
        for team_name, stage in sim_result.items():
            reached_idx = STAGE_ORDER[stage]
            for s in LIVE_STAGES:
                if STAGE_ORDER[s] <= reached_idx:
                    stage_counts[team_name][s] += 1

    # --- Build results dataframe for all 48 real teams ---
    rows = []
    all_teams = {t for group in GROUPS.values() for t in group}
    for team_name in all_teams:
        counts = stage_counts.get(team_name, {})
        row = {"team": team_name}
        for stage in STAGES:
            pct = (counts.get(stage, 0) / n_simulations) * 100
            row[f"{stage}%"] = round(pct, 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Winner%", ascending=False)
    df.reset_index(drop=True, inplace=True)

    out_path = Path("outputs/simulation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[simulator] Results saved → {out_path}")

    return df


# =============================================================================
# LEGACY / FROM-SCRATCH PATH — kept for reference, not used by main.py
# =============================================================================
# Everything below simulates the full tournament (group stage included)
# from nothing. It's what Phase 3 originally shipped with, before the real
# tournament had reached the quarterfinals. Reuse it for a future World Cup,
# or for hypothetical "what if the draw were different" scenarios — but for
# the live 2026 forecast, run_monte_carlo_2026() above is what's current.

def rank_group(
    standings: Dict[str, GroupStanding],
    results:   Dict[Tuple[str, str], Tuple[int, int]],
) -> List[GroupStanding]:
    """Rank all teams in a group applying FIFA 2026 tiebreaker rules."""
    standing_list = list(standings.values())

    def h2h_record(team: str, rivals: List[str]) -> Tuple[int, int, int]:
        pts, gd, gf = 0, 0, 0
        for rival in rivals:
            if (team, rival) in results:
                g_for, g_ag = results[(team, rival)]
            elif (rival, team) in results:
                g_ag, g_for = results[(rival, team)]
            else:
                continue
            gf += g_for
            gd += g_for - g_ag
            if g_for > g_ag:
                pts += 3
            elif g_for == g_ag:
                pts += 1
        return pts, gd, gf

    def sort_key(s: GroupStanding, rivals: List[str]) -> tuple:
        h2h_pts, h2h_gd, h2h_gf = h2h_record(s.team.name, rivals)
        return (
            s.points, h2h_pts, h2h_gd, h2h_gf,
            s.goal_difference, s.goals_for, np.random.random(),
        )

    all_names = [s.team.name for s in standing_list]
    standing_list.sort(
        key=lambda s: sort_key(s, [n for n in all_names if n != s.team.name]),
        reverse=True,
    )
    return standing_list


def simulate_group(group_name: str, teams: List[Team], model) -> Tuple[List[GroupStanding], Dict]:
    """Simulate all 6 matches in a group of 4 and return ranked standings."""
    standings = {t.name: GroupStanding(team=t) for t in teams}
    results: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a, team_b = teams[i], teams[j]
            goals_a, goals_b = simulate_match(team_a, team_b, model, neutral=True)
            standings[team_a.name].update(goals_a, goals_b)
            standings[team_b.name].update(goals_b, goals_a)
            results[(team_a.name, team_b.name)] = (goals_a, goals_b)

    ranked = rank_group(standings, results)
    return ranked, results


def select_best_thirds(third_place_standings: List[GroupStanding]) -> List[Team]:
    """Select the 8 best third-place finishers from 12 groups."""
    third_place_standings.sort(
        key=lambda s: (s.points, s.goal_difference, s.goals_for, np.random.random()),
        reverse=True,
    )
    return [s.team for s in third_place_standings[:8]]


def simulate_tournament(teams_map: Dict[str, Team], model) -> Dict[str, str]:
    """
    NOTE: known limitation — a losing team's stage entry gets overwritten
    by the "Eliminated (...)" label from the round they lose in, which
    means the pre-aggregation dict can under-report R32/R16/QF reach for
    knockout losers. Harmless for the current pipeline since this function
    isn't called by main.py, but worth fixing before reusing this path.
    """
    results: Dict[str, str] = {}

    group_winners:    Dict[str, Team] = {}
    group_runnersup:  Dict[str, Team] = {}
    third_place_list: List[GroupStanding] = []

    for group_letter, team_names in GROUPS.items():
        group_teams = [teams_map[n] for n in team_names]
        ranked, _ = simulate_group(group_letter, group_teams, model)
        group_winners[group_letter]   = ranked[0].team
        group_runnersup[group_letter] = ranked[1].team
        third_place_list.append(ranked[2])
        results[ranked[3].team.name] = "Eliminated (Group)"

    best_thirds = select_best_thirds(third_place_list)
    all_thirds = {s.team.name for s in third_place_list}
    qualified_thirds = {t.name for t in best_thirds}
    for name in all_thirds - qualified_thirds:
        results[name] = "Eliminated (Group)"

    r32_teams = list(group_winners.values()) + list(group_runnersup.values()) + best_thirds
    for team in r32_teams:
        results[team.name] = "R32"

    bracket = build_r32_bracket(group_winners, group_runnersup, best_thirds)

    def run_knockout_round(matchups, stage):
        winners = []
        for team_a, team_b in matchups:
            winner = simulate_knockout_match(team_a, team_b, model)
            loser  = team_b if winner == team_a else team_a
            results[loser.name]  = f"Eliminated ({stage})"
            results[winner.name] = stage
            winners.append(winner)
        return winners

    r16_teams   = run_knockout_round(bracket, "R16")
    r16_matchups = [(r16_teams[i], r16_teams[i+1]) for i in range(0, len(r16_teams), 2)]
    qf_teams    = run_knockout_round(r16_matchups, "QF")
    qf_matchups  = [(qf_teams[i], qf_teams[i+1]) for i in range(0, len(qf_teams), 2)]
    sf_teams    = run_knockout_round(qf_matchups, "SF")
    sf_matchups  = [(sf_teams[i], sf_teams[i+1]) for i in range(0, len(sf_teams), 2)]
    finalists   = run_knockout_round(sf_matchups, "Final")

    champion = simulate_knockout_match(finalists[0], finalists[1], model)
    results[champion.name] = "Winner"

    return results


def run_monte_carlo(n_simulations: int = N_SIMULATIONS) -> pd.DataFrame:
    """Legacy full-tournament-from-scratch Monte Carlo loop. See module docstring."""
    print(f"[simulator] Loading model and team data...")
    model     = load_xg_model()
    teams_map = load_teams()

    stage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print(f"[simulator] Running {n_simulations:,} simulations...\n")
    for _ in tqdm(range(n_simulations), desc="Simulating", unit="sim"):
        sim_result = simulate_tournament(teams_map, model)
        for team_name, stage in sim_result.items():
            if stage not in STAGE_ORDER:
                continue
            reached_idx = STAGE_ORDER[stage]
            for s, idx in STAGE_ORDER.items():
                if idx <= reached_idx:
                    stage_counts[team_name][s] += 1

    rows = []
    all_teams = {t for group in GROUPS.values() for t in group}
    for team_name in all_teams:
        counts = stage_counts.get(team_name, {})
        row = {"team": team_name}
        for stage in STAGES:
            pct = (counts.get(stage, 0) / n_simulations) * 100
            row[f"{stage}%"] = round(pct, 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Winner%", ascending=False)
    df.reset_index(drop=True, inplace=True)

    out_path = Path("outputs/simulation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[simulator] Results saved → {out_path}")

    return df
