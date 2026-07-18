"""
tournament.py
-------------
Defines the complete 2026 FIFA World Cup structure.

Format Facts (Official FIFA 2026)
----------------------------------
- 48 teams total
- 12 groups of 4 teams each
- Top 2 from each group advance automatically (24 teams)
- Best 8 third-place finishers also advance (8 teams)
- Total 32 teams enter the Round of 32 (knockout)
- Knockout rounds: R32 → R16 → QF → SF → Final
- All knockout draws decided by extra time + penalties if level at 90

Group Stage Tiebreaker Rules (FIFA Official)
--------------------------------------------
When teams are level on points, FIFA breaks ties in this strict order:
  1. Points in head-to-head matches among tied teams
  2. Goal difference in head-to-head matches
  3. Goals scored in head-to-head matches
  4. Goal difference across all group matches
  5. Goals scored across all group matches
  6. Drawing of lots (we simulate this as random)

We implement rules 1-5 plus random for rule 6.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Team:
    """
    Represents a single national team with all features needed
    for match simulation and tournament tracking.
    """
    name: str
    elo: float
    rolling_scored_5:    float = 1.2
    rolling_conceded_5:  float = 1.2
    rolling_scored_10:   float = 1.2
    rolling_conceded_10: float = 1.2

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class GroupStanding:
    """
    Tracks a single team's record within its group.
    Used to apply FIFA tiebreaker rules correctly.
    """
    team: Team
    played:       int = 0
    wins:         int = 0
    draws:        int = 0
    losses:       int = 0
    goals_for:    int = 0
    goals_against:int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def update(self, scored: int, conceded: int) -> None:
        """Update standing after a match result."""
        self.played += 1
        self.goals_for     += scored
        self.goals_against += conceded
        if scored > conceded:
            self.wins   += 1
        elif scored == conceded:
            self.draws  += 1
        else:
            self.losses += 1


# ---------------------------------------------------------------------------
# 2026 World Cup Groups — REAL, CONFIRMED DRAW
# ---------------------------------------------------------------------------
# This is the actual, official group draw the 2026 tournament was played
# under (all 72 group-stage matches are already complete and recorded in
# data/results_2026.csv). Team name spellings match results_2026.csv and
# should also match your historical data/results.csv — see the note in
# data_loader.py about verifying this.

GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Australia", "Turkey", "Paraguay"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "DR Congo", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


# ---------------------------------------------------------------------------
# 2026 Knockout Stage — REAL, CONFIRMED RESULTS STATE (as of this build)
# ---------------------------------------------------------------------------
# The group stage, Round of 32, and Round of 16 are ALL complete in reality
# (see data/results_2026.csv, stage column). Only two of the four
# quarterfinals have been played so far.
#
# QUARTERFINALISTS_2026 is the ground-truth set of the 8 real quarterfinal
# teams. This can't be fully re-derived from results_2026.csv alone, because
# a couple of knockout matches were decided on penalties (recorded as a draw
# in the score columns) — so the confirmed teams are recorded explicitly
# here rather than inferred.
#
# QF_BRACKET_2026 fixes the actual pairing for all four quarterfinals, in
# bracket order. The pairing order matters: simulator.py derives the real
# semifinal matchups from it as winner(QF[0]) vs winner(QF[1]), and
# winner(QF[2]) vs winner(QF[3]) — which is exactly how the real bracket
# resolves (France/Spain into one semifinal, the other two into the other).

QUARTERFINALISTS_2026: List[str] = [
    "France", "Morocco", "Spain", "Belgium",
    "Norway", "England", "Argentina", "Switzerland",
]

QF_BRACKET_2026: List[tuple] = [
    ("France", "Morocco"),        # QF1 — already decided: France won 2-0
    ("Spain", "Belgium"),         # QF2 — already decided: Spain won 2-1
    ("Norway", "England"),        # QF3 — not yet played
    ("Argentina", "Switzerland"), # QF4 — not yet played
]


# ---------------------------------------------------------------------------
# Knockout bracket seeding (legacy — full from-scratch simulation)
# ---------------------------------------------------------------------------
# The functions below build a Round of 32 bracket from a SIMULATED group
# stage. They're kept for reference / for reuse in a future tournament, but
# are no longer part of the live 2026 pipeline now that the real group
# stage and Round of 32 are already complete — see simulator.py's
# derive_tournament_state() and simulate_tournament_2026() instead.

def build_r32_bracket(
    group_winners:   Dict[str, "Team"],
    group_runnersup: Dict[str, "Team"],
    best_thirds:     List["Team"],
) -> List[tuple]:
    """
    Construct the Round of 32 bracket from group stage results.

    FIFA 2026 bracket pairing (simplified sequential draw):
    Winners of groups face runners-up from other groups, with
    third-place teams slotted in according to FIFA allocation.

    Parameters
    ----------
    group_winners   : dict mapping group letter → Team
    group_runnersup : dict mapping group letter → Team
    best_thirds     : ordered list of 8 best third-place Teams

    Returns
    -------
    List of 16 (team_a, team_b) tuples representing R32 matchups.
    """
    groups = sorted(group_winners.keys())  # ['A','B',...,'L']

    matchups = []

    # 12 winner vs runner-up matches (cross-group)
    group_pairs = [
        (groups[i], groups[i + 1])
        for i in range(0, len(groups), 2)
    ]

    for g1, g2 in group_pairs:
        matchups.append((group_winners[g1],   group_runnersup[g2]))
        matchups.append((group_winners[g2],   group_runnersup[g1]))

    # 8 third-place team matches vs winners from remaining groups
    for i, third_team in enumerate(best_thirds):
        opponent_group = groups[(i + 6) % len(groups)]
        matchups.append((third_team, group_winners[opponent_group]))

    return matchups
