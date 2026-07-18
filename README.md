# 🏆 WC26 Forecaster

**A live, self-updating Monte Carlo forecasting engine for the 2026 FIFA World Cup — built to track the real tournament as it happens, not just predict it in advance.**

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)
![Status](https://img.shields.io/badge/status-live-brightgreen.svg)

---

## What is this?

WC26 Forecaster started as a straightforward question: *given a team's Elo rating, recent form, and FIFA's group/knockout rules, what's the actual mathematical probability they win the World Cup?*

Then the real 2026 tournament kicked off mid-project — and the engine had to grow up. Instead of staying a purely hypothetical, pre-tournament prediction tool, it now does something more interesting: it **locks in every real result as it happens** — every group match, every knockout round — and only spends its 10,000 Monte Carlo simulations on whatever is genuinely still undecided. Right now, that's exactly one match: the Final.

The dashboard isn't a static snapshot. It's built to be re-run as the tournament progresses, narrowing its own uncertainty round by round until nothing is left to simulate at all.

**[🔗 Live app](#)** *(https://wc26-forecast.streamlit.app/)*

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Live Forecast Architecture](#the-live-forecast-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Methodology Notes](#methodology-notes)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## How It Works

The engine is built in four phases, each one feeding the next:

### Phase 1 — Data & Elo Ratings
A historical dataset of international football results (2000–present, sourced from Kaggle) is merged with every real 2026 World Cup match played so far. A custom Elo rating system tracks team strength over time, using:
- **Tournament-weighted K-factors** — a World Cup match moves a team's rating far more than a friendly
- **Home-advantage adjustment** (+100 Elo for the home side, skipped for neutral-venue matches)
- **A margin-of-victory multiplier** (FiveThirtyEight-style) so a 4–0 win shifts ratings more than a 1–0 win, with a correction so it doesn't overweight already-dominant teams

Rolling 5- and 10-match scoring/conceding averages are also computed per team as additional model features.

### Phase 2 — Expected Goals & Scoreline Probabilities
An **XGBoost regressor** predicts each team's expected goals (λ) for a given matchup, based on Elo, Elo differential, home advantage, and recent scoring form. Those λ values feed a **Poisson distribution**, which produces a full probability matrix over every realistic scoreline (0–0 through 10–10), from which win/draw/loss probabilities are derived directly.

### Phase 3 — Monte Carlo Simulation *(now live-state-aware)*
The original design simulated the entire 48-team, 12-group tournament from scratch, 10,000 times, applying FIFA's official tiebreaker rules (head-to-head points → goal difference → goals scored → overall goal difference → overall goals → lots) and the real Round-of-32 qualification structure (top 2 per group + best 8 third-place teams).

That's still in the codebase — but it's not what actually runs anymore. See [below](#the-live-forecast-architecture) for what replaced it.

### Phase 4 — The Dashboard
A multi-page Streamlit app, styled as a dark, mobile-app-inspired dashboard with a floating pill navigation bar:

| Page | What it shows |
|---|---|
| **Home** | The Final matchup and both finalists' championship odds |
| **All Teams** | Every one of the 48 real teams, sortable by probability of reaching each stage |
| **Groups** | Real final group-stage standings (W/D/L/Pts), computed from actual results — not simulated |
| **Path to the Final** | The complete real bracket, Round of 32 through the Final, with interactive round tabs |
| **About** | This methodology, in-app |

---

## The Live Forecast Architecture

This is the part that makes the project more than a template Monte Carlo simulator.

As of the current build, the tournament has been played through both semifinals. Rather than re-simulating matches that have already happened, the engine:

1. **Reads the real results** from `data/results_2026.csv` — every match tagged by stage (Group / R32 / R16 / QF / SF / Final)
2. **Derives the current tournament state** — which teams are already eliminated (and at which stage), which knockout matches are decided, and which are still pending
3. **Locks in every decided match** exactly as it really happened — no randomness applied
4. **Simulates only what's genuinely uncertain** — right now, just the Final
5. Outputs probabilities that are **100% or 0% for anything already determined**, and genuinely probabilistic only for what hasn't been played yet

As the tournament concludes, updating the forecast is a three-step loop:
1. Append the new real result to `data/results_2026.csv`
2. Re-run `main.py` (recomputes Elo with the new result, re-derives tournament state, re-simulates whatever's left)
3. Commit the refreshed `outputs/simulation_results.csv` — the dashboard picks it up automatically

---

## Tech Stack

- **Python 3.12**
- **pandas / numpy** — data pipeline and Elo computation
- **XGBoost** — expected-goals regression
- **SciPy** — Poisson distribution math
- **Streamlit** — dashboard framework, deployed on Streamlit Community Cloud
- **tqdm** — simulation progress tracking

---

## Project Structure

```
world_cup_forecaster/
├── Home.py                    # Streamlit entry point
├── main.py                    # Full pipeline: Phase 1 → 2 → 3
├── requirements.txt
├── data/
│   ├── results.csv            # Historical international results (2000–present)
│   └── results_2026.csv       # Real 2026 World Cup results, tagged by stage
├── outputs/                   # Generated by main.py — must be committed for deployment
│   ├── final_elo_ratings.csv
│   ├── features.csv
│   ├── xg_model.json
│   └── simulation_results.csv
├── pages/                     # Streamlit multi-page app
│   ├── 1_All_Teams.py
│   ├── 2_Groups.py
│   ├── 3_Path_to_Final.py
│   └── 4_About.py
└── src/
    ├── data_loader.py         # Loads + merges + normalises team names
    ├── elo.py                 # Elo rating engine
    ├── features.py            # Rolling-form feature engineering
    ├── model.py                # XGBoost expected-goals model
    ├── poisson.py              # Scoreline probability matrix
    ├── tournament.py           # Real 2026 groups, bracket, team constants
    ├── simulator.py             # Live Monte Carlo forecast engine
    ├── bracket_data.py          # Derives real standings + bracket for the dashboard
    └── theme.py                # Shared dashboard styling, nav, and data loading
```

---

## Getting Started

**1. Clone the repo and install dependencies**

```bash
git clone https://github.com/Am-Jerry/world_cup_forecaster.git
cd world_cup_forecaster
pip install -r requirements.txt
```

**2. Get the historical dataset**

Download *"International football results from 1872 to 2024"* from Kaggle and save it as `data/results.csv`.

**3. Check team name consistency**

Different datasets spell team names differently (e.g. "Curaçao" vs "Curacao"). Run:

```bash
python check_team_names.py
```

This flags any of the 48 real 2026 teams that don't exactly match your `results.csv`, so Elo ratings carry over correctly instead of silently starting fresh at 1500 for a mismatched name.

**4. Run the pipeline**

```bash
python main.py
```

This runs all three phases end to end: builds the Elo/feature dataset, trains the expected-goals model, derives the current real tournament state, and simulates the remaining uncertainty 10,000 times. Outputs land in `outputs/`.

---

## Methodology Notes

Some deliberate simplifications, documented rather than hidden:

- **Poisson independence is assumed** — home and away goals are treated as statistically independent. In reality they're weakly correlated (Dixon-Coles 1997 proposed a low-score correction); the effect on match-winner probabilities is small enough to omit here.
- **Extra time** is modeled as a 35% reduction in both teams' λ, reflecting fatigue and more conservative play.
- **Penalty shootouts** are treated as a 50/50 coin flip — empirically, shootout outcomes are close to random regardless of team quality.
- **Group tiebreakers** implement FIFA's actual rule hierarchy (head-to-head points → head-to-head goal difference → head-to-head goals scored → overall goal difference → overall goals scored → drawing of lots), simulated as random for the final tiebreaker.

---

## Known Limitations

- **Team name matching is fragile.** Elo continuity depends on exact string matches between `results.csv` and `results_2026.csv`. `check_team_names.py` exists specifically because this fails silently otherwise.
- **Elo/xG-based forecasting doesn't know about injuries, suspensions, tactical news, or anything outside historical scoring patterns and results.**
- **Not affiliated with FIFA.** Team names, group draws, and match data are used for statistical/educational purposes only.

---

## License

Licensed under the **GNU General Public License v3.0**. This is free software — you're welcome to redistribute and/or modify it under the terms of that license. See [`LICENSE`](./LICENSE) for the full text.

© 2026, Jerry, Inc.
