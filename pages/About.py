"""
pages/4_About.py
---------------------
Methodology explanation and disclaimer. Static content, no data
dependency beyond the shared theme.
"""

import streamlit as st
from src.theme import inject_css, COLORS, render_bottom_nav, render_footer

st.set_page_config(page_title="About — WC26", page_icon="ℹ️", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_bottom_nav("about")

st.markdown("## ℹ️ About This Model")
st.markdown(
    '<p class="wc-muted">How the forecasting engine works, end to end.</p>',
    unsafe_allow_html=True,
)

items = [
    ("🛡️", "Elo Rating Engine",
     "A custom Elo system with tournament-weighted K-factors (World Cup matches move ratings "
     "furthest), home-advantage adjustment, and a margin-of-victory multiplier, computed from "
     "international results since 2000 plus every real 2026 World Cup match played so far."),
    ("🎯", "Poisson Expected-Goals Model",
     "An XGBoost regressor predicts each team's expected goals (λ) from their Elo rating and "
     "recent scoring form. A Poisson distribution converts those λ values into a full scoreline "
     "probability matrix for any matchup."),
    ("⚡", "Monte Carlo Simulation",
     "The tournament is simulated 10,000 times. Every match that's already been played — the "
     "full group stage, Round of 32, Round of 16, and quarterfinals — is locked in exactly as it "
     "really happened. Only what's genuinely still undecided gets simulated."),
    ("🏆", "Real FIFA 2026 Format",
     "48 teams, 12 groups of 4. Top 2 from each group plus the best 8 third-place teams advance "
     "to a 32-team knockout bracket, with FIFA's official tiebreaker rules applied throughout."),
]

for icon, title, desc in items:
    st.markdown(
        f"""
        <div class="wc-card" style="display:flex; gap:14px; align-items:flex-start;">
            <div style="width:38px; height:38px; border-radius:10px; background:rgba(59,130,246,0.12);
                        display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0;">
                {icon}
            </div>
            <div>
                <div style="font-weight:700; font-size:15px; margin-bottom:4px;">{title}</div>
                <div class="wc-muted" style="font-size:13px; line-height:1.6;">{desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p class="wc-muted" style="font-size:11.5px;">'
    "Unofficial fan-built forecasting project. Not affiliated with or endorsed by FIFA. "
    "Probabilities are statistical estimates for entertainment and educational purposes only."
    "</p>",
    unsafe_allow_html=True,
)

render_footer()
