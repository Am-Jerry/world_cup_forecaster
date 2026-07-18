"""
Home.py
-----------
Home page of the World Cup 2026 Forecaster. This is the entry point
Streamlit runs — the pages/ directory supplies the other tabs
automatically via the sidebar.
"""

import streamlit as st
from src.theme import inject_css, load_results, flag, COLORS, render_bottom_nav, render_footer
from src.bracket_data import build_full_bracket

st.set_page_config(
    page_title="WC26 Forecaster",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
render_bottom_nav("home")

df = load_results()
bracket = build_full_bracket()

# Only the two finalists have Winner% > 0 now that both SFs are decided
finalists = df[df["Winner%"] > 0].sort_values("Winner%", ascending=False)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="wc-eyebrow">● Live — Final Stage</div>', unsafe_allow_html=True)
st.markdown("## 2026 World Cup Forecast")
st.markdown(
    '<p class="wc-muted">Every match — group stage through both semifinals — is locked in as a '
    "real result. Only the Final remains. Simulated 10,000 times using the Elo + Poisson "
    "expected-goals model.</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# The Final fixture card
# ---------------------------------------------------------------------------
final_match = bracket["Final"][0]
t1 = final_match.get("t1") or "TBD"
t2 = final_match.get("t2") or "TBD"

def winner_pct(team):
    row = df[df["team"] == team]
    return float(row["Winner%"].iloc[0]) if not row.empty else 0.0

p1 = winner_pct(t1)
p2 = winner_pct(t2)

st.markdown("#### 🏆 The Final — July 19, MetLife Stadium")
st.markdown(
    f"""
    <div class="wc-card">
        <div class="wc-muted" style="font-size:11px; font-weight:700; letter-spacing:1.5px;
             text-transform:uppercase; margin-bottom:16px;">
            FIFA World Cup Final · East Rutherford, New Jersey
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:14px; width:42%;">
                <span style="font-size:42px;">{flag(t1)}</span>
                <div>
                    <div style="font-weight:700; font-size:18px;">{t1}</div>
                    <div style="font-size:22px; font-weight:800; color:{COLORS['gold']};">{p1:.1f}%</div>
                    <div class="wc-muted" style="font-size:11px; letter-spacing:1px; text-transform:uppercase;">Win probability</div>
                </div>
            </div>
            <div style="text-align:center;">
                <div class="wc-pill" style="font-size:14px; padding:8px 18px;">VS</div>
                <div class="wc-muted" style="font-size:11px; margin-top:8px;">Jul 19 · 4PM ET</div>
            </div>
            <div style="display:flex; align-items:center; gap:14px; width:42%; justify-content:flex-end; text-align:right;">
                <div>
                    <div style="font-weight:700; font-size:18px;">{t2}</div>
                    <div style="font-size:22px; font-weight:800; color:{COLORS['gold']};">{p2:.1f}%</div>
                    <div class="wc-muted" style="font-size:11px; letter-spacing:1px; text-transform:uppercase;">Win probability</div>
                </div>
                <span style="font-size:42px;">{flag(t2)}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# The two finalists — win probability cards only (no "Reach Final" row
# since Final% is 100 for both — that would be noise not signal)
# ---------------------------------------------------------------------------
st.markdown("#### The Finalists")
cols = st.columns(2)
for col, (_, row) in zip(cols, finalists.iterrows()):
    with col:
        st.markdown(
            f"""
            <div class="wc-card" style="text-align:center; padding:28px 22px;">
                <div style="font-size:52px; margin-bottom:8px;">{flag(row['team'])}</div>
                <div style="font-size:20px; font-weight:700; margin-bottom:18px;">{row['team']}</div>
                <div style="font-size:44px; font-weight:800; color:{COLORS['gold']}; line-height:1;">
                    {row['Winner%']:.1f}%
                </div>
                <div class="wc-muted" style="text-transform:uppercase; font-size:10px;
                     letter-spacing:1.5px; margin-top:6px;">
                    Championship probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p class="wc-muted">See <b>All Teams</b> for the full 48-team probability breakdown, '
    "<b>Groups</b> for the real group-stage standings, or "
    "<b>Path to the Final</b> for the complete bracket.</p>",
    unsafe_allow_html=True,
)

render_footer()
