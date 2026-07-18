"""
pages/2_Groups.py
---------------------
The 12 real groups with their actual final standings — W/D/L/GF/GA/Pts
computed from the real matches in data/results_2026.csv.

Fix from v1: switched from st.markdown(unsafe_allow_html=True) to
st.components.v1.html() for the group card rows, which Streamlit's
sandboxing was stripping of their inline styles.
"""

import streamlit as st
import streamlit.components.v1 as components
from src.theme import inject_css, flag, COLORS, render_bottom_nav, render_footer
from src.bracket_data import group_standings
from src.tournament import GROUPS

st.set_page_config(page_title="Groups — WC26", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_bottom_nav("groups")

st.markdown("## 🌍 Group Stage — Final Standings")
st.markdown(
    '<p class="wc-muted">All 12 groups, 72 matches, as they actually played out. '
    "The top 2 in each group plus the best 8 third-place teams advanced to the Round of 32.</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

standings = group_standings()
letters = sorted(GROUPS.keys())

BG = COLORS["bg"]
CARD = COLORS["card"]
MUTED = COLORS["text_muted"]
ACCENT = COLORS["accent"]


def group_card_html(letter, teams):
    rows = ""
    for pos, t in enumerate(teams):
        bg = f"background:rgba(59,130,246,0.08);" if pos < 2 else ""
        rows += f"""
        <div style="display:flex;align-items:center;padding:9px 14px;{bg}
                    border-top:1px solid rgba(255,255,255,0.04);font-size:13px;font-family:system-ui,sans-serif;">
            <span style="width:18px;color:rgba(255,255,255,0.3);font-weight:700;font-size:11px;flex-shrink:0;">{pos+1}</span>
            <span style="font-size:18px;margin-right:8px;flex-shrink:0;">{flag(t['team'])}</span>
            <span style="flex:1;font-weight:600;color:rgba(255,255,255,0.92);
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:6px;">{t['team']}</span>
            <span style="width:24px;text-align:center;color:rgba(255,255,255,0.65);">{t['w']}</span>
            <span style="width:24px;text-align:center;color:rgba(255,255,255,0.65);">{t['d']}</span>
            <span style="width:24px;text-align:center;color:rgba(255,255,255,0.65);">{t['l']}</span>
            <span style="width:32px;text-align:center;font-weight:800;color:#fff;">{t['pts']}</span>
        </div>
        """
    return f"""
    <div style="background:{CARD};border:1px solid rgba(255,255,255,0.06);
                border-radius:18px;overflow:hidden;margin-bottom:14px;font-family:system-ui,sans-serif;">
        <div style="padding:12px 16px;background:rgba(255,255,255,0.02);
                    border-bottom:1px solid rgba(255,255,255,0.05);
                    font-weight:700;font-size:14px;color:#fff;">
            Group {letter}
        </div>
        <div style="display:flex;padding:7px 14px 5px;font-size:9px;font-weight:700;
                    letter-spacing:1px;text-transform:uppercase;color:{MUTED};">
            <span style="width:18px;"></span>
            <span style="flex:1;margin-left:26px;">Team</span>
            <span style="width:24px;text-align:center;">W</span>
            <span style="width:24px;text-align:center;">D</span>
            <span style="width:24px;text-align:center;">L</span>
            <span style="width:32px;text-align:center;">PTS</span>
        </div>
        {rows}
    </div>
    """


# Render all 12 groups in 3 columns using components.html so styles aren't stripped
col1, col2, col3 = st.columns(3)
col_map = {0: col1, 1: col2, 2: col3}

for idx, letter in enumerate(letters):
    teams = standings[letter]
    html = f"""
    <html><body style="margin:0;padding:0;background:transparent;">
    {group_card_html(letter, teams)}
    </body></html>
    """
    # Height: header (~44) + col-head (~28) + 4 rows (~42 each) + bottom padding
    height = 44 + 28 + len(teams) * 42 + 20
    with col_map[idx % 3]:
        components.html(html, height=height, scrolling=False)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p class="wc-muted">Blue-highlighted rows advanced automatically as group winner or runner-up. '
    "Some 3rd-place teams also advanced as one of the 8 best across all groups — see "
    "<b>Path to the Final</b> for the full Round of 32 bracket.</p>",
    unsafe_allow_html=True,
)

render_footer()
