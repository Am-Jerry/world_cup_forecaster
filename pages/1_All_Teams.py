"""
pages/1_All_Teams.py
-------------------------
Full probability table for all 48 real teams, reading directly from
outputs/simulation_results.csv. Sortable via the column headers Streamlit
gives st.dataframe for free.
"""

import streamlit as st
from src.theme import inject_css, load_results, flag, render_bottom_nav, render_footer

st.set_page_config(page_title="All Teams — WC26", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_bottom_nav("teams")

st.markdown("## 📊 All 48 Teams")
st.markdown(
    '<p class="wc-muted">Every team\'s probability of reaching each stage. '
    "Teams already eliminated show 0% for every stage beyond their real exit point — "
    "that's not a forecast, it's what actually happened.</p>",
    unsafe_allow_html=True,
)

df = load_results().copy()
df.insert(0, "Flag", df["team"].apply(flag))
df = df.sort_values("Winner%", ascending=False).reset_index(drop=True)
df.index = df.index + 1

search = st.text_input("Search for a team", placeholder="e.g. Argentina")
if search:
    df = df[df["team"].str.contains(search, case=False)]

st.dataframe(
    df,
    use_container_width=True,
    height=560,
    column_config={
        "Flag": st.column_config.TextColumn("", width="small"),
        "team": st.column_config.TextColumn("Team"),
        "R32%": st.column_config.ProgressColumn("R32%", min_value=0, max_value=100, format="%.0f%%"),
        "R16%": st.column_config.ProgressColumn("R16%", min_value=0, max_value=100, format="%.0f%%"),
        "QF%": st.column_config.ProgressColumn("QF%", min_value=0, max_value=100, format="%.0f%%"),
        "SF%": st.column_config.ProgressColumn("SF%", min_value=0, max_value=100, format="%.0f%%"),
        "Final%": st.column_config.ProgressColumn("Final%", min_value=0, max_value=100, format="%.1f%%"),
        "Winner%": st.column_config.ProgressColumn("Winner%", min_value=0, max_value=100, format="%.1f%%"),
    },
)

render_footer()
