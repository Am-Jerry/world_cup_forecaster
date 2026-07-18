"""
pages/3_Path_to_Final.py
----------------------------
The real knockout bracket rendered as a proper left-to-right SVG/HTML bracket.
Round tabs at the top are interactive — clicking one smoothly scrolls the
bracket so that round is centred in view, with no page reload or content swap.
The bracket itself is unchanged from the original visual design; only the
tab → scroll behaviour is new.
"""

import streamlit as st
import streamlit.components.v1 as components
from src.theme import inject_css, load_results, flag, COLORS, render_bottom_nav, render_footer
from src.bracket_data import build_full_bracket

st.set_page_config(page_title="Path to the Final — WC26", page_icon="🏆", layout="wide", initial_sidebar_state="collapsed")
inject_css()
render_bottom_nav("final")

st.markdown("## 🏆 Path to the Final")
st.markdown(
    '<p class="wc-muted">Click a round tab to jump to it. Scroll left/right inside the bracket '
    "for the full picture. Completed matches show the real result; the Final shows simulated odds.</p>",
    unsafe_allow_html=True,
)

bracket    = build_full_bracket()
results_df = load_results()

CARD  = COLORS["card"]
MUTED = COLORS["text_muted"]
BG    = COLORS["bg"]
GOLD  = COLORS["gold"]
ACCENT = COLORS["accent"]

# ---------------------------------------------------------------------------
# Layout constants — match the original design
# ---------------------------------------------------------------------------
CARD_W    = 230
CARD_H    = 88
V_GAP     = 18
COL_GAP   = 56
TOP_PAD   = 32
unit      = CARD_H + V_GAP

ROUND_ORDER  = ["R32", "R16", "QF", "SF", "Final"]
ROUND_LABELS = {
    "R32":   "Round of 32",
    "R16":   "Round of 16",
    "QF":    "Quarterfinals",
    "SF":    "Semifinals",
    "Final": "Final",
}

# ---------------------------------------------------------------------------
# Layout geometry
# ---------------------------------------------------------------------------
positions    = {}
prev_centers = []

for r_idx, rk in enumerate(ROUND_ORDER):
    matches = bracket[rk]
    centers = []
    for i, m in enumerate(matches):
        if r_idx == 0:
            raw = i * unit + unit / 2
        else:
            raw = (prev_centers[2 * i] + prev_centers[2 * i + 1]) / 2
        centers.append(raw)
        cy = raw + TOP_PAD
        positions[(rk, i)] = {
            "x":      r_idx * (CARD_W + COL_GAP),
            "y":      cy - CARD_H / 2,
            "center": cy,
        }
    prev_centers = centers

total_h = len(bracket["R32"]) * unit + TOP_PAD + 24
total_w = len(ROUND_ORDER) * CARD_W + (len(ROUND_ORDER) - 1) * COL_GAP

# ---------------------------------------------------------------------------
# SVG connector lines (winner exit → next-round entry)
# ---------------------------------------------------------------------------
connectors = []
for r_idx in range(len(ROUND_ORDER) - 1):
    ck, pk = ROUND_ORDER[r_idx], ROUND_ORDER[r_idx + 1]
    for p_idx, parent in enumerate(bracket[pk]):
        for offset in (0, 1):
            c_idx = p_idx * 2 + offset
            children = bracket[ck]
            if c_idx >= len(children):
                continue
            child = children[c_idx]
            if not child.get("winner"):
                continue
            cp = positions[(ck, c_idx)]
            pp = positions[(pk, p_idx)]
            sx  = cp["x"] + CARD_W
            sy  = cp["center"] + (-1 if offset == 0 else 1) * (CARD_H / 4)
            ex  = pp["x"]
            ey  = pp["center"] + (-1 if parent.get("t1") == child["winner"] else 1) * (CARD_H / 4)
            mx  = sx + COL_GAP / 2
            connectors.append(f"M {sx} {sy} H {mx} V {ey} H {ex}")

svg_paths = "".join(
    f'<path d="{d}" fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="1.8"/>'
    for d in connectors
)

# ---------------------------------------------------------------------------
# Match card HTML builder
# ---------------------------------------------------------------------------
def winner_pct(team: str) -> float:
    if not team:
        return 0.0
    row = results_df[results_df["team"] == team]
    return float(row["Winner%"].iloc[0]) if not row.empty else 0.0


def team_row(team, score, is_winner, placeholder=None):
    if team is None:
        return (
            f'<div style="padding:5px 10px;font-size:11px;'
            f'font-style:italic;color:rgba(255,255,255,0.28);">'
            f'{placeholder}</div>'
        )
    w = "700" if is_winner else "500"
    c = "#fff" if is_winner else "rgba(255,255,255,0.42)"
    score_html = (
        f'<span style="font-size:13px;font-weight:{w};color:{c};">{score}</span>'
        if score is not None else ""
    )
    return (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;padding:5px 10px;">'
        f'<div style="display:flex;align-items:center;gap:6px;min-width:0;">'
        f'<span style="font-size:16px;flex-shrink:0;">{flag(team)}</span>'
        f'<span style="font-size:12px;font-weight:{w};color:{c};'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team}</span>'
        f'</div>{score_html}</div>'
    )


def match_card(m, rk, pos):
    status = m["status"]
    t1, t2 = m.get("t1"), m.get("t2")
    p = positions[(rk, pos)]

    badge_bg = "rgba(59,130,246,0.18)" if status == "UPCOMING" else "rgba(255,255,255,0.07)"
    badge_c  = "#60A5FA"               if status == "UPCOMING" else "rgba(255,255,255,0.45)"

    if rk == "Final" and t1 is None:
        body = (team_row(None, None, False, "Winner · SF1") +
                '<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 8px;"></div>' +
                team_row(None, None, False, "Winner · SF2"))
        foot = ""
    elif status == "UPCOMING":
        p1, p2 = winner_pct(t1), winner_pct(t2)
        body = (team_row(t1, f"{p1:.0f}%", False) +
                '<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 8px;"></div>' +
                team_row(t2, f"{p2:.0f}%", False))
        foot = (f'<div style="padding:1px 10px 5px;font-size:8px;font-weight:700;'
                f'letter-spacing:.8px;text-transform:uppercase;'
                f'color:rgba(96,165,250,0.65);">Win odds</div>')
    else:
        s1, s2 = m["score"]
        w = m.get("winner")
        body = (team_row(t1, s1, w == t1) +
                '<div style="height:1px;background:rgba(255,255,255,0.05);margin:0 8px;"></div>' +
                team_row(t2, s2, w == t2))
        foot = ""

    return (
        f'<div style="position:absolute;left:{p["x"]}px;top:{p["y"]}px;'
        f'width:{CARD_W}px;height:{CARD_H}px;'
        f'background:{CARD};border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:12px;overflow:hidden;font-family:system-ui,sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;padding:5px 10px 2px;">'
        f'<span style="font-size:9px;font-weight:700;letter-spacing:1px;'
        f'text-transform:uppercase;color:{MUTED};">{ROUND_LABELS[rk]}</span>'
        f'<span style="font-size:8px;font-weight:700;letter-spacing:.8px;padding:2px 7px;'
        f'border-radius:10px;background:{badge_bg};color:{badge_c};">{status}</span>'
        f'</div>'
        f'<div style="border-top:1px solid rgba(255,255,255,0.04);margin-top:2px;">'
        f'{body}</div>{foot}</div>'
    )


# Build all cards
all_cards = "".join(
    match_card(m, rk, i)
    for rk in ROUND_ORDER
    for i, m in enumerate(bracket[rk])
)

# Round label headers
round_headers = "".join(
    f'<div id="label-{rk}" style="position:absolute;left:{positions[(rk,0)]["x"]}px;top:0;'
    f'width:{CARD_W}px;text-align:center;font-size:10px;font-weight:700;'
    f'letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};">'
    f'{ROUND_LABELS[rk]}</div>'
    for rk in ROUND_ORDER
)

# ---------------------------------------------------------------------------
# Compute the x-offset each tab should scroll to (centre that round)
# ---------------------------------------------------------------------------
round_scroll_x = {}
for rk in ROUND_ORDER:
    col_x = positions[(rk, 0)]["x"]
    round_scroll_x[rk] = max(0, col_x - 40)   # 40px left margin

scroll_map_js = "{" + ",".join(f'"{k}":{v}' for k, v in round_scroll_x.items()) + "}"

# ---------------------------------------------------------------------------
# Full HTML component
# ---------------------------------------------------------------------------
html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; font-family: system-ui, -apple-system, sans-serif; }}

  /* ── Tab bar ── */
  .tab-bar {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px;
    background: #121212;
    border-radius: 999px;
    width: fit-content;
    margin: 0 auto 16px;
    border: 1px solid rgba(255,255,255,0.06);
  }}
  .tab-btn {{
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    cursor: pointer;
    color: rgba(255,255,255,0.4);
    background: transparent;
    border: none;
    outline: none;
    transition: all .2s;
    white-space: nowrap;
  }}
  .tab-btn:hover  {{ color: rgba(255,255,255,.8); }}
  .tab-btn.active {{ color: #fff; background: #2C2C2E; }}

  /* ── Bracket scroll container ── */
  .bracket-wrap {{
    overflow-x: auto;
    overflow-y: auto;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    background: rgba(255,255,255,0.012);
    scroll-behavior: smooth;
  }}
  .bracket-wrap::-webkit-scrollbar {{ height: 5px; width: 5px; }}
  .bracket-wrap::-webkit-scrollbar-track {{ background: transparent; }}
  .bracket-wrap::-webkit-scrollbar-thumb {{ background: rgba(59,130,246,.35); border-radius: 3px; }}
</style>
</head>
<body>

<!-- Tab bar -->
<div class="tab-bar" id="tabBar">
  <button class="tab-btn active" data-round="R32"   onclick="jumpTo('R32')">R32</button>
  <button class="tab-btn"        data-round="R16"   onclick="jumpTo('R16')">R16</button>
  <button class="tab-btn"        data-round="QF"    onclick="jumpTo('QF')">QF</button>
  <button class="tab-btn"        data-round="SF"    onclick="jumpTo('SF')">SF</button>
  <button class="tab-btn"        data-round="Final" onclick="jumpTo('Final')">🏆 Final</button>
</div>

<!-- Bracket canvas -->
<div class="bracket-wrap" id="bracketWrap" style="height:68vh;">
  <div style="position:relative;width:{total_w}px;height:{total_h + 30}px;">

    <!-- SVG connectors -->
    <svg width="{total_w}" height="{total_h}"
         style="position:absolute;top:26px;left:0;overflow:visible;">
      <defs>
        <marker id="arr" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="rgba(255,255,255,0.7)"/>
        </marker>
      </defs>
      {svg_paths}
    </svg>

    <!-- Round headers + match cards -->
    <div style="position:relative;top:26px;">
      {round_headers}
      {all_cards}
    </div>

  </div>
</div>

<script>
const SCROLL_X = {scroll_map_js};

function jumpTo(round) {{
  // Update active tab
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.dataset.round === round);
  }});
  // Scroll bracket horizontally to that round
  const wrap = document.getElementById('bracketWrap');
  wrap.scrollTo({{ left: SCROLL_X[round], behavior: 'smooth' }});
}}

// Update active tab when user scrolls manually
document.getElementById('bracketWrap').addEventListener('scroll', function() {{
  const scrollX = this.scrollLeft;
  const rounds = ["R32","R16","QF","SF","Final"];
  let closest = rounds[0];
  let minDist = Infinity;
  rounds.forEach(r => {{
    const dist = Math.abs(SCROLL_X[r] - scrollX);
    if (dist < minDist) {{ minDist = dist; closest = r; }}
  }});
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.classList.toggle('active', btn.dataset.round === closest);
  }});
}});
</script>
</body>
</html>
"""

components.html(html, height=780, scrolling=False)

render_footer()
