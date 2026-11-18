"""Visual theme: design tokens and global CSS."""

from __future__ import annotations

import streamlit as st

# Design tokens (light, clinical-research aesthetic: blue/teal accents on neutral surfaces).
INK = "#14202e"
INK_SECONDARY = "#4b5868"
INK_MUTED = "#7a8696"
SURFACE = "#ffffff"
CANVAS = "#f5f7fa"
BORDER = "#e3e8ef"
PRIMARY = "#1f6fc5"
PRIMARY_SOFT = "#e8f1fb"
TEAL = "#0e8f84"
TEAL_SOFT = "#e5f5f3"
# Status colours are reserved for state and always paired with an icon and a label.
GOOD = "#0b8a3e"
GOOD_SOFT = "#e9f7ef"
CRITICAL = "#c23a3a"
CRITICAL_SOFT = "#fbeceb"
WARNING = "#a86a00"
WARNING_SOFT = "#fff6e5"

_CSS = f"""
<style>
:root {{
  --ink: {INK}; --ink-2: {INK_SECONDARY}; --ink-3: {INK_MUTED};
  --surface: {SURFACE}; --canvas: {CANVAS}; --border: {BORDER};
  --primary: {PRIMARY}; --primary-soft: {PRIMARY_SOFT};
  --teal: {TEAL}; --teal-soft: {TEAL_SOFT};
  --good: {GOOD}; --good-soft: {GOOD_SOFT};
  --critical: {CRITICAL}; --critical-soft: {CRITICAL_SOFT};
  --warning: {WARNING}; --warning-soft: {WARNING_SOFT};
}}
html, body, [class*="css"] {{
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}
h2, h3 {{ font-weight: 650; }}
[data-testid="stSidebar"] {{ background: var(--surface); }}
[data-testid="stSidebarNav"] a span {{ font-weight: 500; }}
[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: var(--ink-2); font-size: 0.82rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: var(--ink); font-size: clamp(1.2rem, 2.2vw, 1.55rem); font-weight: 650; }}
/* Wrap instead of truncating ("Model co...") when the cards are narrow. */
[data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] > div {{ white-space: normal; overflow: visible; text-overflow: clip; }}
[data-testid="stFileUploaderDropzone"] {{
  border: 1.5px dashed #b8c6d8; background: #fbfcfe; border-radius: 12px;
}}
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: var(--border); background: var(--surface); }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ padding: 6px 14px; }}

/* ---------- custom components ---------- */
.sc-hero {{
  background: linear-gradient(180deg, #ffffff 0%, #f7fafd 100%);
  border: 1px solid var(--border); border-radius: 18px; padding: 34px 36px 28px; margin-bottom: 18px;
}}
.sc-eyebrow {{ color: var(--teal); font-weight: 600; font-size: 0.78rem; letter-spacing: .08em; text-transform: uppercase; }}
.sc-hero h1 {{ font-size: 2.35rem; margin: 6px 0 2px; line-height: 1.15; }}
.sc-hero .sc-sub {{ color: var(--primary); font-size: 1.15rem; font-weight: 550; margin-bottom: 12px; }}
.sc-hero p {{ color: var(--ink-2); font-size: 1.0rem; line-height: 1.6; max-width: 820px; margin: 0; }}
.sc-page-title {{ margin-bottom: 4px; }}
.sc-page-title h1 {{ font-size: 1.9rem; margin: 0; }}
.sc-page-title p {{ color: var(--ink-2); margin: 4px 0 0; font-size: 0.98rem; }}
.sc-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px;
  height: 100%; box-shadow: 0 1px 2px rgba(20, 32, 46, 0.04);
}}
.sc-card h4 {{ margin: 0 0 6px; font-size: 1.0rem; }}
.sc-card p, .sc-card li {{ color: var(--ink-2); font-size: 0.92rem; line-height: 1.55; margin: 0; }}
.sc-card ul {{ padding-left: 1.1rem; margin: 6px 0 0; }}
.sc-card .sc-kicker {{ color: var(--ink-3); font-size: 0.76rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.sc-chip {{
  display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: 999px;
  font-size: 0.76rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}}
.sc-chip.ok {{ background: var(--good-soft); color: var(--good); border-color: #cbe9d6; }}
.sc-chip.off {{ background: #f1f3f6; color: var(--ink-3); border-color: var(--border); }}
.sc-chip.primary {{ background: var(--primary-soft); color: var(--primary); border-color: #cfe1f6; }}
.sc-chip.teal {{ background: var(--teal-soft); color: var(--teal); border-color: #c6e8e4; }}
.sc-chip.warn {{ background: var(--warning-soft); color: var(--warning); border-color: #f3dfb5; }}
.sc-notice {{
  border-radius: 12px; padding: 12px 16px; font-size: 0.88rem; line-height: 1.55; color: var(--ink-2);
  background: #f7f9fc; border: 1px solid var(--border); border-left: 4px solid var(--primary);
}}
.sc-notice.warn {{ border-left-color: var(--warning); background: #fffbf3; }}
.sc-notice b {{ color: var(--ink); }}
.sc-result {{
  border-radius: 16px; padding: 20px 22px; background: var(--surface); border: 1px solid var(--border);
  border-left: 6px solid var(--ink-3);
}}
.sc-result.benign {{ border-left-color: var(--good); }}
.sc-result.malignant {{ border-left-color: var(--critical); }}
.sc-result .sc-label {{ font-size: 0.8rem; color: var(--ink-3); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }}
.sc-result .sc-verdict {{ display: flex; align-items: center; gap: 10px; font-size: 1.9rem; font-weight: 700; margin: 4px 0 6px; }}
.sc-result.benign .sc-verdict {{ color: var(--good); }}
.sc-result.malignant .sc-verdict {{ color: var(--critical); }}
.sc-result .sc-icon {{
  display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 50%;
  font-size: 1.1rem; font-weight: 800;
}}
.sc-result.benign .sc-icon {{ background: var(--good-soft); }}
.sc-result.malignant .sc-icon {{ background: var(--critical-soft); }}
.sc-result p {{ color: var(--ink-2); font-size: 0.9rem; margin: 0; line-height: 1.5; }}
.sc-caption {{ color: var(--ink-3); font-size: 0.82rem; text-align: center; margin-top: 4px; }}
.sc-figure-title {{ font-weight: 600; color: var(--ink); font-size: 0.92rem; margin-bottom: 6px; }}
.sc-flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
.sc-flow .step {{
  background: var(--primary-soft); color: var(--primary); border-radius: 8px; padding: 6px 10px;
  font-size: 0.8rem; font-weight: 600; border: 1px solid #cfe1f6;
}}
.sc-flow .step.teal {{ background: var(--teal-soft); color: var(--teal); border-color: #c6e8e4; }}
.sc-flow .step.ink {{ background: #f1f3f6; color: var(--ink); border-color: var(--border); }}
.sc-flow .arrow {{ color: var(--ink-3); font-size: 0.9rem; }}
.sc-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
.sc-table th {{ text-align: left; color: var(--ink-3); font-weight: 600; font-size: 0.76rem; text-transform: uppercase;
  letter-spacing: .05em; border-bottom: 1px solid var(--border); padding: 8px 10px; }}
.sc-table td {{ border-bottom: 1px solid #eef1f5; padding: 8px 10px; color: var(--ink); }}
.sc-table td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
.sc-table th.num {{ text-align: right; }}
[data-testid="stPageLink"] a {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 8px 14px;
  justify-content: center; transition: border-color .15s ease, background .15s ease;
}}
[data-testid="stPageLink"] a:hover {{ border-color: #b9cfe8; background: var(--primary-soft); }}
[data-testid="stPageLink"] a p {{ font-weight: 600; color: var(--ink); }}
.sc-card.tile {{ min-height: 128px; }}
.sc-footer {{ color: var(--ink-3); font-size: 0.8rem; text-align: center; margin-top: 36px; }}
</style>
"""


def apply_theme() -> None:
    """Inject the global stylesheet (call once per page run)."""
    st.markdown(_CSS, unsafe_allow_html=True)
