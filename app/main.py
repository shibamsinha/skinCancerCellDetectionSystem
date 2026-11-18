"""Streamlit entry point.

Run from the repository root:

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.theme import apply_theme  # noqa: E402

PAGES_DIR = Path(__file__).resolve().parent / "pages"

st.set_page_config(
    page_title="Skin Cancer Classification · Modified Deep Learning Protocol",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

pages = [
    st.Page(PAGES_DIR / "home.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page(PAGES_DIR / "prediction.py", title="Classify an image", icon=":material/upload_file:", url_path="predict"),
    st.Page(PAGES_DIR / "model_comparison.py", title="Model comparison", icon=":material/bar_chart:", url_path="models"),
    st.Page(PAGES_DIR / "project_information.py", title="Project information", icon=":material/menu_book:", url_path="about"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown(
        "<div style='padding:4px 2px 10px'><div style='font-weight:700;font-size:1.05rem;color:#14202e'>"
        "Skin Cancer Classification</div><div style='color:#0e8f84;font-size:0.85rem;font-weight:600'>"
        "Modified Deep Learning Protocol</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Research demonstration reconstructed from a 2024 BCA project. "
        "Not a medical device; predictions are not diagnoses."
    )

nav.run()
