"""Small, reusable layout components rendered with HTML + the global theme."""

from __future__ import annotations

from html import escape

import streamlit as st

MEDICAL_DISCLAIMER = (
    "<b>Research demonstration, not a diagnostic device.</b> Predictions come from an experimental "
    "machine-learning model trained on the HAM10000 dermatoscopic dataset. They are not medical diagnoses "
    "and do not replace evaluation by a dermatologist. Model confidence is not the probability that a "
    "patient has cancer. Performance can differ across cameras, dermatoscopes, skin types and populations, "
    "and images unlike the training data (for example, ordinary phone photos) may produce unreliable results."
)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="sc-page-title"><h1>{escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str, body: str) -> None:
    st.markdown(
        f"""<div class="sc-hero"><div class="sc-eyebrow">{escape(eyebrow)}</div>
        <h1>{escape(title)}</h1><div class="sc-sub">{escape(subtitle)}</div><p>{body}</p></div>""",
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str, kicker: str = "", chip: str = "", tile: bool = False) -> None:
    """A bordered information card. ``body_html`` must be trusted, pre-escaped HTML."""
    k = f'<div class="sc-kicker">{escape(kicker)}</div>' if kicker else ""
    cls = "sc-card tile" if tile else "sc-card"
    st.markdown(f'<div class="{cls}">{k}<h4>{escape(title)} {chip}</h4>{body_html}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "primary") -> str:
    return f'<span class="sc-chip {kind}">{escape(text)}</span>'


def notice(html: str, kind: str = "") -> None:
    st.markdown(f'<div class="sc-notice {kind}">{html}</div>', unsafe_allow_html=True)


def disclaimer(compact: bool = False) -> None:
    if compact:
        notice(
            "<b>Not a medical diagnosis.</b> Experimental research model; results do not replace a dermatologist. "
            "Confidence is model certainty, not the probability of cancer.",
            "warn",
        )
    else:
        notice(MEDICAL_DISCLAIMER, "warn")


def flow(steps: list[tuple[str, str]]) -> None:
    """Horizontal pipeline diagram; each step is (label, style) with style in {'', 'teal', 'ink'}."""
    parts = []
    for i, (label, style) in enumerate(steps):
        if i:
            parts.append('<span class="arrow">&rarr;</span>')
        parts.append(f'<span class="step {style}">{escape(label)}</span>')
    st.markdown(f'<div class="sc-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list[str]], numeric_cols: set[int] | None = None) -> None:
    """Simple styled table. Cell values are escaped."""
    numeric_cols = numeric_cols or set()
    th = "".join(f'<th class="{"num" if i in numeric_cols else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in numeric_cols else ""}">{escape(str(c))}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    st.markdown(f'<table class="sc-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="sc-footer">Reconstruction of a 2024 BCA academic project &middot; '
        "HAM10000 (CC BY-NC 4.0) &middot; For research and education only</div>",
        unsafe_allow_html=True,
    )


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or x != x:  # NaN
        return "n/a"
    return f"{100 * x:.{digits}f}%"
