"""
Presentation helpers for the Streamlit UI: the stylesheet and a few small
HTML fragments that Streamlit has no native widget for (status chips, file
pills, the empty-state hero).

Everything here is cosmetic. The app stays fully usable if a future
Streamlit release changes the internal test-ids these rules hang off, so
selectors are additive and never rely on `!important` to hide core chrome.
"""

from __future__ import annotations

import html

import streamlit as st

# Colours are expressed against Streamlit's own theme variables where they
# exist, so the app reads correctly whether the user runs the bundled dark
# theme or flips to light in Streamlit's settings menu.
STYLESHEET = """
<style>
:root {
    --sp-accent: var(--primary-color, #6d5efc);
    --sp-surface: color-mix(in srgb, var(--text-color, #e6edf3) 4%, transparent);
    --sp-surface-strong: color-mix(in srgb, var(--text-color, #e6edf3) 8%, transparent);
    --sp-border: color-mix(in srgb, var(--text-color, #e6edf3) 14%, transparent);
    --sp-muted: color-mix(in srgb, var(--text-color, #e6edf3) 62%, transparent);
    --sp-ok: #2ea043;
    --sp-warn: #d29922;
    --sp-bad: #f85149;
}

/* Give the main column room to breathe without going edge-to-edge.
   The generous top padding keeps page titles clear of the container's
   clipping edge -- at 2.2rem the cap heights were being shaved off. */
.stMainBlockContainer { padding-top: 3.4rem; max-width: 1500px; }

/* ---------- brand header ---------- */
.sp-header { display: flex; align-items: center; gap: .85rem; margin-bottom: .55rem; }
.sp-mark {
    width: 38px; height: 38px; border-radius: 11px; flex: 0 0 38px;
    background: linear-gradient(135deg, var(--sp-accent), #22d3ee);
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; box-shadow: 0 6px 18px -8px var(--sp-accent);
}
.sp-title { font-size: 1.42rem; font-weight: 680; letter-spacing: -.02em; line-height: 1.35; }
.sp-subtitle { color: var(--sp-muted); font-size: .86rem; margin-top: .1rem; }

/* ---------- status chips ---------- */
.sp-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .55rem 0 .2rem; }
.sp-chip {
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .2rem .6rem; border-radius: 999px;
    background: var(--sp-surface); border: 1px solid var(--sp-border);
    font-size: .765rem; color: var(--sp-muted); white-space: nowrap;
}
.sp-chip b { color: var(--text-color, inherit); font-weight: 600; }
.sp-chip.ok    { border-color: color-mix(in srgb, var(--sp-ok) 45%, transparent); }
.sp-chip.warn  { border-color: color-mix(in srgb, var(--sp-warn) 45%, transparent); }
.sp-chip.bad   { border-color: color-mix(in srgb, var(--sp-bad) 45%, transparent); }
.sp-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--sp-muted); }
.sp-dot.ok { background: var(--sp-ok); } .sp-dot.warn { background: var(--sp-warn); }
.sp-dot.bad { background: var(--sp-bad); }

/* ---------- chat ---------- */
.stChatMessage {
    background: transparent; padding: .35rem 0 .1rem; border-radius: 14px;
}
.stChatMessage:has(.sp-user-marker) {
    background: var(--sp-surface); border: 1px solid var(--sp-border);
    padding: .55rem .9rem; margin-left: 8%;
}
.sp-user-marker { display: none; }

/* attachment pills inside a message */
.sp-files { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .45rem; }
.sp-file {
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .18rem .55rem; border-radius: 7px;
    background: var(--sp-surface-strong); border: 1px solid var(--sp-border);
    font-size: .76rem; font-family: ui-monospace, "SF Mono", Menlo, monospace;
}

/* ---------- empty state ---------- */
.sp-hero {
    border: 1px dashed var(--sp-border); border-radius: 16px;
    padding: 2.1rem 1.6rem; text-align: center; background: var(--sp-surface);
    margin: .6rem 0 1.1rem;
}
.sp-hero h3 { margin: 0 0 .35rem; font-size: 1.12rem; font-weight: 640; }
.sp-hero p { margin: 0 auto; max-width: 46ch; color: var(--sp-muted); font-size: .89rem; line-height: 1.55; }
.sp-hero-icon { font-size: 2rem; display: block; margin-bottom: .5rem; }

/* ---------- artifact panel ---------- */
.sp-panel-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: .5rem; margin-bottom: .3rem;
}
.sp-panel-title { font-weight: 640; font-size: .95rem; letter-spacing: -.01em; }
.sp-panel-sub { color: var(--sp-muted); font-size: .78rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stSelectbox label p { font-size: .84rem; }
.sp-rule {
    background: var(--sp-surface); border: 1px solid var(--sp-border);
    border-left: 3px solid var(--sp-accent);
    border-radius: 8px; padding: .45rem .6rem; font-size: .8rem; line-height: 1.45;
}
.sp-rule .sp-scope {
    display: block; font-size: .68rem; text-transform: uppercase;
    letter-spacing: .06em; color: var(--sp-muted); margin-bottom: .15rem;
}

/* ---------- chat input ---------- */
.stChatInput textarea { font-size: .93rem; }

/* Tool-call expanders should read as quiet metadata, not primary content. */
.sp-toolcall summary { font-size: .8rem; }
</style>
"""


def inject_styles() -> None:
    """Add the app stylesheet. Safe to call on every rerun."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


def header(title: str, subtitle: str, icon: str = "🧬") -> None:
    st.markdown(
        f"""
        <div class="sp-header">
          <div class="sp-mark">{icon}</div>
          <div>
            <div class="sp-title">{html.escape(title)}</div>
            <div class="sp-subtitle">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chips(items: list[tuple[str, str]]) -> None:
    """Render a row of status chips. Each item is (text, tone).

    Tone is one of "", "ok", "warn", "bad" and colours both the dot and the
    chip border.
    """
    if not items:
        return
    parts = [
        f'<span class="sp-chip {tone}"><span class="sp-dot {tone}"></span>{text}</span>'
        for text, tone in items
    ]
    st.markdown(f'<div class="sp-chips">{"".join(parts)}</div>', unsafe_allow_html=True)


def file_pills(names: list[str]) -> str:
    """HTML for a row of attached-file pills (returned, not written)."""
    if not names:
        return ""
    pills = "".join(
        f'<span class="sp-file">📄 {html.escape(n)}</span>' for n in names
    )
    return f'<div class="sp-files">{pills}</div>'


def hero(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="sp-hero">
          <span class="sp-hero-icon">{icon}</span>
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rule_card(rule: str, scope: str) -> None:
    st.markdown(
        f'<div class="sp-rule"><span class="sp-scope">{html.escape(scope)}</span>'
        f"{html.escape(rule)}</div>",
        unsafe_allow_html=True,
    )


def panel_head(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sp-panel-head"><span class="sp-panel-title">{html.escape(title)}</span>'
        f'<span class="sp-panel-sub">{html.escape(subtitle)}</span></div>',
        unsafe_allow_html=True,
    )
