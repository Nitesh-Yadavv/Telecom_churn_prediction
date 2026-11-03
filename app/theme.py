"""
app/theme.py

The small amount of styling the theme file cannot do: a serif for the big numbers, white cards on the
off-white page, and the brand block in the sidebar. Colours and fonts mirror .streamlit/config.toml.
"""
import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: "Source Serif 4", Georgia, serif; }
[data-testid="stMetricValue"] { font-feature-settings: "tnum" 1; }
[class*="st-key-card"] { background: #FFFFFF; border-radius: 12px; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #E6F3F0; border-radius: 12px; padding: 0.7rem 0.9rem;
}
.brand { font-family: "Source Serif 4", Georgia, serif; font-weight: 600; font-size: 1.4rem;
         color: #0F2A36; line-height: 1.2; }
.brand-sub { color: #4F6674; font-size: 0.85rem; margin: 0 0 1rem 0; }
/* Darker teal for dropdown selectboxes and inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    background-color: #C8E6DF !important;
    border: 1px solid #ACD5CB !important;
    border-radius: 10px;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"]:hover {
    border-color: #0B7F77 !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: #0B7F77 !important;
    box-shadow: 0 0 0 1px #0B7F77 !important;
}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)