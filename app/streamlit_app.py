"""
app/streamlit_app.py
The dashboard's entry point: the theme, the shared sidebar (business assumptions) and the top
navigation. Every page reads the three assumptions from st.session_state ("r", "d", "months"), so a
slider moved here changes all pages at once.
Run locally:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.theme import apply_theme
from churn_core import config
from churn_core.economics import cutoff_from

st.set_page_config(page_title="Churn decisions", page_icon=PROJECT_ROOT/"docs"/"churn_decisions_favicon.ico", layout="wide")
apply_theme()

defaults = config.default_assumptions()
with st.sidebar:
    st.markdown('<div class="brand">Churn decisions</div><div class="brand-sub">Telecom churn prediction</div>',
                unsafe_allow_html=True)
    st.subheader("Business assumptions")
    st.caption("These are assumptions, not measured facts: the dataset has no record of offers or responses.")
    r = st.slider("Offer success rate", 0.05, 1.0, float(defaults.r), 0.05, key="r",
                  help="Share of contacted churners who accept the offer and stay.")
    d = st.slider("Discount", 0.0, 0.5, float(defaults.d), 0.05, key="d",
                  help="Discount on the monthly bill, given to everyone who is contacted.")
    st.slider("Months a retained customer keeps paying", 1, 36, int(defaults.months), key="months")
    st.metric("Break-even probability", f"{cutoff_from(r, d):.1%}",
              help="Above this chance of leaving, an offer earns more than it costs on average.")
    st.caption(f"Model: {config.MODEL_VERSION}")

pages = [
    st.Page("views/predict.py", title="Predict", icon=":material/person_search:", default=True),
    st.Page("views/simulator.py", title="Simulator", icon=":material/tune:"),
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("views/performance.py", title="Performance", icon=":material/monitoring:"),
    st.Page("views/explorer.py", title="Explorer", icon=":material/table_rows:"),
]
st.navigation(pages, position="top").run()