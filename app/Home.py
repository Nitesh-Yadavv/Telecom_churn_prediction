"""
app/Home.py

One-page predictor: describe a customer, see their chance of leaving, the risk band, whether a
retention offer pays for itself under the assumptions in the sidebar, and why.
Run locally:  streamlit run app/Home.py   (with the API running)
"""
import plotly.graph_objects as go
import streamlit as st
from api_client import ApiError, explain, model_info, predict

from churn_core.economics import cutoff_from

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
ADD_ONS = [("online_security", "Online security"), ("online_backup", "Online backup"),
           ("device_protection", "Device protection"), ("tech_support", "Tech support"),
           ("streaming_tv", "Streaming TV"), ("streaming_movies", "Streaming movies")]
RAISES, LOWERS = "#D85A30", "#1D9E75"

st.set_page_config(page_title="Churn decision demo", layout="wide")


@st.cache_data(ttl=600, show_spinner=False)
def load_model_info() -> dict:
    return model_info()


st.title("Who is worth a retention offer?")
st.caption("Describe a customer. The model estimates their chance of leaving, and the assumptions in the "
           "sidebar decide whether an offer pays for itself. The first request may take a few seconds "
           "while the service wakes up.")

# ---------------- sidebar: business assumptions ----------------
try:
    with st.spinner("Contacting the prediction service..."):
        info = load_model_info()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

defaults = info["assumptions"]
with st.sidebar:
    st.header("Business assumptions")
    st.caption("These are assumptions, not measured facts: the dataset has no record of offers or responses.")
    r = st.slider("Offer success rate", 0.05, 1.0, float(defaults["r"]), 0.05,
                  help="Share of contacted churners who accept the offer and stay.")
    d = st.slider("Discount", 0.0, 0.5, float(defaults["d"]), 0.05,
                  help="Discount on the monthly bill, given to everyone who is contacted.")
    months = st.slider("Months a retained customer keeps paying", 1, 36, int(defaults["months"]))
    cutoff = cutoff_from(r, d)
    st.metric("Break-even probability", f"{cutoff:.1%}",
              help="Above this chance of leaving, an offer earns more than it costs on average.")
    st.caption(f"Model: {info['model_version']}")

# ---------------- the customer ----------------
st.subheader("The customer")
left, middle, right = st.columns(3)
with left:
    contract = st.selectbox("Contract", CONTRACTS, key="contract")
    tenure = st.number_input("Months as a customer", 0, 120, 5, key="tenure")
    monthly = st.number_input("Monthly bill", 0.0, 500.0, 89.9, 0.5, key="monthly")
    total = st.number_input("Total paid so far (estimated from the two above)", 0.0, 100000.0,
                            float(round(tenure * monthly, 2)), 10.0)
    payment = st.selectbox("Payment method", PAYMENTS, key="payment")
    paperless = st.checkbox("Paperless billing", True, key="paperless")
with middle:
    senior = st.checkbox("Senior citizen", False, key="senior")
    partner = st.checkbox("Has a partner", False, key="partner")
    dependents = st.checkbox("Has dependents", False, key="dependents")
    phone = st.checkbox("Phone service", True, key="phone")
    if phone:
        multiple = st.selectbox("Multiple phone lines", ["No", "Yes"], key="multiple")
    else:
        multiple = "No phone service"
        st.caption("No phone service: multiple lines does not apply.")
with right:
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"], key="internet")
    add_ons = {}
    if internet == "No":
        add_ons = {field: "No internet service" for field, _ in ADD_ONS}
        st.caption("No internet service: the add-ons do not apply.")
    else:
        for field, label in ADD_ONS:
            add_ons[field] = st.selectbox(label, ["No", "Yes"], key=field)

customer = {
    "senior_citizen": senior, "partner": partner, "dependents": dependents, "tenure": int(tenure),
    "phone_service": phone, "multiple_lines": multiple, "internet_service": internet, **add_ons,
    "contract": contract, "paperless_billing": paperless, "payment_method": payment,
    "monthly_charges": float(monthly), "total_charges": float(total),
}

# ---------------- score ----------------
if st.button("Score this customer", type="primary"):
    try:
        with st.spinner("Scoring (the first request after a quiet period can take a few seconds)..."):
            st.session_state["result"] = {
                "prediction": predict(customer, {"r": r, "d": d, "months": months}),
                "explanation": explain(customer),
                "months": months,
            }
    except ApiError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    p, e = result["prediction"], result["explanation"]
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Chance of leaving", f"{p['probability']:.1%}")
    c2.metric("Risk band", p["band"], help=p["band_description"])
    c3.metric(f"Expected gain of an offer, {result['months']} months", f"{p['expected_gain']:,.0f}",
              help="In the same currency units as the bill. Negative means the offer is expected to lose money.")

    if p["flag"]:
        st.success(f"Worth an offer: their {p['probability']:.1%} chance of leaving is above the "
                   f"{p['cutoff']:.1%} break-even.")
    else:
        st.info(f"Not worth an offer: their {p['probability']:.1%} chance of leaving is below the "
                f"{p['cutoff']:.1%} break-even, so an offer would cost more than it saves on average.")
    for warning in p["warnings"]:
        st.warning(warning)

    st.subheader("Why the model says this")
    for driver in e["drivers"]:
        arrow = "⬆" if driver["direction"] == "raises" else "⬇"
        line = f"{arrow} {driver['sentence']}"
        if driver["lever"]:
            line += f"  \n&nbsp;&nbsp;&nbsp;*Idea ({driver['lever']['status']}): {driver['lever']['text']}*"
        st.markdown(line)

    items = sorted(e["waterfall"], key=lambda w: abs(w["effect"]), reverse=True)[:10][::-1]
    if items:
        fig = go.Figure(go.Bar(
            x=[w["effect"] for w in items], y=[f"{w['label']}: {w['value']}" for w in items],
            orientation="h", marker_color=[RAISES if w["effect"] > 0 else LOWERS for w in items]))
        fig.update_layout(height=60 + 38 * len(items), margin={"l": 0, "r": 10, "t": 10, "b": 0},
                          xaxis_title="Lowers the risk (left)  |  raises the risk (right)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(e["note"])

st.divider()
st.caption("Demo built on the public IBM Telco customer churn dataset. The probability is calibrated; the "
           "money figures depend entirely on the sidebar assumptions. Retention ideas are untested hypotheses.")