"""Streamlit demo: pick a traffic intervention, run user-equilibrium on Sioux Falls."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.network import load_network
from src.assignment import solve_user_equilibrium
from src.scenarios import Scenario, apply_scenario

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SCENARIOS = ROOT / "experiments" / "scenarios.json"


@st.cache_resource(show_spinner="Loading network and baseline equilibrium...")
def baseline():
    net = load_network(DATA / "SiouxFalls_net.tntp", DATA / "SiouxFalls_trips.tntp", "SiouxFalls")
    res = solve_user_equilibrium(net, max_iter=5000, tol=1e-5)
    return net, res.tstt


st.set_page_config(page_title="Traffic planning with LLMs", layout="wide")
st.title("Sioux Falls traffic planning: user-equilibrium solver")
st.caption(
    "Choose an intervention. The solver computes the new user equilibrium and reports the change in "
    "total system travel time (TSTT). Solving takes about 10–15 seconds."
)

net, base_tstt = baseline()
links = sorted({(l.init, l.term) for l in net.links})

with st.form("intervention"):
    op = st.selectbox("Intervention", ["close_link", "capacity", "toll", "demand"],
                      format_func=lambda o: {
                          "close_link": "Close a link",
                          "capacity": "Change a link's capacity",
                          "toll": "Add a toll to a link",
                          "demand": "Scale total demand",
                      }[o])
    col1, col2, col3 = st.columns(3)
    if op == "demand":
        factor = col1.number_input("Demand factor", min_value=0.5, max_value=1.5, value=1.1, step=0.05)
        params = {"factor": float(factor)}
    else:
        pair = col1.selectbox("Link (from → to)", links, format_func=lambda p: f"{p[0]} → {p[1]}")
        params = {"init": pair[0], "term": pair[1]}
        if op == "capacity":
            params["factor"] = float(col2.number_input("Capacity factor", 0.5, 3.0, 1.5, 0.5))
        elif op == "toll":
            params["amount"] = float(col2.number_input("Toll", 0.0, 50.0, 10.0, 5.0))
    submitted = st.form_submit_button("Run equilibrium")

if submitted:
    sc = Scenario("demo", "", op, params)
    with st.spinner("Solving user equilibrium..."):
        new_net, tolls = apply_scenario(net, sc)
        res = solve_user_equilibrium(new_net, max_iter=5000, tol=1e-5, extra_costs=tolls)
    change = 100.0 * (res.tstt - base_tstt) / base_tstt
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline TSTT", f"{base_tstt:,.0f}")
    m2.metric("Scenario TSTT", f"{res.tstt:,.0f}", f"{change:+.2f}%")
    m3.metric("Relative gap", f"{res.relative_gap:.1e}")
    st.caption("Tolls change routing but are not counted in TSTT.")

if SCENARIOS.exists():
    st.subheader("Benchmark scenarios with reference answers")
    rows = json.loads(SCENARIOS.read_text())
    df = pd.DataFrame([
        {"id": r["sid"], "request": r["text"], "TSTT change (%)": round(r["reference"]["tstt_change_pct"], 2)}
        for r in rows
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
