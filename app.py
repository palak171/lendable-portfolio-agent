import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agent import ask

load_dotenv()

st.set_page_config(page_title="Lendable Portfolio Agent", page_icon="\U0001F4CA", layout="wide")

st.title("Lendable Portfolio Agent")
st.caption(
    "Ask a question about the loan portfolio in plain English. "
    "The agent writes and runs SQL against the database to answer it."
)

def _server_side_key() -> str:
    """A key configured by the app owner (via .env locally, or Streamlit
    Cloud 'Secrets' when deployed). Never shown in the UI."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

_server_key = _server_side_key()

with st.sidebar:
    st.header("Setup")
    if _server_key:
        st.success("Groq API key configured by the app owner — you're ready to go.")
        api_key = _server_key
    else:
        api_key = st.text_input(
            "Groq API key", value="", type="password",
            help="Get a free key at console.groq.com. Used only for your session, never stored or shown to anyone else.",
        )
    st.markdown("---")
    st.subheader("Try one of the case-study questions")
    sample_questions = [
        "Is PAR going up?",
        "What is driving PAR up?",
        "Does PAR include the lateness from rescheduled loans?",
        "How much of PAR is driven by rescheduled loans?",
        "Do rescheduled loans perform differently than non-rescheduled loans?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.chat_input("Ask about the portfolio...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        _render_result = item["render"]
        _render_result()

def render_result(result):
    st.write(result.explanation)

    if result.error:
        st.warning(result.error)

    if result.dataframe is None or result.dataframe.empty:
        st.info("The agent's last query returned no rows.")
    else:
        df = result.dataframe
        if result.display == "chart" and result.chart_type and result.x:
            try:
                if result.chart_type == "line":
                    fig = px.line(df, x=result.x, y=result.y, color=result.series, markers=True, title=result.title)
                elif result.chart_type == "bar":
                    fig = px.bar(df, x=result.x, y=result.y, color=result.series, title=result.title)
                elif result.chart_type == "grouped_bar":
                    fig = px.bar(df, x=result.x, y=result.y, color=result.series, barmode="group", title=result.title)
                elif result.chart_type == "pie":
                    fig = px.pie(df, names=result.x, values=result.y, title=result.title)
                else:
                    fig = None
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not render chart ({e}); showing table instead.")
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    with st.expander(f"SQL the agent ran ({len(result.sql_trail)} quer{'y' if len(result.sql_trail)==1 else 'ies'})"):
        for i, step in enumerate(result.sql_trail, 1):
            st.code(step["query"], language="sql")
            st.caption(f"-> {step['row_count']} row(s)")

if question:
    if not api_key:
        st.error("Enter a Groq API key in the sidebar first (free at console.groq.com).")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking and running SQL..."):
                result = ask(question, api_key=api_key)
            render_result(result)
        st.session_state["history"].append(
            {"question": question, "render": lambda r=result: render_result(r)}
        )
