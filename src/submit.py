"""Streamlit entry point: python -m streamlit run src/submit.py."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import streamlit as st

from coursekit import MissingKey
from data.IUB_policy_corpus import IUB_Policy
from src.navigator import answer, memory_answer, retrieve

logger = logging.getLogger(__name__)

st.set_page_config(page_title="IUB Policy Navigator", page_icon="🎓")
st.title("IUB Policy Navigator")
st.caption("Understand the IU policies that may affect your student situation.")
st.info(
    "Answers use a local IU policy snapshot, not your student records. Coverage includes university-wide policies and may omit campus deadlines or aid details."
)
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.header("Your conversation")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Memory lasts for this browser session. Clearing it removes the conversation.")
    st.write(f"{len(IUB_Policy):,} policy passages")
    with st.expander("How answers are checked"):
        st.write(
            "Three prompting techniques: student-friendly persona, few-shot examples, and structured JSON output. Policy responses use draft → critique → revision. Evidence quotes and source IDs are checked in code."
        )


def display(message: dict) -> None:
    """Render saved turns and their original evidence across Streamlit reruns."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved policy passages"):
                for i, passage in enumerate(message["sources"], 1):
                    st.markdown(f"**[S{i}]**")
                    st.text(passage)
        if message.get("check"):
            st.caption(message["check"])


for message in st.session_state.messages:
    display(message)
if question := st.chat_input("Ask about a policy or describe your situation", max_chars=4000):
    question = question.strip()
    if not question:
        st.stop()
    history = list(st.session_state.messages)
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    display(user_message)
    sources = []
    check = "Answered from conversation memory."
    try:
        reply = memory_answer(question, history)
        if reply is None:
            with st.spinner("Finding policies and checking the answer…"):
                sources = retrieve(question, history, IUB_Policy)
                reply = answer(question, sources, history)
                check = "Policy response checked using the reflection and evidence pipeline."
    except MissingKey:
        reply = "Set NRP_API_KEY in your environment or project .env file, then try again."
        check = "No policy answer was generated."
    except Exception as exc:
        # Do not log exception bodies: SDK errors can contain request data.
        logger.warning("Policy request failed (%s)", type(exc).__name__)
        reply = "The policy service could not complete this request. Please try again."
        check = "Service error; this does not mean the corpus lacks an answer."
    message = {"role": "assistant", "content": reply, "sources": sources, "check": check}
    st.session_state.messages.append(message)
    display(message)
