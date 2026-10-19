import time
import uuid
import base64

import streamlit as st
from langchain_core.messages import HumanMessage

from symptomTriage import app

st.set_page_config(page_title="Symptom Triage & Appointment Scheduler")


def set_seamless_bg_with_sidebar_border(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read())
    st.markdown(
        f"""
        <style>
        /* Apply background to the entire application wrapper */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_string.decode()}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Make sidebar transparent so the image flows through */
        [data-testid="stSidebar"] {{
            background-color: transparent !important;
            background-image: none !important;

            /* EXPLICIT SIDEBAR BORDERS & SEPARATION */
            border-right: 2px solid rgba(255, 90, 69, 0.4) !important;
            box-shadow: 5px 0px 15px rgba(0, 0, 0, 0.5);

            /* Subtle dark tint overlay so text remains readable */
            background-image: linear-gradient(
                rgba(18, 23, 42, 0.75),
                rgba(18, 23, 42, 0.75)
            ) !important;
        }}

        [data-testid="stSidebarHeader"] {{
            background-color: transparent !important;
        }}

        .stApp h1, .stApp h2, .stApp h3 {{
            border-bottom: 3px solid #FF5A45;
            padding-bottom: 0.35rem;
        }}

        .stButton > button, .stFormSubmitButton > button {{
            background-color: #FF5A45;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background-color: #e14a36;
            color: #ffffff;
        }}

        [data-testid="stJson"] {{
            background-color: #12172A;
            border: 1px solid #FF5A45;
            border-radius: 10px;
            padding: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_seamless_bg_with_sidebar_border("assets/triage_bg.png")

st.markdown(
    "<h1 style='text-align: center;'>🩺 Symptom Triage & Appointment Scheduler</h1>",
    unsafe_allow_html=True,
)

st.caption(
    "General health information and appointment support only. Not a diagnosis. "
    "In an emergency, call 112 (emergency) or 108 (ambulance)."
)

# one thread_id per browser session, so the graph remembers this conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "details" not in st.session_state:
    st.session_state.details = {}

if "reply" not in st.session_state:
    st.session_state.reply = ""


with st.form("message_form", clear_on_submit=True):
    user_input = st.text_input("Describe your symptom or what you need")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("🩺 Reading your message..."):
        # run_name labels each turn in LangSmith when tracing is on
        session = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "run_name": "symptom-triage-scheduler"
        }

        start = time.perf_counter()

        # per-turn fields are reset; messages and follow_ups carry over
        result = app.invoke(
            {
                "user_input": user_input,
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "urgency": "",
                "triage_reason": "",
                "result": ""
            },
            session
        )
        latency = time.perf_counter() - start

        st.session_state.conversation.append({"role": "You", "text": user_input})
        st.session_state.conversation.append({"role": "Assistant", "text": result["result"]})

        st.session_state.details = {
            "intent": result["intent"],
            "urgency": result["urgency"] or "-",
            "triage_reason": result["triage_reason"] or "-",
            "latency_seconds": round(latency, 2)
        }

        # Jev's probability for each router label (only when USE_JEV is on)
        if result.get("router_probabilities"):
            st.session_state.details["router_probabilities"] = result["router_probabilities"]
        if result.get("triage_probabilities"):
            st.session_state.details["triage_probabilities"] = result["triage_probabilities"]
        st.session_state.reply = result["result"]

    st.toast("✅ Message received.")


with st.sidebar:
    st.markdown(
        "<h1 style='text-align: center;'>💬 Conversation</h1>",
        unsafe_allow_html=True,
    )

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation = []
        st.session_state.details = {}
        st.session_state.reply = ""
        st.rerun()

    for item in st.session_state.conversation:
        st.caption(item["role"])
        st.markdown(item["text"])


if st.session_state.reply:
    st.header("💬 Response")

    urgency = st.session_state.details["urgency"]
    if urgency == "emergency":
        st.error(st.session_state.reply)
    elif urgency == "needs_more_info":
        st.warning(st.session_state.reply)
    else:
        st.markdown(st.session_state.reply)

    st.header("🧭 Triage Details")
    st.json(st.session_state.details)
