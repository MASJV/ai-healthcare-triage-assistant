import time
import uuid
import base64

import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Healthcare Triage Assistant")

# imported once per server process; on a new server the first import also
# builds the health knowledge base (see ingest.ensure_knowledge_base)
try:
    with st.spinner("Starting the assistant..."):
        from symptomTriage import app
except Exception as error:
    if type(error).__name__ == "KnowledgeBaseError":
        st.error(str(error))  # written in ingest.py to be safe to show
    else:
        print(f"App failed to start: {type(error).__name__}", flush=True)
        st.error(
            "The assistant could not start. Check that OPENAI_API_KEY is set "
            "(in .env locally, or in the app's Secrets on Streamlit Community Cloud) and see the app logs."
        )
    st.stop()


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
            border-right: 2px solid rgba(30, 136, 229, 0.5) !important;
            box-shadow: 5px 0px 15px rgba(0, 0, 0, 0.5);

            /* Subtle dark tint overlay so text remains readable */
            background-image: linear-gradient(
                rgba(18, 23, 42, 0.75),
                rgba(18, 23, 42, 0.75)
            ) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        [data-testid="stSidebarHeader"] {{
            background-color: transparent !important;
        }}

        .stApp h1, .stApp h2, .stApp h3 {{
            border-bottom: 3px solid #1E88E5;
            padding-bottom: 0.35rem;
        }}

        .stButton > button, .stFormSubmitButton > button {{
            background-color: #1E88E5;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background-color: #1565C0;
            color: #ffffff;
        }}

        [data-testid="stJson"] {{
            background-color: #12172A;
            border: 1px solid #1E88E5;
            border-radius: 10px;
            padding: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_seamless_bg_with_sidebar_border("assets/triage_bg.png")

st.markdown(
    "<h1 style='text-align: center;'>🩺 AI Healthcare Triage Assistant</h1>",
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


# what each agent did, shown while the message is being processed
STEP_TEXT = {
    "Router Agent": "Understood the request",
    "Triage Agent": "Checked how urgent it is",
    "Knowledge Agent": "Searched the health documents",
    "Scheduling Agent": "Checked the appointment slots",
    "Escalation Agent": "Prepared emergency guidance"
}


def show_conversation():
    for item in st.session_state.conversation:
        st.caption(item["role"])
        st.markdown(item["text"])


with st.sidebar:
    st.markdown(
        "<h1 style='text-align: center;'>💬 Conversation</h1>",
        unsafe_allow_html=True,
    )

    if st.button("New conversation", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation = []
        st.session_state.details = {}
        st.session_state.reply = ""
        st.rerun()

    # filled again after each reply, so the new message shows up at once
    history = st.empty()
    with history.container():
        show_conversation()


with st.form("message_form", clear_on_submit=True):
    user_input = st.text_input("Describe your symptom or what you need")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # the user's message appears in the sidebar before the agents run
    st.session_state.conversation.append({"role": "You", "text": user_input})
    with history.container():
        show_conversation()

    st.markdown(f"**You:** {user_input}")

    with st.status("🩺 Working on your message...", expanded=True) as status:
        # run_name labels each turn in LangSmith when tracing is on
        session = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "run_name": "symptom-triage-scheduler"
        }

        start = time.perf_counter()

        try:
            # per-turn fields are reset; messages and follow_ups carry over.
            # stream() gives the output of each agent as soon as it finishes
            for update in app.stream(
                {
                    "user_input": user_input,
                    "messages": [HumanMessage(content=user_input)],
                    "intent": "",
                    "urgency": "",
                    "triage_reason": "",
                    "result": ""
                },
                session,
                stream_mode="updates"
            ):
                for agent, output in update.items():
                    text = STEP_TEXT.get(agent, agent)
                    if agent == "Router Agent" and output:
                        text += f": {output.get('intent', '')}"
                    if agent == "Triage Agent" and output:
                        text += f": {output.get('urgency', '')}"
                    st.write(f"✔️ {text}")
        except Exception as error:
            # log only the error type: API errors can include parts of the request
            print(f"Message failed: {type(error).__name__}", flush=True)
            status.update(label="Something went wrong", state="error", expanded=False)
            st.error(
                "Sorry, something went wrong while handling that message. Please try again. "
                "In an emergency, call 112 (emergency) or 108 (ambulance)."
            )
            st.stop()

        latency = time.perf_counter() - start
        result = app.get_state(session).values

        status.update(
            label=f"✅ Done in {latency:.1f} s",
            state="complete",
            expanded=False
        )

    st.session_state.conversation.append({"role": "Assistant", "text": result["result"]})
    with history.container():
        show_conversation()

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
