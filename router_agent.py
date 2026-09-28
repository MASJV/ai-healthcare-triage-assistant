from langchain_core.messages import AIMessage

from state import UserState, format_history
from config import llm
from triage_agent import check_red_flags


OTHER_REPLY = (
    "I can help with health symptoms, self-care questions and doctor appointments. "
    "Please tell me about a symptom or what you need."
)


#Router Agent
def router_agent(state: UserState):

    # every message except the one being classified now
    history = format_history(state["messages"][:-1])

    prompt = f"""
    Conversation so far:
    {history}

    Latest User Message: {state['user_input']}

    Classify the intent of the latest user message into exactly one of these labels:
    symptom   - user is describing symptoms or asking what they might mean
    self_care - user is asking for home care or self-care advice
    booking   - user wants to book, reschedule or cancel an appointment
    other     - not related to health or appointments

    If the assistant just asked a follow-up question about symptoms and the
    user is answering it, the label is symptom.
    If the assistant just asked about an appointment (name, slot) and the
    user is answering it, the label is booking.
    If the message describes a symptom, the label is symptom, even if the
    user also asks for an appointment.

    Reply with only the label, nothing else.
    """

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent not in ["symptom", "self_care", "booking", "other"]:
        intent = "symptom"

    # red-flag symptoms always go through triage, whatever label the LLM picked
    if check_red_flags(state["user_input"]) is not None:
        intent = "symptom"

    if intent == "other":
        return {
            "intent": intent,
            "result": OTHER_REPLY,
            "messages": [AIMessage(content=OTHER_REPLY)]
        }

    return {
        "intent": intent
    }


def route(state: UserState):

    if state["intent"] == "booking":
        return "Scheduling Agent"

    if state["intent"] == "symptom":
        return "Triage Agent"

    if state["intent"] == "other":
        return "End"

    return "Knowledge Agent"
