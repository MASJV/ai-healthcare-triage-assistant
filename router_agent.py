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
    symptom   - a person is describing their own or someone else's symptoms,
                or asking what they might mean. This includes emotional
                distress, hopelessness and sleep or mood problems.
    self_care - a general question about human health: home care, first aid
                for minor problems, diet, prevention, or how an illness works
                (e.g. "how is a migraine different from a headache?")
    booking   - user wants to book, reschedule, cancel or check a doctor's
                appointment
    other     - anything that is not about a person's health or a doctor's
                appointment: animals, plants, devices, vehicles, travel or
                ticket bookings, subscriptions, greetings, jokes, general
                knowledge

    Rules, in order of priority:
    1. If the message describes a symptom, the label is symptom, even if the
       user also asks to see a doctor or book an appointment
       (e.g. "my knee has been swollen for days, I need a doctor" -> symptom).
    2. If the assistant just asked a follow-up question about symptoms and the
       user is answering it, the label is symptom.
    3. If the assistant just asked about an appointment (name, phone, slot)
       and the user is answering it, the label is booking.
    4. Words like "book", "cancel" or "fix" only mean booking or self_care
       when they are about a doctor's appointment or a person's health.

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
