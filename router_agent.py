from langchain_core.messages import AIMessage

from state import UserState, format_history
from config import llm
from triage_agent import check_red_flags


OTHER_REPLY = (
    "I can help with health symptoms, general health questions and doctor appointments. "
    "Please tell me about a symptom or what you need."
)


#Router Agent
def router_agent(state: UserState):

    # red-flag symptoms always go through triage, so no LLM call is needed
    if check_red_flags(state["user_input"]) is not None:
        return {
            "intent": "symptom"
        }

    # every message except the one being classified now
    history = format_history(state["messages"][:-1])

    prompt = f"""
    Conversation so far:
    {history}

    Latest User Message: {state['user_input']}

    Decide which workflow the latest user message needs.
    Classify it into exactly one of these labels:
    symptom     - a current symptom, injury or health problem of a specific
                  person (the user or someone else), or a new worry about one.
                  This includes emotional distress, hopelessness and sleep or
                  mood problems. It stays symptom even if the user asks what
                  they can do at home, whether they can just rest, or whether
                  they need a doctor
                  (e.g. "I have a mild cold, what can I do at home?").
    health_info - a general health question with no current health problem
                  of a specific person: prevention, diet, general wellness,
                  how an illness works, or how a problem is usually treated
                  (e.g. "how can I prevent dehydration in summer?"). A question
                  about a health answer the assistant just gave is also
                  health_info.
    booking     - user wants to book, reschedule, cancel or check a doctor's
                  appointment, with no new health problem to assess
    other       - anything that is not about a person's health or a doctor's
                  appointment: animals, plants, devices, vehicles, travel or
                  ticket bookings, subscriptions, greetings, jokes, general
                  knowledge

    Rules, in order of priority:
    1. If the message describes a current health problem, the label is
       symptom, even if the user also asks for home care, a doctor or an
       appointment (e.g. "my knee has been swollen for days, I need a doctor"
       -> symptom). A decision on whether home care is enough is made later,
       not here.
    2. If the assistant just asked a follow-up question about symptoms and the
       user is answering it, the label is symptom.
    3. If the assistant just asked about an appointment (name, phone, slot)
       and the user is answering it, the label is booking.
    4. Words like "book", "cancel" or "fix" only mean booking or health_info
       when they are about a doctor's appointment or a person's health.

    Reply with only the label, nothing else.
    """

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent not in ["symptom", "health_info", "booking", "other"]:
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
