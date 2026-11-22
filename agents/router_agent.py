from langchain_core.messages import AIMessage, HumanMessage

from state import UserState, format_history
from config import llm, USE_JEV, SYMPTOM_THRESHOLD
from agents.triage_agent import check_red_flags
from jev import jev_choice, conversation_list


OTHER_REPLY = (
    "I can help with health symptoms, general health questions and doctor appointments. "
    "Please tell me about a symptom or what you need."
)


# the same four labels and meanings are used by both classifiers,
# so a run with Jev and a run with the LLM differ only in who decides
LABELS = {
    "symptom": (
        "A current symptom, injury or health problem of a specific person "
        "(the user or someone else), or a new worry about one, including "
        "emotional distress, hopelessness and sleep or mood problems. Still "
        "symptom if the user also asks for home care, a doctor or an "
        "appointment, or is answering the assistant's follow-up question "
        "about symptoms."
    ),
    "health_info": (
        "A general health question with no current health problem of a "
        "specific person: prevention, diet, general wellness, how an illness "
        "works, or how a problem is usually treated. A question about a "
        "health answer the assistant just gave. A diet or lifestyle question "
        "about a condition a doctor already diagnosed, with no new symptom."
    ),
    "booking": (
        "Book, reschedule, cancel or check a doctor's appointment, with no "
        "new health problem to assess, or answering the assistant's question "
        "about name, phone or slot."
    ),
    "other": (
        "Anything not about a person's health or a doctor's appointment: "
        "animals, plants, devices, vehicles, travel or ticket bookings, "
        "subscriptions, greetings, jokes, general knowledge."
    )
}


# the routing rules from the LLM prompt, given to Jev as its instructions
JEV_INSTRUCTIONS = (
    "Which workflow does `latest_user_message` need? Read it together with "
    "`conversation_so_far`. A current health problem is symptom even when "
    "the user also asks for home care, a doctor or an appointment; whether "
    "home care is enough is decided later, not here. If "
    "`assistant_last_message` asked a follow-up question about symptoms and "
    "the user answers it, it is symptom. If it asked for a name, phone "
    "number or slot and the user answers it, it is booking. \"Book\" or "
    "\"cancel\" mean booking only when they are about a doctor's "
    "appointment, and \"fix\" means health_info only when it is about a "
    "person's health; otherwise (flights, subscriptions, devices) it is other."
)


def last_assistant_message(messages):
    """
    The assistant's most recent reply, or "" at the start of a chat.
    """
    last = ""
    for message in messages:
        if not isinstance(message, HumanMessage):
            last = message.content
    return last


def classify_with_jev(messages, user_input):
    """
    Jev picks one label and returns a probability for each.
    If symptom is not the top label but still has SYMPTOM_THRESHOLD or more,
    the message goes to triage anyway.
    If the API call fails, None is returned and the LLM decides instead.
    """
    try:
        intent, probabilities = jev_choice(
            state={
                "conversation_so_far": conversation_list(messages),
                "assistant_last_message": last_assistant_message(messages),
                "latest_user_message": user_input
            },
            instructions=JEV_INSTRUCTIONS,
            criteria=LABELS
        )
    except Exception as error:
        print(f"Jev call failed, the LLM decides instead: {error}")
        return None

    if intent not in LABELS:
        intent = "symptom"

    if probabilities.get("symptom", 0) >= SYMPTOM_THRESHOLD:
        intent = "symptom"

    return intent, probabilities


def is_declined_booking(state):
    """
    An appointment was offered and not booked yet (booking_urgency is set).
    True if the user turns it down or says they will manage at home, with no
    new or changed symptom. Such a reply is not a new problem for triage.
    """
    prompt = f"""
    The assistant offered the user a doctor's appointment.

    Assistant's last message: {last_assistant_message(state["messages"][:-1])}

    Latest User Message: {state['user_input']}

    Reply decline if the user turns down the appointment or says they will
    manage at home, rest or not now, and mentions no new or changed symptom.
    Otherwise reply other.

    Reply with only the word, nothing else.
    """

    response = llm.invoke(prompt)
    return response.content.strip().lower() == "decline"


def classify_with_llm(history, user_input):

    prompt = f"""
    Conversation so far:
    {history}

    Latest User Message: {user_input}

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

    # the LLM gives no probabilities
    return intent, {}


#Router Agent
def router_agent(state: UserState):

    # an offered appointment is waiting: "no thanks" or "I'll treat it at home"
    # answers that offer, so it is not sent to triage as a new problem.
    # booking_urgency is kept, so a later "ok, book it" keeps the level
    if state.get("booking_urgency") and check_red_flags(state["user_input"]) is None:
        if is_declined_booking(state):
            return {
                "intent": "declined_booking",
                "router_probabilities": {}
            }

    # red-flag symptoms always go through triage, so no LLM call is needed.
    # With Jev, the router decides these too, so a prevention question
    # ("how do I lower my risk of a heart attack?") can go to health_info;
    # a real red-flag symptom still reaches triage, which checks the keywords
    if not USE_JEV and check_red_flags(state["user_input"]) is not None:
        return {
            "intent": "symptom",
            "router_probabilities": {}
        }

    # every message except the one being classified now
    history = format_history(state["messages"][:-1])

    jev_result = None
    if USE_JEV:
        jev_result = classify_with_jev(state["messages"][:-1], state["user_input"])

    if jev_result is not None:
        intent, probabilities = jev_result
    else:
        intent, probabilities = classify_with_llm(history, state["user_input"])

    if intent == "symptom":
        return {
            "intent": intent,
            "router_probabilities": probabilities
        }

    # the user left triage, so an unanswered follow-up question is dropped
    if intent == "other":
        return {
            "intent": intent,
            "router_probabilities": probabilities,
            "follow_ups": 0,
            "result": OTHER_REPLY,
            "messages": [AIMessage(content=OTHER_REPLY)]
        }

    return {
        "intent": intent,
        "router_probabilities": probabilities,
        "follow_ups": 0
    }


def route(state: UserState):

    if state["intent"] == "booking":
        return "Scheduling Agent"

    if state["intent"] == "symptom":
        return "Triage Agent"

    if state["intent"] == "other":
        return "End"

    return "Knowledge Agent"
