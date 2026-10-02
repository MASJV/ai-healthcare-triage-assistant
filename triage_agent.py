import re

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable

from state import UserState, format_history
from config import llm, MAX_FOLLOW_UPS

EMERGENCY_KEYWORDS = [
    "chest pain", "pain in my chest", "heart attack",
    "can't breathe", "cannot breathe", "not breathing", "difficulty breathing",
    "shortness of breath", "choking",
    "unconscious", "fainted", "passed out", "knocked out", "not waking up",
    "seizure", "fits", "convulsion",
    "stroke", "face drooping", "slurred speech", "one side of the body",
    "severe bleeding", "heavy bleeding", "bleeding heavily", "won't stop bleeding",
    "coughing blood", "vomiting blood",
    "severe burn", "head injury", "snake bite",
    "poisoning", "overdose", "swallowed",
    "throat swelling", "tongue swelling",
    "suicide", "kill myself", "end my life",
    "want to die", "don't want to live", "dont want to live",
    "no reason to live", "no point in going on"
]


#Rule-Based Filter
@traceable(name="checkRedFlags")
def check_red_flags(user_input):

    message = user_input.lower()

    for keyword in EMERGENCY_KEYWORDS:
        pattern = rf"\b{re.escape(keyword)}"
        if re.search(pattern, message):
            return keyword

    return None


#Triage Agent
def triage_agent(state: UserState):

    # rule-based filter
    red_flag = check_red_flags(state["user_input"])

    if red_flag is not None:
        return {
            "urgency": "emergency",
            "triage_reason": f"Red-flag symptom detected: '{red_flag}'",
            "follow_ups": 0
        }

    follow_ups = state.get("follow_ups", 0)

    if follow_ups < MAX_FOLLOW_UPS:
        follow_up_option = """
    needs_more_info - too vague to judge safely; ask one follow-up question
                      about the red flags for this symptom
                      (e.g. "he hit his head" -> did he pass out, vomit,
                      or seem confused?)
    """
        levels = "emergency / urgent / routine / self_care / needs_more_info"
    else:
        follow_up_option = ""
        levels = "emergency / urgent / routine / self_care"

    # LLM reasoning for the ones which were not caught
    prompt = f"""
    Conversation so far:
    {format_history(state["messages"][:-1])}

    Latest User Message: {state['user_input']}

    You are a medical triage assistant. Using the whole conversation,
    decide how soon the user needs care.

    emergency - needs care right now: life-threatening, or time-critical
                even without severe pain
                (e.g. severe pain, breathing trouble, confusion, heavy
                bleeding, sudden change in vision, new numbness or weakness,
                a child who is unusually drowsy or hard to wake, bleeding or
                strong pain in pregnancy, a new or different symptom in
                someone with a serious past heart or brain condition)
    urgent    - should see a doctor within 24-48 hours
                (e.g. high fever for 3+ days, worsening pain, infection signs)
    routine   - should see a doctor, but it can wait for a normal appointment
                (e.g. the user asks to see a doctor, medicine refills,
                check-ups, a problem that has lasted weeks or keeps coming
                back, something a doctor needs to examine)
    self_care - mild, recent or familiar problem with no warning signs that
                can be managed at home and watched; see a doctor only if it
                gets worse or does not improve
                (e.g. a small cut, a minor bruise, a mild sunburn)
    {follow_up_option}
    If the user asks to see a doctor, do not choose self_care.
    If you are unsure between two levels, choose the more serious one.

    Reply in exactly this format:
    Level: <{levels}>
    Reason: <one short sentence>
    Question: <the follow-up question, only for needs_more_info>
    """

    response = llm.invoke(prompt)

    urgency = ""
    reason = ""
    question = ""

    for line in response.content.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("level:"):
            urgency = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()

    if urgency == "needs_more_info" and question and follow_ups < MAX_FOLLOW_UPS:
        return {
            "urgency": urgency,
            "triage_reason": reason,
            "follow_ups": follow_ups + 1,
            "result": question,
            "messages": [AIMessage(content=question)]
        }

    # unclear answer -> treated as urgent
    if urgency not in ["emergency", "urgent", "routine", "self_care"]:
        urgency = "urgent"

    # the user's messages for this problem: the first one plus the answers
    # to each follow-up question, so the knowledge agent can search with them
    user_messages = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            user_messages.append(message.content)
    case_size = follow_ups + 1
    symptom_text = " ".join(user_messages[-case_size:])

    return {
        "urgency": urgency,
        "triage_reason": reason,
        "follow_ups": 0,
        "symptom_text": symptom_text
    }


def triage_route(state: UserState):

    if state["urgency"] == "emergency":
        return "Escalation Agent"

    if state["urgency"] == "needs_more_info":
        return "Ask User"

    if state["urgency"] == "self_care":
        return "Knowledge Agent"

    return "Scheduling Agent"
