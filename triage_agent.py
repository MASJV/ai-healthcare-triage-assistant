import re

from langchain_core.messages import AIMessage

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
        levels = "emergency / urgent / routine / needs_more_info"
    else:
        follow_up_option = ""
        levels = "emergency / urgent / routine"

    # LLM reasoning for the ones which were not caught
    prompt = f"""
    Conversation so far:
    {format_history(state["messages"][:-1])}

    Latest User Message: {state['user_input']}

    You are a medical triage assistant. Using the whole conversation,
    decide how soon the user needs care.

    emergency - life-threatening, needs care right now
                (e.g. severe pain, breathing trouble, confusion, heavy bleeding)
    urgent    - should see a doctor within 24-48 hours
                (e.g. high fever for 3+ days, worsening pain, infection signs)
    routine   - can wait for a normal appointment
                (e.g. mild symptoms, follow-ups, check-ups)
    {follow_up_option}
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

    # unclear answer -> treared as urgent
    if urgency not in ["emergency", "urgent", "routine"]:
        urgency = "urgent"

    return {
        "urgency": urgency,
        "triage_reason": reason,
        "follow_ups": 0
    }


def triage_route(state: UserState):

    if state["urgency"] == "emergency":
        return "Escalation Agent"

    if state["urgency"] == "needs_more_info":
        return "Ask User"

    return "Scheduling Agent"
