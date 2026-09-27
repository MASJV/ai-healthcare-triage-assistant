import re

from state import UserState
from config import llm

EMERGENCY_KEYWORDS = [
    "chest pain", "pain in my chest", "heart attack",
    "can't breathe", "cannot breathe", "not breathing", "difficulty breathing",
    "shortness of breath", "choking",
    "unconscious", "fainted", "passed out", "not waking up",
    "seizure", "fits", "convulsion",
    "stroke", "face drooping", "slurred speech", "one side of the body",
    "severe bleeding", "bleeding heavily", "won't stop bleeding",
    "coughing blood", "vomiting blood",
    "severe burn", "head injury", "snake bite",
    "poisoning", "overdose", "swallowed",
    "throat swelling", "tongue swelling",
    "suicide", "kill myself", "end my life"
]


#Rule-Based Filter
def check_red_flags(user_input):

    message = user_input.lower()

    for keyword in EMERGENCY_KEYWORDS:
        pattern = r"\b" + re.escape(keyword)
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
            "triage_reason": f"Red-flag symptom detected: '{red_flag}'"
        }

    # LLM reasoning for the ones which were not caught
    prompt = f"""
    User Message: {state['user_input']}

    You are a medical triage assistant. Decide how soon the user needs care.

    emergency - life-threatening, needs care right now
                (e.g. severe pain, breathing trouble, confusion, heavy bleeding)
    urgent    - should see a doctor within 24-48 hours
                (e.g. high fever for 3+ days, worsening pain, infection signs)
    routine   - can wait for a normal appointment
                (e.g. mild symptoms, follow-ups, check-ups)

    If you are unsure between two levels, choose the more serious one.

    Reply in exactly this format:
    Level: <emergency / urgent / routine>
    Reason: <one short sentence>
    """

    response = llm.invoke(prompt)

    urgency = ""
    reason = ""

    for line in response.content.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("level:"):
            urgency = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    # unclear answer -> treared as urgent
    if urgency not in ["emergency", "urgent", "routine"]:
        urgency = "urgent"

    return {
        "urgency": urgency,
        "triage_reason": reason
    }


def triage_route(state: UserState):

    if state["urgency"] == "emergency":
        return "Escalation Agent"

    return "Scheduling Agent"
