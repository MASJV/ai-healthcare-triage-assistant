import re

from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable

from state import UserState, format_history
from config import llm, MAX_FOLLOW_UPS, USE_JEV
from jev import jev_choice, conversation_list

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


# the same levels and meanings as the LLM prompt, given to Jev as criteria
LEVELS = {
    "emergency": (
        "Needs care right now: life-threatening, or time-critical even "
        "without severe pain, e.g. severe pain, breathing trouble, confusion, "
        "heavy bleeding, sudden change in vision, new numbness or weakness, "
        "a child who is unusually drowsy or hard to wake, bleeding or strong "
        "pain in pregnancy, a new or different symptom in someone with a "
        "serious past heart or brain condition."
    ),
    "urgent": (
        "Should see a doctor within 24-48 hours, e.g. high fever for 3+ days, "
        "worsening pain, infection signs."
    ),
    "routine": (
        "Should see a doctor, but it can wait for a normal appointment, e.g. "
        "the user asks to see a doctor, medicine refills, check-ups, a problem "
        "that has lasted weeks or keeps coming back, something a doctor needs "
        "to examine."
    ),
    "self_care": (
        "Mild, recent or familiar problem with no warning signs that can be "
        "managed at home and watched; see a doctor only if it gets worse or "
        "does not improve, e.g. a small cut, a minor bruise, a mild sunburn."
    )
}

NEEDS_MORE_INFO = (
    "Too vague to judge safely; one follow-up question about the red flags "
    "for this symptom is needed first, e.g. \"he hit his head\" (did he pass "
    "out, vomit, or seem confused?)."
)

JEV_INSTRUCTIONS = (
    "You are triaging a patient. Using `conversation_so_far` and "
    "`latest_user_message`, how soon does the user need care? If the user "
    "asks to see a doctor, do not choose self_care. If you are unsure "
    "between two levels, choose the more serious one."
)


@traceable(name="writeFollowUpQuestion")
def write_follow_up_question(state):
    """
    Jev does not write text, so the LLM writes the follow-up question
    when Jev decides more information is needed.
    """
    prompt = f"""
    Conversation so far:
    {format_history(state["messages"][:-1])}

    Latest User Message: {state['user_input']}

    The message is too vague to judge how soon the user needs care.
    Write one short follow-up question about the red flags for this symptom
    (e.g. "he hit his head" -> Did he pass out, vomit, or seem confused?).
    Reply with only the question.
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def triage_with_jev(state, follow_ups):
    """
    Jev picks the level (or needs_more_info) and returns a probability
    for each option.
    Returns (urgency, reason, question, probabilities). If the API call
    fails, None is returned and the LLM decides instead.
    """
    criteria = {}
    for level in LEVELS:
        criteria[level] = LEVELS[level]
    if follow_ups < MAX_FOLLOW_UPS:
        criteria["needs_more_info"] = NEEDS_MORE_INFO

    try:
        urgency, probabilities = jev_choice(
            state={
                "conversation_so_far": conversation_list(state["messages"][:-1]),
                "latest_user_message": state["user_input"]
            },
            instructions=JEV_INSTRUCTIONS,
            criteria=criteria
        )
    except Exception as error:
        print(f"Jev call failed, the LLM decides instead: {error}")
        return None

    reason = f"Jev: {urgency} ({probabilities.get(urgency, 0):.2f})"

    question = ""
    if urgency == "needs_more_info":
        question = write_follow_up_question(state)

    return urgency, reason, question, probabilities


def triage_with_llm(state, follow_ups):
    """
    The LLM picks the level and writes the reason and follow-up question.
    Returns (urgency, reason, question, probabilities); the LLM gives no
    probabilities.
    """
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

    return urgency, reason, question, {}


#Triage Agent
def triage_agent(state: UserState):

    # rule-based filter
    red_flag = check_red_flags(state["user_input"])

    if red_flag is not None:
        return {
            "urgency": "emergency",
            "triage_reason": f"Red-flag symptom detected: '{red_flag}'",
            "triage_probabilities": {},
            "follow_ups": 0
        }

    follow_ups = state.get("follow_ups", 0)

    jev_result = None
    if USE_JEV:
        jev_result = triage_with_jev(state, follow_ups)

    if jev_result is not None:
        urgency, reason, question, probabilities = jev_result
    else:
        urgency, reason, question, probabilities = triage_with_llm(state, follow_ups)

    if urgency == "needs_more_info" and question and follow_ups < MAX_FOLLOW_UPS:
        return {
            "urgency": urgency,
            "triage_reason": reason,
            "triage_probabilities": probabilities,
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
        "triage_probabilities": probabilities,
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
