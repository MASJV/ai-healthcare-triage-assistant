import re
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
from state import UserState
from config import llm

CALL_FOR_HELP = (
    "This may be a medical emergency.\n"
    "- Call 108 for an ambulance or 112 for emergency help right now.\n"
    "- If a hospital is very close and the person can be moved safely, "
    "take them straight to the emergency department."
)

FIRST_AID = [
    {
        "category": "bleeding",
        "description": "heavy or uncontrolled bleeding, losing a lot of blood",
        "keywords": ["bleeding", "wound"],
        "steps": [
            "Press firmly on the wound with a clean cloth and keep pressing.",
            "If blood soaks through, put another cloth on top. Do not remove the first one.",
            "If the wound is on an arm or leg, raise it above the heart if possible.",
            "Keep the person lying down and warm."
        ]
    },
    {
        "category": "chest_pain",
        "description": "chest pain, pressure or tightness, possible heart attack",
        "keywords": ["chest pain", "pain in my chest", "heart attack"],
        "steps": [
            "Stop all activity. Help the person sit down and rest.",
            "Loosen tight clothing around the neck and chest.",
            "Do not give food or drink.",
            "Stay with them and watch their breathing."
        ]
    },
    {
        "category": "choking",
        "description": "something stuck in the throat, choking on food",
        "keywords": ["choking"],
        "steps": [
            "Encourage them to keep coughing.",
            "If they cannot cough, speak or breathe, give firm blows between the shoulder blades with the heel of your hand."
        ]
    },
    {
        "category": "breathing",
        "description": "struggling to breathe, gasping, severe allergic swelling of throat or tongue",
        "keywords": ["can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath",
                     "throat swelling", "tongue swelling"],
        "steps": [
            "Help them sit upright. Do not make them lie flat.",
            "Loosen tight clothing.",
            "If they have their own inhaler or adrenaline pen, help them use it."
        ]
    },
    {
        "category": "unconscious",
        "description": "unconscious, fainted, unresponsive, not breathing",
        "keywords": ["unconscious", "fainted", "passed out", "knocked out", "not waking up", "not breathing"],
        "steps": [
            "Check if they are breathing.",
            "If breathing: turn them on their side so they do not choke.",
            "If not breathing: start pushing hard and fast in the centre of the chest if you know how, and follow the call handler's instructions."
        ]
    },
    {
        "category": "seizure",
        "description": "seizure, fits, uncontrolled shaking or jerking",
        "keywords": ["seizure", "fits", "convulsion"],
        "steps": [
            "Do not hold them down and do not put anything in their mouth.",
            "Move hard or sharp objects away and cushion their head.",
            "Note the time the seizure started.",
            "When the shaking stops, turn them on their side."
        ]
    },
    {
        "category": "stroke",
        "description": "stroke signs: face drooping, sudden weakness on one side, slurred speech",
        "keywords": ["stroke", "face drooping", "slurred speech", "one side of the body"],
        "steps": [
            "Note the time the symptoms started. Doctors will ask for it.",
            "Do not give food, drink or medicine.",
            "Keep them lying down with the head slightly raised."
        ]
    },
    {
        "category": "burn",
        "description": "burns from fire, hot liquid, chemicals or electricity",
        "keywords": ["burn"],
        "steps": [
            "Cool the burn under cool running water for 20 minutes.",
            "Remove rings or tight items near the burn, unless stuck to the skin.",
            "Do not put ice, butter or toothpaste on it.",
            "Cover loosely with a clean cloth or cling film."
        ]
    },
    {
        "category": "head_injury",
        "description": "blow or injury to the head",
        "keywords": ["head injury", "hit his head", "hit her head", "hit my head", "hit their head"],
        "steps": [
            "Keep the person still. If they fell, do not move their neck.",
            "Hold a cold pack wrapped in cloth on the injury.",
            "Watch for vomiting, drowsiness or confusion and tell the doctors."
        ]
    },
    {
        "category": "poisoning",
        "description": "poisoning, overdose, swallowed something harmful",
        "keywords": ["poisoning", "overdose", "swallowed"],
        "steps": [
            "Do not make them vomit.",
            "Keep the packet, bottle or container and show it to the doctors."
        ]
    },
    {
        "category": "snake_bite",
        "description": "snake bite",
        "keywords": ["snake bite"],
        "steps": [
            "Keep the person calm and as still as possible.",
            "Keep the bitten arm or leg still. Remove rings, watches or tight items near the bite.",
            "Do not cut the bite, suck it or tie it tightly."
        ]
    },
    {
        "category": "suicide",
        "description": "suicidal thoughts, wanting to die or not wanting to live, self-harm",
        "keywords": [
            "suicide", "kill myself", "end my life",
            "want to die", "don't want to live", "dont want to live",
            "no reason to live", "no point in going on"
        ],
        "steps": [
            "Do not leave the person alone.",
            "Remove anything they could use to harm themselves, if it is safe to do so.",
            "You can also call Tele-MANAS on 14416 to talk to a counsellor."
        ]
    }
]

GENERAL_STEPS = [
    "Stay with the person and keep them calm.",
    "Do not give food, drink or medicine unless a doctor tells you to.",
    "Keep your phone free for the ambulance team."
]

CLOSING = "This assistant cannot replace emergency services. Please get help now."


@traceable(name="findCategoriesByKeywords")
def find_categories_by_keywords(text):
    """
    Returns the category of every emergency type whose keywords
    appear in the text. Several categories can match.
    """
    categories = []

    for emergency in FIRST_AID:
        for keyword in emergency["keywords"]:
            pattern = rf"\b{re.escape(keyword)}"
            if re.search(pattern, text):
                categories.append(emergency["category"])
                break

    return categories


@traceable(name="findCategoriesByLlm")
def find_categories_by_llm(text):
    """
    Asks the LLM only to pick category names from the fixed list, so
    phrases like "losing a lot of blood" map to "bleeding".
    The LLM never writes advice; unknown names are ignored.
    """
    category_lines = []
    valid_categories = []
    for emergency in FIRST_AID:
        category_lines.append(f"{emergency['category']} - {emergency['description']}")
        valid_categories.append(emergency["category"])

    # one category per line
    categories_text = "\n".join(category_lines)

    prompt = f"""
    User messages: {text}

    Which of these emergency categories does the situation describe?

    {categories_text}

    Reply with the matching category names separated by commas,
    or "none" if nothing matches. Reply with names only.
    """

    try:
        response = llm.invoke(prompt)
    except Exception:
        # if the LLM call fails, fallS back to keyword matching
        return []

    categories = []
    for name in response.content.lower().split(","):
        name = name.strip()
        if name in valid_categories and name not in categories:
            categories.append(name)

    return categories


@traceable(name="findFirstAidSteps")
def find_first_aid_steps(categories):
    """
    Returns the fixed steps from FIRST_AID for the given categories,
    in FIRST_AID order.
    """
    matched_steps = []

    for emergency in FIRST_AID:
        if emergency["category"] in categories:
            matched_steps.extend(emergency["steps"])

    return matched_steps


def escalation_agent(state: UserState):

    user_text = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            user_text = f"{user_text} {message.content.lower()}"

    # keywords catch exact phrases, the LLM catches other wordings;
    # both only choose categories, the steps always come from FIRST_AID
    categories = find_categories_by_keywords(user_text)
    for category in find_categories_by_llm(user_text):
        if category not in categories:
            categories.append(category)

    steps = find_first_aid_steps(categories)
    if not steps:
        steps = GENERAL_STEPS

    step_lines = []
    for number, step in enumerate(steps, start=1):
        step_lines.append(f"{number}. {step}")

    steps_text = "\n".join(step_lines)
    response = f"{CALL_FOR_HELP}\n\nWhile waiting for help:\n{steps_text}\n\n{CLOSING}"

    return {
        "result": response,
        "messages": [AIMessage(content=response)]
    }
