from state import UserState
from config import llm


#Router Agent
def router_agent(state: UserState):

    prompt = f"""
    User Message: {state['user_input']}

    Classify the intent of the user message into exactly one of these labels:
    symptom   - user is describing symptoms or asking what they might mean
    self_care - user is asking for home care or self-care advice
    booking   - user wants to book, reschedule or cancel an appointment

    Reply with only the label, nothing else.
    """

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent not in ["symptom", "self_care", "booking"]:
        intent = "symptom"

    return {
        "intent": intent
    }


#Conditional Function
def route(state: UserState):

    if state["intent"] == "booking":
        return "Scheduling Agent"

    return "Knowledge Agent"
