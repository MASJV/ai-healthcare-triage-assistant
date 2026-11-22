from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


#Shared State
class UserState(TypedDict):
    messages: Annotated[list, add_messages]   
    user_input: str
    intent: str
    router_probabilities: dict   # Jev's probability for each label (empty without Jev)
    urgency: str
    triage_reason: str
    triage_probabilities: dict   # Jev's probability for each level (empty without Jev)
    symptom_text: str      # the user's messages about the problem triage last judged
    follow_ups: int  
    booking_urgency: str   # urgency kept until the booking is done
    result: str


#Conversation History
def format_history(messages):
    """
    Turns the saved messages into plain text for a prompt:
        User: ...
        Assistant: ...
    """
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        else:
            lines.append(f"Assistant: {message.content}")

    if not lines:
        return "(no earlier messages)"

    return "\n".join(lines)
