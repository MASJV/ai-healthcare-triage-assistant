from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


#Shared State
class UserState(TypedDict):
    messages: Annotated[list, add_messages]   
    user_input: str
    intent: str
    urgency: str
    triage_reason: str
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
