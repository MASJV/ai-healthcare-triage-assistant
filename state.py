from typing import TypedDict


#Shared State
class UserState(TypedDict):
    user_input: str
    intent: str
    urgency: str
    triage_reason: str
    result: str
