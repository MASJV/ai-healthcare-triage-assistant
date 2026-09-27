from state import UserState
from config import llm


#Scheduling Agent
def scheduling_agent(state: UserState):

    prompt = f"""
    User Message: {state['user_input']}

    You are an appointment scheduling assistant.
    Identify what the user wants (book / reschedule / cancel) and any
    doctor, department, date or time mentioned.
    Keep the answer within 5 lines.
    """

    response = llm.invoke(prompt)

    return {
        "result": response.content
    }
