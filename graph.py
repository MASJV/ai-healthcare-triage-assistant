import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import UserState
from agents.router_agent import router_agent, route
from agents.triage_agent import triage_agent, triage_route
from agents.knowledge_agent import knowledge_agent
from agents.scheduling_agent import scheduling_agent
from agents.escalation_agent import escalation_agent


#Build Graph
graph = StateGraph(UserState)

graph.add_node("Router Agent", router_agent)
graph.add_node("Triage Agent", triage_agent)
graph.add_node("Knowledge Agent", knowledge_agent)
graph.add_node("Scheduling Agent", scheduling_agent)
graph.add_node("Escalation Agent", escalation_agent)

graph.add_edge(START, "Router Agent")

graph.add_conditional_edges(
    "Router Agent",
    route,
    {
        "Triage Agent": "Triage Agent",
        "Knowledge Agent": "Knowledge Agent",
        "Scheduling Agent": "Scheduling Agent",
        "End": END
    }
)

graph.add_conditional_edges(
    "Triage Agent",
    triage_route,
    {
        "Scheduling Agent": "Scheduling Agent",
        "Escalation Agent": "Escalation Agent",
        "Knowledge Agent": "Knowledge Agent",
        "Ask User": END           
    }
)

graph.add_edge("Knowledge Agent", END)
graph.add_edge("Scheduling Agent", END)
graph.add_edge("Escalation Agent", END)

# MemorySaver keeps the state of each chat session in memory, keyed by thread_id
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


#Chat Loop
if __name__ == "__main__":

    # one thread_id per run per chat session
    # run_name labels each turn in LangSmith when tracing is on
    session = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "run_name": "ai-healthcare-triage-assistant"
    }

    while True:

        user_input = input("\nEnter your message (or 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        # per-turn fields are reset; messages and follow_ups carry over
        result = app.invoke(
            {
                "user_input": user_input,
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "urgency": "",
                "triage_reason": "",
                "result": ""
            },
            session
        )

        print("\n========== RESULT ==========\n")
        print("Intent:", result["intent"])
        if result["urgency"]:
            print("Urgency:", result["urgency"], "-", result["triage_reason"])
        if result["result"]:
            print(result["result"])

    print("\nGraph Structure:\n")
    print(app.get_graph().draw_ascii())
