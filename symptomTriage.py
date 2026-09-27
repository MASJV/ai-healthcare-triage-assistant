from langgraph.graph import StateGraph, START, END

from state import UserState
from router_agent import router_agent, route
from knowledge_agent import knowledge_agent
from scheduling_agent import scheduling_agent


#Build Graph
graph = StateGraph(UserState)

graph.add_node("Router Agent", router_agent)
graph.add_node("Knowledge Agent", knowledge_agent)
graph.add_node("Scheduling Agent", scheduling_agent)

graph.add_edge(START, "Router Agent")

graph.add_conditional_edges(
    "Router Agent",
    route,
    {
        "Knowledge Agent": "Knowledge Agent",
        "Scheduling Agent": "Scheduling Agent"
    }
)

graph.add_edge("Knowledge Agent", END)
graph.add_edge("Scheduling Agent", END)

app = graph.compile()


#Chat Loop
if __name__ == "__main__":

    while True:

        user_input = input("\nEnter your message (or 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {
                "user_input": user_input
            }
        )

        print("\n========== RESULT ==========\n")
        print("Intent:", result["intent"])
        print(result["result"])

    print("\nGraph Structure:\n")
    print(app.get_graph().draw_mermaid())
