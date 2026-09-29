"""
eval_router.py

Runs the Router Agent on the 50 cases in router_test_cases.py
and prints the accuracy.

Run:
    python eval_router.py
"""

from langchain_core.messages import HumanMessage, AIMessage

from router_agent import router_agent
from evaluation.router_test_cases import TEST_CASES

LABELS = ["symptom", "self_care", "booking", "other"]


def build_messages(history, message):
    """
    Turns the (role, text) history plus the latest message
    into the message list the router expects.
    """
    messages = []
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))

    messages.append(HumanMessage(content=message))
    return messages


def run_eval():

    correct = 0
    wrong_cases = []

    # per-label counts
    total_per_label = {}
    correct_per_label = {}
    for label in LABELS:
        total_per_label[label] = 0
        correct_per_label[label] = 0

    for number, case in enumerate(TEST_CASES, start=1):

        state = {
            "messages": build_messages(case["history"], case["message"]),
            "user_input": case["message"]
        }

        predicted = router_agent(state)["intent"]
        expected = case["expected"]

        total_per_label[expected] += 1

        if predicted == expected:
            correct += 1
            correct_per_label[expected] += 1
            status = "PASS"
        else:
            wrong_cases.append((number, case, predicted))
            status = "FAIL"

        print(f"[{number:02d}] {status}  expected={expected:<9}  predicted={predicted:<9}  {case['message']}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print("\n========== ROUTER EVALUATION ==========\n")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1f}%\n")

    print("Per label:")
    for label in LABELS:
        label_total = total_per_label[label]
        label_correct = correct_per_label[label]
        if label_total > 0:
            label_accuracy = label_correct / label_total * 100
            print(f"  {label:<9} {label_correct}/{label_total} = {label_accuracy:.1f}%")

    if wrong_cases:
        print("\nWrong cases:")
        for number, case, predicted in wrong_cases:
            print(f"  [{number:02d}] {case['scenario']}")
            print(f"       message:   {case['message']}")
            print(f"       expected:  {case['expected']}   predicted: {predicted}")


if __name__ == "__main__":
    run_eval()
