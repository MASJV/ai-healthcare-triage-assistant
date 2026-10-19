"""
eval_triage.py

Runs the Triage Agent on the 50 cases in triage_test_cases.py
and prints the accuracy.

Besides accuracy it counts:
    under-triage - a lower level than expected (e.g. emergency -> urgent),
                   the dangerous mistake
    over-triage  - a higher level than expected (e.g. routine -> urgent),
                   the safe mistake

Run:
    python eval_triage.py
"""

import time

from langchain_core.messages import HumanMessage, AIMessage

from triage_agent import triage_agent
from evaluation.triage_test_cases import TEST_CASES
from evaluation.latency import print_latency

LABELS = ["emergency", "urgent", "routine", "self_care", "needs_more_info"]

# needs_more_info is not a level, so it is left out of under/over-triage
SEVERITY = {
    "self_care": 0,
    "routine": 1,
    "urgent": 2,
    "emergency": 3
}


def build_messages(history, message):
    """
    Turns the (role, text) history plus the latest message
    into the message list triage expects.
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
    under_triage = 0
    over_triage = 0
    latencies = []

    # per-label counts
    total_per_label = {}
    correct_per_label = {}
    for label in LABELS:
        total_per_label[label] = 0
        correct_per_label[label] = 0

    for number, case in enumerate(TEST_CASES, start=1):

        state = {
            "messages": build_messages(case["history"], case["message"]),
            "user_input": case["message"],
            "follow_ups": case["follow_ups"]
        }

        start = time.perf_counter()
        output = triage_agent(state)
        latencies.append(time.perf_counter() - start)
        predicted = output["urgency"]
        probabilities = output.get("triage_probabilities", {})
        expected = case["expected"]

        total_per_label[expected] += 1

        if predicted == expected:
            correct += 1
            correct_per_label[expected] += 1
            status = "PASS"
        else:
            wrong_cases.append((number, case, predicted, probabilities))
            status = "FAIL"

            if expected in SEVERITY and predicted in SEVERITY:
                if SEVERITY[predicted] < SEVERITY[expected]:
                    under_triage += 1
                else:
                    over_triage += 1

        print(f"[{number:02d}] {status}  expected={expected:<15}  predicted={predicted:<15}  {case['message']}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print("\n========== TRIAGE EVALUATION ==========\n")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1f}%\n")

    print("Per label:")
    for label in LABELS:
        label_total = total_per_label[label]
        label_correct = correct_per_label[label]
        if label_total > 0:
            label_accuracy = label_correct / label_total * 100
            print(f"  {label:<15} {label_correct}/{label_total} = {label_accuracy:.1f}%")

    print(f"\nUnder-triage (level too low):  {under_triage}")
    print(f"Over-triage (level too high):  {over_triage}")

    print_latency(latencies)

    if wrong_cases:
        print("\nWrong cases:")
        for number, case, predicted, probabilities in wrong_cases:
            print(f"  [{number:02d}] {case['scenario']}")
            print(f"       message:   {case['message']}")
            print(f"       expected:  {case['expected']}   predicted: {predicted}")
            if probabilities:
                print(f"       probabilities: {probabilities}")


if __name__ == "__main__":
    run_eval()
