"""
eval_scheduling.py

Runs the Scheduling Agent's LLM step (read_request) on the 52 cases in
scheduling_test_cases.py and prints the accuracy.

Only read_request is called, so appointments.db is never read or changed.
The phone is cleaned with clean_phone() and the time is written as HH:MM
before comparing, the same way the agent does before booking.

A case passes only if all five fields are right.

Run:
    python -m evaluation.eval_scheduling
"""

from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from agents.scheduling_agent import read_request, clean_phone
from evaluation.test_cases.scheduling_test_cases import TEST_CASES

FIELDS = ["action", "name", "phone", "date", "time"]
ACTIONS = ["book", "reschedule", "cancel", "check", "none"]


def build_messages(history, message):
    """
    Turns the (role, text) history plus the latest message
    into the message list the agent expects.
    """
    messages = []
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))

    messages.append(HumanMessage(content=message))
    return messages


def clean_time(time):
    """
    "9:30" -> "09:30". Anything that is not a valid time stays as it is.
    """
    try:
        return datetime.strptime(time, "%H:%M").strftime("%H:%M")
    except ValueError:
        return time


def clean_request(request):
    """
    Puts the LLM's fields in the same form as the expected ones.
    """
    return {
        "action": request["action"],
        "name": request["name"].strip().lower(),
        "phone": clean_phone(request["phone"]),
        "date": request["date"].strip(),
        "time": clean_time(request["time"].strip())
    }


def run_eval():

    correct = 0
    wrong_cases = []

    correct_per_field = {}
    for field in FIELDS:
        correct_per_field[field] = 0

    total_per_action = {}
    correct_per_action = {}
    for action in ACTIONS:
        total_per_action[action] = 0
        correct_per_action[action] = 0

    for number, case in enumerate(TEST_CASES, start=1):

        state = {
            "messages": build_messages(case["history"], case["message"]),
            "user_input": case["message"],
            "urgency": case.get("urgency", "")
        }

        predicted = clean_request(read_request(state))

        expected = dict(case["expected"])
        expected["name"] = expected["name"].lower()

        wrong_fields = []
        for field in FIELDS:
            if predicted[field] == expected[field]:
                correct_per_field[field] += 1
            else:
                wrong_fields.append(field)

        total_per_action[expected["action"]] += 1

        if not wrong_fields:
            correct += 1
            correct_per_action[expected["action"]] += 1
            status = "PASS"
        else:
            wrong_cases.append((number, case, predicted, wrong_fields))
            status = "FAIL"

        print(f"[{number:02d}] {status}  {case['message']}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print("\n========== SCHEDULING EVALUATION ==========\n")
    print(f"All five fields right: {correct}/{total} = {accuracy:.1f}%\n")

    print("Per field:")
    for field in FIELDS:
        field_accuracy = correct_per_field[field] / total * 100
        print(f"  {field:<7} {correct_per_field[field]}/{total} = {field_accuracy:.1f}%")

    print("\nPer action (all five fields right):")
    for action in ACTIONS:
        action_total = total_per_action[action]
        if action_total > 0:
            action_accuracy = correct_per_action[action] / action_total * 100
            print(f"  {action:<10} {correct_per_action[action]}/{action_total} = {action_accuracy:.1f}%")

    if wrong_cases:
        print("\nWrong cases:")
        for number, case, predicted, wrong_fields in wrong_cases:
            print(f"  [{number:02d}] {case['scenario']}")
            print(f"       message:   {case['message']}")
            for field in wrong_fields:
                print(f"       {field:<7} expected: {case['expected'][field]!r:<15} predicted: {predicted[field]!r}")


if __name__ == "__main__":
    run_eval()
