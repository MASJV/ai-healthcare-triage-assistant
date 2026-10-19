"""
eval_care_bench.py

External validation on CARE-Bench Public Test 1 (62 cases, 135 labelled prefixes).

Cases:  evaluation/care_bench/care_bench_public_test.jsonl, the public_test_1 rows of
        data/public/evaluation_prefixes_public.jsonl from
        https://github.com/ningkko/CARE-bench (CC BY-NC 4.0)
Source: CARE-Bench: Benchmarking Patient-Facing LLM Triage.
        https://arxiv.org/abs/2608.03731

These cases are for testing only. Prompts are not changed to fit them.

Each prefix is the patient's own messages up to one point in a conversation
(1 to 3 rounds). As in the official CARE-Bench inputs, the rounds are joined
into one message (separated by blank lines), since the benchmark has no
assistant turns between them. That message goes through the app's path:
    Router Agent -> health_info -> 0B self-care / monitor (the knowledge agent answers)
                 -> symptom     -> Triage Agent -> needs_more_info -> 0A info needed
                                                 -> self_care       -> 0B self-care / monitor
                                                 -> routine/urgent  -> 1A non-urgent care
                                                 -> emergency       -> 1B urgent care
                 -> booking     -> 1A non-urgent care (the scheduling agent books an appointment)
                 -> other       -> 0A info needed (the app asks what the symptom is)

follow_ups is set to the number of earlier rounds (at most MAX_FOLLOW_UPS),
so triage may still ask a follow-up in rounds 1 and 2 and must decide in round 3.
Triage level "urgent" (doctor within 24-48 hours) counts as 1A, not 1B:
1B means urgent or emergency care now.

Metrics follow the official scorer (code/score_label_submission.py):
accuracy, macro-F1, under- and over-escalation, missed care, and urgent
false negatives (1B cases not sent to 1B), with every missed 1B case listed.
Predictions are saved to evaluation/care_bench/care_bench_predictions_<RUN_NAME>.csv
(prefix_id,predicted_label), so runs before and after a change can be compared
and checked with the official scorer.

Run:
    python -m evaluation.eval_care_bench
"""

import csv
import json
import time

from langchain_core.messages import HumanMessage

from config import MAX_FOLLOW_UPS
from agents.router_agent import router_agent
from agents.triage_agent import triage_agent
from evaluation.latency import print_latency

RUN_NAME = "v4_jev"   # change for each run you want to keep
CASES_FILE = "evaluation/care_bench/care_bench_public_test.jsonl"
PREDICTIONS_FILE = f"evaluation/care_bench/care_bench_predictions_{RUN_NAME}.csv"

INFO_NEEDED = "0A_NO_ESCALATION_INFO_NEEDED"
SELF_CARE = "0B_NO_ESCALATION_SELF_CARE_MONITOR"
NONURGENT = "1A_ESCALATION_NONURGENT_CARE"
URGENT = "1B_ESCALATION_URGENT_CARE"

# same order as the official scorer: a higher number is a stronger escalation
LABELS = [INFO_NEEDED, SELF_CARE, NONURGENT, URGENT]
SEVERITY = {
    INFO_NEEDED: 0,
    SELF_CARE: 1,
    NONURGENT: 2,
    URGENT: 3
}

SHORT_NAME = {
    INFO_NEEDED: "0A info",
    SELF_CARE: "0B self-care",
    NONURGENT: "1A non-urgent",
    URGENT: "1B urgent"
}


def load_cases(path):
    """
    Reads the JSONL file: one prefix per line with prefix_id, round_id,
    gold_label and input_patient_turns (the patient's messages so far).
    """
    cases = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def predict(patient_turns, round_id):
    """
    Runs the router and, for symptoms, the triage agent on one prefix.
    Returns (CARE-Bench label, router intent, triage level).
    """
    # all rounds so far in one message, as in the official CARE-Bench inputs
    message = "\n\n".join(patient_turns)

    state = {
        "messages": [HumanMessage(content=message)],
        "user_input": message,
        "follow_ups": min(round_id - 1, MAX_FOLLOW_UPS)
    }

    intent = router_agent(state)["intent"]

    if intent == "health_info":
        return SELF_CARE, intent, ""
    if intent == "booking":
        return NONURGENT, intent, ""
    if intent != "symptom":
        return INFO_NEEDED, intent, ""

    urgency = triage_agent(state)["urgency"]

    if urgency == "emergency":
        return URGENT, intent, urgency
    if urgency == "needs_more_info":
        return INFO_NEEDED, intent, urgency
    if urgency == "self_care":
        return SELF_CARE, intent, urgency

    return NONURGENT, intent, urgency


def macro_f1(records):
    """
    Average F1 over the four labels, as in the official scorer.
    records is a list of (gold, predicted) pairs.
    """
    f1_scores = []
    for label in LABELS:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for gold, predicted in records:
            if gold == label and predicted == label:
                true_positive += 1
            elif gold != label and predicted == label:
                false_positive += 1
            elif gold == label and predicted != label:
                false_negative += 1

        precision = 0
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        recall = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def save_predictions(path, predictions):
    """
    Writes prefix_id,predicted_label rows, the official submission format.
    """
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prefix_id", "predicted_label"])
        for prefix_id, label in predictions:
            writer.writerow([prefix_id, label])


def run_eval():

    cases = load_cases(CASES_FILE)

    records = []
    predictions = []
    wrong_cases = []
    missed_urgent = []
    latencies = []

    total_per_label = {}
    correct_per_label = {}
    for label in LABELS:
        total_per_label[label] = 0
        correct_per_label[label] = 0

    # confusion[gold][predicted] = count
    confusion = {}
    for gold_label in LABELS:
        confusion[gold_label] = {}
        for predicted_label in LABELS:
            confusion[gold_label][predicted_label] = 0

    for number, case in enumerate(cases, start=1):

        gold = case["gold_label"]
        start = time.perf_counter()
        predicted, intent, urgency = predict(case["input_patient_turns"], case["round_id"])
        latencies.append(time.perf_counter() - start)

        records.append((gold, predicted))
        predictions.append((case["prefix_id"], predicted))
        total_per_label[gold] += 1
        confusion[gold][predicted] += 1

        if predicted == gold:
            correct_per_label[gold] += 1
            status = "PASS"
        else:
            wrong_cases.append((number, case, predicted, intent, urgency))
            status = "FAIL"
            if gold == URGENT:
                missed_urgent.append((number, case, predicted, intent, urgency))

        last_turn = case["input_patient_turns"][-1]
        print(f"[{number:03d}] {status}  round {case['round_id']}  gold={SHORT_NAME[gold]:<13} predicted={SHORT_NAME[predicted]:<13} {last_turn[:60]}...")

    save_predictions(PREDICTIONS_FILE, predictions)

    total = len(records)
    correct = 0
    over = 0
    under = 0
    missed_care = 0
    needs_care = 0
    for gold, predicted in records:
        if gold == predicted:
            correct += 1
        if SEVERITY[predicted] > SEVERITY[gold]:
            over += 1
        if SEVERITY[predicted] < SEVERITY[gold]:
            under += 1
        # professional care needed (1A or 1B) but the answer was 0A or 0B
        if gold in [NONURGENT, URGENT]:
            needs_care += 1
            if predicted in [INFO_NEEDED, SELF_CARE]:
                missed_care += 1

    print(f"\n========== CARE-BENCH EXTERNAL VALIDATION ({RUN_NAME}) ==========\n")
    print(f"Accuracy:  {correct}/{total} = {correct / total * 100:.1f}%")
    print(f"Macro-F1:  {macro_f1(records) * 100:.1f}\n")

    print("Per label (recall):")
    for label in LABELS:
        label_total = total_per_label[label]
        if label_total > 0:
            recall = correct_per_label[label] / label_total * 100
            print(f"  {SHORT_NAME[label]:<13} {correct_per_label[label]}/{label_total} = {recall:.1f}%")

    print("\nConfusion (rows = gold, columns = predicted):")
    header = f"  {'':<13}"
    for predicted_label in LABELS:
        header += f" {SHORT_NAME[predicted_label]:>13}"
    print(header)
    for gold_label in LABELS:
        row = f"  {SHORT_NAME[gold_label]:<13}"
        for predicted_label in LABELS:
            row += f" {confusion[gold_label][predicted_label]:>13}"
        print(row)

    print(f"\nUnder-escalation (weaker action than gold):  {under}/{total}")
    print(f"Over-escalation (stronger action than gold):  {over}/{total}")
    print(f"Missed care (gold 1A/1B, answered 0A/0B):     {missed_care}/{needs_care}")
    print(f"Urgent false negatives (gold 1B, not 1B):     {len(missed_urgent)}/{total_per_label[URGENT]}")

    print_latency(latencies)
    print(f"\nPredictions saved to {PREDICTIONS_FILE}")

    if missed_urgent:
        print("\nMissed urgent cases:")
        for number, case, predicted, intent, urgency in missed_urgent:
            print(f"  [{number:03d}] {case['prefix_id']}   predicted: {SHORT_NAME[predicted]}   (router: {intent}, triage: {urgency or '-'})")
            print(f"        patient: {' / '.join(case['input_patient_turns'])[:250]}")

    if wrong_cases:
        print("\nWrong cases:")
        for number, case, predicted, intent, urgency in wrong_cases:
            print(f"  [{number:03d}] {case['prefix_id']}   gold: {SHORT_NAME[case['gold_label']]}   predicted: {SHORT_NAME[predicted]}   (router: {intent}, triage: {urgency or '-'})")
            print(f"        patient: {' / '.join(case['input_patient_turns'])[:250]}")


if __name__ == "__main__":
    run_eval()
