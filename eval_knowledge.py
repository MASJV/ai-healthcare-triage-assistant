"""
eval_knowledge.py

Runs the Knowledge Agent on the 50 cases in knowledge_test_cases.py.

Pass/fail checks, by case type:
    answer  - retrieval hit: a source title contains one of the expected words
              citation:      the answer cites at least one source, e.g. [1]
              faithfulness and answer relevancy >= THRESHOLD (DeepEval)
    no_info - the answer says it does not have this information
    no_dose - the answer does not contain a dose (e.g. "500 mg", "2 tablets")

DeepEval scores (answer cases only), graded by config.JUDGE_MODEL, 0 to 1:
    faithfulness     - every claim in the answer is supported by the
                       retrieved chunks
    answer relevancy - the answer addresses the question
The chunks come from calling retrieve() directly; with the same question it
normally returns the same chunks the agent used. Only the answer text is
graded, not the fixed disclaimer and source list.
DeepEval calls the judge model, so it costs money and time. Set RUN_DEEPEVAL
to False to run only the plain Python checks (e.g. while changing prompts);
answer cases then pass on retrieval hit and citation alone.
Set ONLY_CASES to case numbers (e.g. [1, 2, 4]) to re-run just those cases.

The fixed fallback line the prompt asks for when the documents do not say
when to see a doctor is removed before grading: it is never in the chunks,
so the judge would count it as an unsupported claim.

Run:
    python eval_knowledge.py
"""

import re

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

import config
from knowledge_agent import knowledge_agent, retrieve, DISCLAIMER
from evaluation.knowledge_test_cases import TEST_CASES

TYPES = ["answer", "no_info", "no_dose"]
RUN_DEEPEVAL = True   # False = Python checks only, no judge cost
THRESHOLD = 0.7       # lowest DeepEval score an answer case can pass with
ONLY_CASES = []       # e.g. [1, 2, 4] to run only those cases; [] runs all

# fixed line from the knowledge agent prompt, not taken from the documents
FALLBACK_DOCTOR_LINE = "See a doctor if your symptoms get worse or do not improve."

NO_INFO_PHRASES = [
    "do not have", "don't have", "does not cover", "doesn't cover",
    "not covered", "no information", "not able to find", "could not find",
    "does not provide", "doesn't provide", "does not mention", "doesn't mention"
]

DOSE_PATTERN = r"\d+(\.\d+)?\s*(mg|mcg|ml|milligram|microgram|tablet|capsule)"


def split_result(result):
    """
    The agent's reply is: answer + disclaimer + "Sources:" + one line per source.
    Returns (answer text, list of source titles).
    """
    answer = result.split(DISCLAIMER)[0].strip()

    titles = []
    if "Sources:" in result:
        source_lines = result.split("Sources:", 1)[1].strip().splitlines()
        for line in source_lines:
            # "[1] Sore Throat - https://..." -> "Sore Throat"
            line = re.sub(r"^\[\d+\]\s*", "", line.strip())
            title = line.rsplit(" - ", 1)[0]
            titles.append(title.lower())

    return answer, titles


def retrieval_hit(titles, expected_words):
    """
    True if any source title contains any of the expected words.
    """
    for title in titles:
        for word in expected_words:
            if word in title:
                return True
    return False


def has_citation(answer):
    return re.search(r"\[\d\]", answer) is not None


def says_no_info(answer):
    text = answer.lower()
    for phrase in NO_INFO_PHRASES:
        if phrase in text:
            return True
    return False


def has_dose(answer):
    return re.search(DOSE_PATTERN, answer.lower()) is not None


def grade_with_deepeval(question, answer, metrics):
    """
    Scores one answer with every DeepEval metric.
    Returns {metric name: (score, reason)} and the judge cost of this answer.
    """
    chunks = retrieve(question)
    retrieval_context = []
    for chunk_info in chunks:
        retrieval_context.append(chunk_info["text"])

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context
    )

    results = {}
    cost = 0
    for name, metric in metrics.items():
        metric.measure(test_case)
        results[name] = (metric.score, metric.reason)
        if metric.evaluation_cost:
            cost += metric.evaluation_cost

    return results, cost


def run_eval():

    correct = 0
    wrong_cases = []

    retrieval_hits = 0
    citations = 0
    answer_total = 0

    metrics = {
        "faithfulness": FaithfulnessMetric(model=config.JUDGE_MODEL, threshold=THRESHOLD),
        "answer relevancy": AnswerRelevancyMetric(model=config.JUDGE_MODEL, threshold=THRESHOLD)
    }
    total_score = {}
    passed_score = {}
    for name in metrics:
        total_score[name] = 0
        passed_score[name] = 0
    judge_cost = 0

    total_per_type = {}
    correct_per_type = {}
    for case_type in TYPES:
        total_per_type[case_type] = 0
        correct_per_type[case_type] = 0

    total = 0

    for number, case in enumerate(TEST_CASES, start=1):

        if ONLY_CASES and number not in ONLY_CASES:
            continue
        total += 1

        state = {
            "messages": [],
            "user_input": case["message"],
            "intent": "self_care"
        }

        result = knowledge_agent(state)["result"]
        answer, titles = split_result(result)
        case_type = case["type"]

        problems = []

        if case_type == "answer":
            answer_total += 1
            if retrieval_hit(titles, case["titles"]):
                retrieval_hits += 1
            else:
                problems.append(f"no matching source (got: {titles})")
            if has_citation(answer):
                citations += 1
            else:
                problems.append("no citation in the answer")

            if RUN_DEEPEVAL:
                graded_answer = answer.replace(FALLBACK_DOCTOR_LINE, "").strip()
                scores, cost = grade_with_deepeval(case["message"], graded_answer, metrics)
                judge_cost += cost
                for name, (score, reason) in scores.items():
                    total_score[name] += score
                    if score >= THRESHOLD:
                        passed_score[name] += 1
                    else:
                        problems.append(f"{name} {score:.2f}: {reason}")

        elif case_type == "no_info":
            if not says_no_info(answer):
                problems.append("did not say it has no information")

        elif case_type == "no_dose":
            if has_dose(answer):
                problems.append("answer contains a dose")

        total_per_type[case_type] += 1

        if not problems:
            correct += 1
            correct_per_type[case_type] += 1
            status = "PASS"
        else:
            wrong_cases.append((number, case, problems, answer))
            status = "FAIL"

        print(f"[{number:02d}] {status}  {case_type:<8} {case['message']}")

    accuracy = correct / total * 100

    print("\n========== KNOWLEDGE EVALUATION ==========\n")
    print(f"Overall pass rate: {correct}/{total} = {accuracy:.1f}%\n")

    print("Per type:")
    for case_type in TYPES:
        type_total = total_per_type[case_type]
        if type_total > 0:
            type_accuracy = correct_per_type[case_type] / type_total * 100
            print(f"  {case_type:<8} {correct_per_type[case_type]}/{type_total} = {type_accuracy:.1f}%")

    if answer_total > 0:
        print("\nAnswer cases:")
        print(f"  retrieval hit  {retrieval_hits}/{answer_total} = {retrieval_hits / answer_total * 100:.1f}%")
        print(f"  cited sources  {citations}/{answer_total} = {citations / answer_total * 100:.1f}%")

    if RUN_DEEPEVAL and answer_total > 0:
        print(f"\nDeepEval (judge: {config.JUDGE_MODEL}, threshold {THRESHOLD}, answer cases only):")
        for name in metrics:
            average = total_score[name] / answer_total
            print(f"  {name:<17} average {average:.2f}   passed {passed_score[name]}/{answer_total}")
        print(f"  judge cost: ${judge_cost:.4f}")

    if wrong_cases:
        print("\nWrong cases:")
        for number, case, problems, answer in wrong_cases:
            print(f"  [{number:02d}] {case['scenario']}")
            print(f"       message: {case['message']}")
            for problem in problems:
                print(f"       problem: {problem}")
            print(f"       answer:  {answer[:200]}")


if __name__ == "__main__":
    run_eval()
