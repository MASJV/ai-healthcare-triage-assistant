"""
jev.py

Calls Jev, TypeSafe AI's decision model, through OpenRouter.
Jev does not write text: it answers typed questions and returns
probabilities. The router and triage each ask one choice question
(pick one option); jev_ask can send several questions in one request.

Needs OPENROUTER_API_KEY in .env.
"""

import os

import requests
from langchain_core.messages import HumanMessage
from langsmith import traceable

import config


def conversation_list(messages):
    """
    The conversation as a list of {"from": "user"/"assistant", "text": ...},
    the structured format TypeSafe recommends for Jev's state.
    """
    conversation = []
    for message in messages:
        if isinstance(message, HumanMessage):
            speaker = "user"
        else:
            speaker = "assistant"
        conversation.append({"from": speaker, "text": message.content})
    return conversation


@traceable(name="jevAsk")
def jev_ask(state, questions):
    """
    Sends one request to Jev.
        state     - dict of context fields (e.g. the conversation)
        questions - {question_id: {"type": ..., "instructions": ..., "criteria": ...}}
    Returns {question_id: answer}. Raises if the API call fails.
    """
    response = requests.post(
        config.JEV_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.JEV_MODEL,
            "state": state,
            "questions": questions
        },
        timeout=10   # a normal call takes about a second
    )
    response.raise_for_status()

    return response.json()["answers"]


def jev_choice(state, instructions, criteria):
    """
    Asks Jev one choice question.
    Returns (choice, probabilities).
    """
    answers = jev_ask(
        state,
        {
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria
            }
        }
    )
    answer = answers["decision"]
    return answer["choice"], answer["probabilities"]
