"""
scheduling_test_cases.py

52 test cases for the Scheduling Agent's LLM step (read_request).
The LLM only fills in five fields; every booking rule is plain Python,
so this checks what the LLM reads from the conversation.

Each case has:
    scenario - short tag of what the case checks
    history  - earlier messages as (role, text); role is "user" or "assistant"
    message  - the latest user message
    urgency  - (only for messages sent on by triage) the level triage gave
    expected - the five fields:
               action - book / reschedule / cancel / check / none
               name   - full name as written, "" if not given
               phone  - 10 digits after cleanup, "" if not given
               date   - YYYY-MM-DD, "" if not given
               time   - HH:MM (24-hour), "" if not given

Dates are worked out from today, so "tomorrow" always means the real tomorrow.
No case asks for a time later today, so the result does not depend on the
hour the test is run.
"""

from datetime import date, timedelta

TODAY = date.today()


def day(offset):
    """
    Date offset days from today as YYYY-MM-DD.
    """
    return (TODAY + timedelta(days=offset)).strftime("%Y-%m-%d")


def spoken_day(offset):
    """
    Date offset days from today the way a user writes it, e.g. "5 October".
    """
    future = TODAY + timedelta(days=offset)
    return f"{future.day} {future.strftime('%B')}"


def days_until(weekday):
    """
    Days from today to the next given weekday (Monday = 0 ... Sunday = 6).
    Always 1-7, never today.
    """
    offset = (weekday - TODAY.weekday()) % 7
    if offset == 0:
        offset = 7
    return offset


NAME_PHONE_REPLY = "Please tell me the patient's full name and phone number."
SLOTS_REPLY = (
    "I can book an appointment for you. Free slots:\n"
    f"{(TODAY + timedelta(days=1)).strftime('%a %d %b')}: 03:30 PM, 04:00 PM, 04:30 PM, 05:00 PM\n\n"
    "Reply with the patient's full name, phone number and the slot you want."
)


TEST_CASES = [

    # ---------- book (16) ----------
    {
        "scenario": "everything in one message",
        "history": [],
        "message": "Book an appointment for tomorrow at 10am. Name Ravi Shah, phone 9876543210",
        "expected": {"action": "book", "name": "Ravi Shah", "phone": "9876543210", "date": day(1), "time": "10:00"}
    },
    {
        "scenario": "day after tomorrow, 12-hour time",
        "history": [],
        "message": "I want to see a doctor day after tomorrow at 3pm",
        "expected": {"action": "book", "name": "", "phone": "", "date": day(2), "time": "15:00"}
    },
    {
        "scenario": "written date, +91 phone with spaces",
        "history": [],
        "message": f"please book me on {spoken_day(5)} at 11:30, Priya Mehta, +91 98250 12345",
        "expected": {"action": "book", "name": "Priya Mehta", "phone": "9825012345", "date": day(5), "time": "11:30"}
    },
    {
        "scenario": "hinglish",
        "history": [],
        "message": "kal subah 10 baje ka appointment chahiye, Amit Desai 9898989898",
        "expected": {"action": "book", "name": "Amit Desai", "phone": "9898989898", "date": day(1), "time": "10:00"}
    },
    {
        "scenario": "vague time must not be guessed",
        "history": [],
        "message": "Book for my mother Sunita Patel, 9123456780, tomorrow evening",
        "expected": {"action": "book", "name": "Sunita Patel", "phone": "9123456780", "date": day(1), "time": ""}
    },
    {
        "scenario": "vague date must not be guessed",
        "history": [],
        "message": "need an appointment sometime next week",
        "expected": {"action": "book", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "time in words",
        "history": [],
        "message": "book me at half past nine tomorrow morning",
        "expected": {"action": "book", "name": "", "phone": "", "date": day(1), "time": "09:30"}
    },
    {
        "scenario": "asked as a question, noon",
        "history": [],
        "message": "can I get a slot tomorrow at noon?",
        "expected": {"action": "book", "name": "", "phone": "", "date": day(1), "time": "12:00"}
    },
    {
        "scenario": "weekday name",
        "history": [],
        "message": "book on Friday at 5 pm",
        "expected": {"action": "book", "name": "", "phone": "", "date": day(days_until(4)), "time": "17:00"}
    },
    {
        "scenario": "first name only",
        "history": [],
        "message": "book an appointment for Rahul",
        "expected": {"action": "book", "name": "Rahul", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "phone with dashes",
        "history": [],
        "message": "Book tomorrow 8:30am for Meera Nair, phone 987-654-3210",
        "expected": {"action": "book", "name": "Meera Nair", "phone": "9876543210", "date": day(1), "time": "08:30"}
    },
    {
        "scenario": "in N days",
        "history": [],
        "message": "book in 3 days at 6pm",
        "expected": {"action": "book", "name": "", "phone": "", "date": day(3), "time": "18:00"}
    },
    {
        "scenario": "date and time from earlier message",
        "history": [
            ("user", "book me tomorrow at 10am"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "Ravi Shah 9876543210",
        "expected": {"action": "book", "name": "Ravi Shah", "phone": "9876543210", "date": day(1), "time": "10:00"}
    },
    {
        "scenario": "picks a slot from the offered list",
        "history": [
            ("user", "I need an appointment tomorrow"),
            ("assistant", SLOTS_REPLY)
        ],
        "message": "Neha Joshi, 9988776655, the 4:30 pm one",
        "expected": {"action": "book", "name": "Neha Joshi", "phone": "9988776655", "date": day(1), "time": "16:30"}
    },
    {
        "scenario": "new time after slot was taken",
        "history": [
            ("user", "book tomorrow 10am, Ravi Shah 9876543210"),
            ("assistant", "That slot is already booked. Free slots:\n"
                          "08:00 AM, 08:30 AM, 11:00 AM, 11:30 AM")
        ],
        "message": "ok then 11 am",
        "expected": {"action": "book", "name": "Ravi Shah", "phone": "9876543210", "date": day(1), "time": "11:00"}
    },
    {
        "scenario": "user corrects the time",
        "history": [
            ("user", "book me for tomorrow at 9 am"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "Karan Singh 9000012345, actually make it 2pm instead",
        "expected": {"action": "book", "name": "Karan Singh", "phone": "9000012345", "date": day(1), "time": "14:00"}
    },

    # ---------- reschedule (8) ----------
    {
        "scenario": "plain reschedule",
        "history": [],
        "message": "Move my appointment to tomorrow at 4pm. Ravi Shah 9876543210",
        "expected": {"action": "reschedule", "name": "Ravi Shah", "phone": "9876543210", "date": day(1), "time": "16:00"}
    },
    {
        "scenario": "reschedule without the word",
        "history": [],
        "message": "I booked for today but can't make it, can we do day after tomorrow 11am instead?",
        "expected": {"action": "reschedule", "name": "", "phone": "", "date": day(2), "time": "11:00"}
    },
    {
        "scenario": "old and new day in one message",
        "history": [],
        "message": "please shift my Thursday appointment to Saturday at 10:30",
        "expected": {"action": "reschedule", "name": "", "phone": "", "date": day(days_until(5)), "time": "10:30"}
    },
    {
        "scenario": "unknown old time must not be guessed",
        "history": [],
        "message": "reschedule it to the same time tomorrow",
        "expected": {"action": "reschedule", "name": "", "phone": "", "date": day(1), "time": ""}
    },
    {
        "scenario": "reschedule over two turns",
        "history": [
            ("user", "I want to reschedule"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "Priya Mehta 9825012345, move it to 5 pm tomorrow",
        "expected": {"action": "reschedule", "name": "Priya Mehta", "phone": "9825012345", "date": day(1), "time": "17:00"}
    },
    {
        "scenario": "indian english 'prepone'",
        "history": [],
        "message": "prepone my appointment to tomorrow 7pm, Anil Kapoor 9811122233",
        "expected": {"action": "reschedule", "name": "Anil Kapoor", "phone": "9811122233", "date": day(1), "time": "19:00"}
    },
    {
        "scenario": "only the time changes",
        "history": [],
        "message": "change my booking time to 9 in the morning, same day",
        "expected": {"action": "reschedule", "name": "", "phone": "", "date": "", "time": "09:00"}
    },
    {
        "scenario": "hinglish",
        "history": [],
        "message": "mera appointment parso 12 baje shift kar do, Sneha Rao 9765432109",
        "expected": {"action": "reschedule", "name": "Sneha Rao", "phone": "9765432109", "date": day(2), "time": "12:00"}
    },

    # ---------- cancel (8) ----------
    {
        "scenario": "plain cancel",
        "history": [],
        "message": "Cancel my appointment. Ravi Shah, 9876543210",
        "expected": {"action": "cancel", "name": "Ravi Shah", "phone": "9876543210", "date": "", "time": ""}
    },
    {
        "scenario": "cancel with a reason, no details",
        "history": [],
        "message": "please cancel my booking, I'm feeling better now",
        "expected": {"action": "cancel", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "cancel over two turns",
        "history": [
            ("user", "cancel my appointment"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "Amit Desai 9898989898",
        "expected": {"action": "cancel", "name": "Amit Desai", "phone": "9898989898", "date": "", "time": ""}
    },
    {
        "scenario": "hinglish",
        "history": [],
        "message": "appointment cancel kar do, Neha Joshi 9988776655",
        "expected": {"action": "cancel", "name": "Neha Joshi", "phone": "9988776655", "date": "", "time": ""}
    },
    {
        "scenario": "cancel without the word, +91 with dashes",
        "history": [],
        "message": "I don't need the doctor anymore, remove my slot. Karan Singh +91-90000-12345",
        "expected": {"action": "cancel", "name": "Karan Singh", "phone": "9000012345", "date": "", "time": ""}
    },
    {
        "scenario": "switch from booking to cancel",
        "history": [
            ("user", "book me tomorrow 10am"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "actually forget it, cancel everything",
        "expected": {"action": "cancel", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "cancel for someone else",
        "history": [],
        "message": "cancel Meera Nair's appointment, number 9876543210",
        "expected": {"action": "cancel", "name": "Meera Nair", "phone": "9876543210", "date": "", "time": ""}
    },
    {
        "scenario": "'call off'",
        "history": [],
        "message": "call off my visit please, Anil Kapoor 9811122233",
        "expected": {"action": "cancel", "name": "Anil Kapoor", "phone": "9811122233", "date": "", "time": ""}
    },

    # ---------- check (8) ----------
    {
        "scenario": "plain check",
        "history": [],
        "message": "When is my appointment? Ravi Shah 9876543210",
        "expected": {"action": "check", "name": "Ravi Shah", "phone": "9876543210", "date": "", "time": ""}
    },
    {
        "scenario": "labelled fields",
        "history": [],
        "message": "do I have any booking? name Priya Mehta phone 98250 12345",
        "expected": {"action": "check", "name": "Priya Mehta", "phone": "9825012345", "date": "", "time": ""}
    },
    {
        "scenario": "check for someone else",
        "history": [],
        "message": "what time is my mom's appointment? Sunita Patel 9123456780",
        "expected": {"action": "check", "name": "Sunita Patel", "phone": "9123456780", "date": "", "time": ""}
    },
    {
        "scenario": "check over two turns",
        "history": [
            ("user", "can you check my appointment?"),
            ("assistant", NAME_PHONE_REPLY)
        ],
        "message": "Neha Joshi, 9988776655",
        "expected": {"action": "check", "name": "Neha Joshi", "phone": "9988776655", "date": "", "time": ""}
    },
    {
        "scenario": "hinglish",
        "history": [],
        "message": "mera appointment kab hai? Amit Desai 9898989898",
        "expected": {"action": "check", "name": "Amit Desai", "phone": "9898989898", "date": "", "time": ""}
    },
    {
        "scenario": "confirm, not book",
        "history": [],
        "message": "just confirming my slot is still on, Karan Singh 9000012345",
        "expected": {"action": "check", "name": "Karan Singh", "phone": "9000012345", "date": "", "time": ""}
    },
    {
        "scenario": "check, no details",
        "history": [],
        "message": "did my booking go through?",
        "expected": {"action": "check", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "indirect check",
        "history": [],
        "message": "remind me what time I'm coming in, Meera Nair 9876543210",
        "expected": {"action": "check", "name": "Meera Nair", "phone": "9876543210", "date": "", "time": ""}
    },

    # ---------- none: sent here by triage, nothing asked yet (10) ----------
    {
        "scenario": "urgent symptom from triage",
        "history": [],
        "message": "I've had a fever of 102 for four days now",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "urgent symptom from triage",
        "history": [],
        "message": "burning when I pee and lower back pain since yesterday",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "symptom with 'yesterday' (not a date request)",
        "history": [],
        "message": "twisted my ankle yesterday, it's very swollen",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "routine symptom from triage",
        "history": [],
        "message": "runny nose and mild sore throat since yesterday",
        "urgency": "routine",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "answer to triage follow-up",
        "history": [
            ("user", "my ear hurts"),
            ("assistant", "How long has it hurt, and is there any discharge or fever?")
        ],
        "message": "3 days, some yellow discharge",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "urgent symptom from triage",
        "history": [],
        "message": "tooth pain for a week and my jaw is swollen",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "routine symptom from triage",
        "history": [],
        "message": "mild acne on my face for a few months",
        "urgency": "routine",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "routine symptom from triage",
        "history": [],
        "message": "my knee aches a bit after running",
        "urgency": "routine",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "urgent symptom from triage",
        "history": [],
        "message": "vomiting and loose motions for 2 days",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "third person symptom from triage",
        "history": [],
        "message": "my child has had a fever for 3 days and is drinking less",
        "urgency": "urgent",
        "expected": {"action": "none", "name": "", "phone": "", "date": "", "time": ""}
    },
    {
        "scenario": "invalid time repeated after the slot list",
        "history": [
            ("user", "book tomorrow 7:45 am, Ravi Shah 9876543210"),
            ("assistant", "Appointments are every 30 minutes between 8:00 and 22:00. Free slots:\n"
                          f"{(TODAY + timedelta(days=1)).strftime('%a %d %b')}: 08:00 AM, 08:30 AM, 09:00 AM")
        ],
        "message": "7:45 AM",
        "expected": {"action": "book", "name": "Ravi Shah", "phone": "9876543210", "date": day(1), "time": "07:45"}
    },
    {
        "scenario": "time not in the offered list",
        "history": [
            ("user", "I need an appointment tomorrow"),
            ("assistant", SLOTS_REPLY)
        ],
        "message": "Neha Joshi, 9988776655, 6:15 pm",
        "expected": {"action": "book", "name": "Neha Joshi", "phone": "9988776655", "date": day(1), "time": "18:15"}
    }
]
