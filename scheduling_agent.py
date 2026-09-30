"""
scheduling_agent.py

Books, reschedules and cancels appointments in a SQLite database.

The LLM only reads the conversation and fills in: action, name, phone,
date, time. Every check (slot free, opening hours, booking window,
48 hours for urgent cases) and every database change is done in plain Python.
"""

import re
import sqlite3
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage

import config
from state import UserState, format_history


# every slot is stored as one text value, e.g. "2026-09-29 10:30"
# this format sorts correctly as text, so slots can be compared with < and >
SLOT_FORMAT = "%Y-%m-%d %H:%M"


#Database Setup
def get_connection():
    """
    Opens the database and makes sure the slots table exists.
    One row = one 30-minute slot.
    """
    connection = sqlite3.connect(config.DB_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS slots (
            slot TEXT PRIMARY KEY,
            status TEXT DEFAULT 'free',
            patient_name TEXT,
            phone TEXT,
            urgency TEXT
        )
    """)
    return connection


def create_slots(connection):
    """
    Adds every slot for the next BOOKING_DAYS days.
    INSERT OR IGNORE skips slots that already exist, so booked
    slots are never touched and this is safe to run every time.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for day in range(config.BOOKING_DAYS + 1):
        slot = today + timedelta(days=day, hours=config.OPEN_HOUR)
        closing = today + timedelta(days=day, hours=config.CLOSE_HOUR)

        while slot < closing:
            connection.execute(
                "INSERT OR IGNORE INTO slots (slot) VALUES (?)",
                (slot.strftime(SLOT_FORMAT),)
            )
            slot = slot + timedelta(minutes=config.SLOT_MINUTES)

    connection.commit()


#Booking Rules
def get_booking_window(urgency):
    """
    Returns (now, latest) as slot text.
    Urgent cases must be seen within URGENT_HOURS, everyone else
    within BOOKING_DAYS.
    """
    now = datetime.now()

    if urgency == "urgent":
        latest = now + timedelta(hours=config.URGENT_HOURS)
    else:
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        latest = today + timedelta(days=config.BOOKING_DAYS + 1)

    return now.strftime(SLOT_FORMAT), latest.strftime(SLOT_FORMAT)


def check_slot(connection, slot, urgency):
    """
    Returns None if the slot can be booked, otherwise the reason why not.
    """
    now, latest = get_booking_window(urgency)

    if slot <= now:
        return "That time has already passed."

    if slot > latest:
        if urgency == "urgent":
            return f"Urgent cases must be seen within {config.URGENT_HOURS} hours."
        return f"Bookings are open only for the next {config.BOOKING_DAYS} days."

    row = connection.execute(
        "SELECT status FROM slots WHERE slot = ?",
        (slot,)
    ).fetchone()

    if row is None:
        return (
            f"Appointments are every {config.SLOT_MINUTES} minutes between "
            f"{config.OPEN_HOUR}:00 and {config.CLOSE_HOUR}:00."
        )

    if row[0] != "free":
        return "That slot is already booked."

    return None


def find_free_slots(connection, urgency, date=""):
    """
    Returns the free slots the user is allowed to book:
    - if a date (YYYY-MM-DD) is given, every free slot on that date
    - otherwise every free slot in the next SHOW_HOURS hours
    """
    now, latest = get_booking_window(urgency)

    if not date:
        next_hours = datetime.now() + timedelta(hours=config.SHOW_HOURS)
        # slot text compares correctly, so min() picks the earlier time
        latest = min(latest, next_hours.strftime(SLOT_FORMAT))

    query = "SELECT slot FROM slots WHERE status = 'free' AND slot > ? AND slot <= ?"
    values = [now, latest]

    if date:
        query = query + " AND slot LIKE ?"
        values.append(f"{date}%")

    query = query + " ORDER BY slot"

    rows = connection.execute(query, values).fetchall()

    slots = []
    for row in rows:
        slots.append(row[0])

    return slots


def find_first_free_slot(connection, urgency):
    """
    Returns the earliest free slot in the whole booking window, or None.
    """
    now, latest = get_booking_window(urgency)

    row = connection.execute(
        "SELECT slot FROM slots WHERE status = 'free' AND slot > ? AND slot <= ? ORDER BY slot",
        (now, latest)
    ).fetchone()

    if row is None:
        return None

    return row[0]


def find_booking(connection, name, phone):
    """
    Returns the patient's upcoming booked slot, or None.
    Name + phone together identify the patient.
    """
    now = datetime.now().strftime(SLOT_FORMAT)

    row = connection.execute(
        """
        SELECT slot FROM slots
        WHERE status = 'booked'
        AND LOWER(patient_name) = LOWER(?)
        AND phone = ?
        AND slot > ?
        ORDER BY slot
        """,
        (name, phone, now)
    ).fetchone()

    if row is None:
        return None

    return row[0]


#Database Changes
def book_slot(connection, slot, name, phone, urgency):

    connection.execute(
        """
        UPDATE slots SET status = 'booked', patient_name = ?, phone = ?, urgency = ?
        WHERE slot = ?
        """,
        (name, phone, urgency, slot)
    )
    connection.commit()


def free_slot(connection, slot):

    connection.execute(
        """
        UPDATE slots SET status = 'free', patient_name = NULL, phone = NULL, urgency = NULL
        WHERE slot = ?
        """,
        (slot,)
    )
    connection.commit()


#Input Cleanup
def make_slot(date, time):
    """
    "2026-09-29", "9:30" -> "2026-09-29 09:30"
    Returns "" if the date or time is not valid.
    """
    try:
        slot = datetime.strptime(f"{date} {time}", SLOT_FORMAT)
    except ValueError:
        return ""

    return slot.strftime(SLOT_FORMAT)


def clean_phone(phone):
    """
    Keeps only the digits of a phone number.
    "+91 98765 43210" -> "9876543210"
    Returns "" if it is not a 10-digit number.
    """
    digits = re.sub(r"\D", "", phone)

    # drop the India country code
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]

    if len(digits) != 10:
        return ""

    return digits


#Reply Helpers
def format_slot(slot):
    """
    "2026-09-29 10:30" -> "Tue 29 Sep, 10:30 AM"
    """
    return datetime.strptime(slot, SLOT_FORMAT).strftime("%a %d %b, %I:%M %p")


def list_free_slots(connection, urgency, date=""):
    """
    Free slots grouped by day, one line per day:
        Tue 29 Sep: 08:00 AM, 08:30 AM, 09:00 AM
    """
    slots = find_free_slots(connection, urgency, date)

    # nothing in the next 24 hours or on that date:
    # show the whole day of the earliest free slot instead
    if not slots:
        first_slot = find_first_free_slot(connection, urgency)
        if first_slot is None:
            return "No free slots found."
        slots = find_free_slots(connection, urgency, first_slot[:10])

    days = {}
    for slot in slots:
        slot_time = datetime.strptime(slot, SLOT_FORMAT)
        day = slot_time.strftime("%a %d %b")
        if day not in days:
            days[day] = []
        days[day].append(slot_time.strftime("%I:%M %p"))

    lines = []
    for day in days:
        lines.append(f"{day}: {', '.join(days[day])}")

    return "\n".join(lines)


#Understanding the Request
def read_request(state):
    """
    Asks the LLM to pull out action, name, phone, date and time from the
    conversation. It only fills in these fields; it never books anything.
    """
    now = datetime.now()

    # the next 7 days with weekday names, so the LLM does not have to
    # work out which date "Saturday" is
    day_lines = []
    for offset in range(1, 8):
        day = now + timedelta(days=offset)
        day_lines.append(day.strftime("%A %Y-%m-%d"))
    next_days = ", ".join(day_lines)

    prompt = f"""
    Conversation so far:
    {format_history(state["messages"][:-1])}

    Latest User Message: {state['user_input']}

    Today is {now.strftime("%A, %Y-%m-%d")} and the time is {now.strftime("%H:%M")}.
    Next 7 days: {next_days}

    You help with doctor appointments. From the conversation, fill in:
    Action - what the user wants now (if they changed their mind, the latest
             request wins):
             book       - a new appointment, even if asked as a question
                          or without a date yet
             reschedule - move an appointment that is already booked
             cancel     - cancel an appointment that is already booked
             check      - ask about an appointment that is already booked
                          (its date or time, or whether it exists)
             none       - the user has not asked for an appointment yet.
                          Describing symptoms, however serious, is none.
             Changing the date or time while still booking is still book.
    Name   - the patient's full name as the user wrote it, anywhere in the
             conversation, or none
    Phone  - the patient's phone number as the user wrote it, or none
    Date   - the date the user wants, as YYYY-MM-DD, or none
             (turn words like "tomorrow" or "next Monday" into a date)
    Time   - the time the user wants, as HH:MM in 24-hour format, or none

    For reschedule, Date and Time are the NEW date and time, never the date
    or time of the old appointment.
    The user may write in Hinglish: aaj = today, kal = tomorrow,
    parso = day after tomorrow, subah = morning, dopahar = afternoon,
    shaam = evening, raat = night, baje = o'clock.

    Use only what the user said. Do not guess any field:
    - Never use today's date or the current time unless the user asked for them.
    - A part of the day or a range of days is not an exact time or date,
      so that field is none.
    - You do not know the date or time of an existing appointment, so if the
      user refers to it instead of giving a new one, that field is none.

    Reply in exactly this format:
    Action: <action>
    Name: <name>
    Phone: <phone>
    Date: <date>
    Time: <time>
    """

    response = config.llm.invoke(prompt)

    request = {"action": "none", "name": "", "phone": "", "date": "", "time": ""}

    for line in response.content.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if value.lower() == "none":
            value = ""

        if key == "action":
            # keep "none" for action instead of an empty string
            if value:
                request["action"] = value.lower()
        elif key in request:
            request[key] = value

    # triage sent this message here in this turn (urgency is reset every turn):
    # a symptom message read as "book" without a date and time is not a
    # booking request yet, so treat it as none and offer the free slots
    # (with the 48-hour advice for urgent cases)
    from_triage = state.get("urgency", "") in ["urgent", "routine"]
    if from_triage and request["action"] == "book":
        if not request["date"] or not request["time"]:
            request["action"] = "none"

    return request


#Scheduling Agent
def scheduling_agent(state: UserState):

    # urgency comes from triage in the same turn; keep it for later turns
    # (e.g. when the user replies with their name)
    urgency = state.get("urgency", "")
    if urgency in ["urgent", "routine"]:
        booking_urgency = urgency
    elif state.get("booking_urgency"):
        booking_urgency = state["booking_urgency"]
    else:
        booking_urgency = "routine"

    connection = get_connection()
    create_slots(connection)

    # if every urgent slot is taken, allow routine slots instead of refusing
    window = booking_urgency
    urgent_note = ""
    if booking_urgency == "urgent" and find_first_free_slot(connection, "urgent") is None:
        window = "routine"
        urgent_note = (
            f"No slots are free in the next {config.URGENT_HOURS} hours, so these are "
            "the earliest regular slots. If symptoms get worse, go to the nearest hospital.\n"
        )

    request = read_request(state)
    action = request["action"]
    name = request["name"]
    phone = clean_phone(request["phone"])
    date = request["date"]
    slot = make_slot(request["date"], request["time"])

    # what is still missing to identify the patient
    missing = []
    if len(name.split()) < 2:
        missing.append("full name (first and last name)")
    if not phone:
        missing.append("10-digit phone number")

    # --- Coming from triage, or nothing asked yet: offer slots ---
    if action not in ["book", "reschedule", "cancel", "check"]:
        if booking_urgency == "urgent" and not urgent_note:
            intro = f"You should see a doctor within {config.URGENT_HOURS} hours. Free slots:"
        else:
            intro = "I can book an appointment for you. Free slots:"
        reply = (
            f"{urgent_note}{intro}\n{list_free_slots(connection, window)}\n\n"
            "Reply with the patient's full name, phone number and the slot you want."
        )

    # --- Every action needs full name and phone ---
    elif missing:
        reply = f"Please tell me the patient's {' and '.join(missing)}."

    # --- Check ---
    elif action == "check":
        booking = find_booking(connection, name, phone)
        if booking:
            reply = f"{name}, your appointment is on {format_slot(booking)}."
        else:
            reply = f"No upcoming appointment found for {name} with that phone number."

    # --- Cancel ---
    elif action == "cancel":
        booking = find_booking(connection, name, phone)
        if booking:
            free_slot(connection, booking)
            reply = f"Cancelled {name}'s appointment on {format_slot(booking)}."
        else:
            reply = f"No upcoming appointment found for {name} with that phone number."

    # --- Book or Reschedule ---
    else:
        existing = find_booking(connection, name, phone)

        if action == "book" and existing:
            reply = (
                f"{name} already has an appointment on {format_slot(existing)}. "
                "Would you like to reschedule it?"
            )
        elif action == "reschedule" and not existing:
            reply = f"No upcoming appointment found for {name} to reschedule."
        elif not slot:
            reply = (
                f"{urgent_note}Which slot would you like?\n"
                f"{list_free_slots(connection, window, date)}"
            )
        else:
            problem = check_slot(connection, slot, window)

            if problem:
                reply = (
                    f"{urgent_note}{problem} Free slots:\n"
                    f"{list_free_slots(connection, window, date)}"
                )
            else:
                if existing:
                    free_slot(connection, existing)
                book_slot(connection, slot, name, phone, booking_urgency)

                if existing:
                    reply = f"Rescheduled: {name} moved to {format_slot(slot)}."
                else:
                    reply = f"Booked: {name} on {format_slot(slot)}."

                # booking done, forget the saved urgency
                booking_urgency = ""

    connection.close()

    return {
        "booking_urgency": booking_urgency,
        "result": reply,
        "messages": [AIMessage(content=reply)]
    }
