# Symptom Triage & Appointment Scheduler

A multi-agent healthcare assistant built with LangGraph. It reads a user's message, decides what they need, and hands it to the right agent:

- **Triage**: rates symptoms as emergency, urgent, routine or self-care, and asks follow-up questions when a message is too vague
- **Escalation**: gives a fixed emergency response (call 108/112 plus first-aid steps) with no LLM-written advice
- **Knowledge**: answers general health questions, and gives self-care advice when triage finds a problem can be managed at home, from public health documents using hybrid RAG, with citations
- **Scheduling**: books, reschedules, cancels and checks appointments in a SQLite database

> **Disclaimer:** This assistant provides general health information and appointment support only. It is not a medical device, does not diagnose, and is not a substitute for professional medical advice. In an emergency, call 112 (emergency) or 108 (ambulance).

---

## Architecture

```mermaid
flowchart TD
    START([User message]) --> R[Router Agent]
    R -->|symptom| T[Triage Agent]
    R -->|health_info| K[Knowledge Agent]
    R -->|booking| S[Scheduling Agent]
    R -->|other| E1([End: redirect reply])

    T -->|emergency| X[Escalation Agent]
    T -->|urgent / routine| S
    T -->|self_care| K
    T -->|needs_more_info| Q[Ask follow-up question]
    Q -.->|user's answer, next turn| R

    K --> END([End])
    S --> END
    X --> END
```

The router decides which workflow a message needs; triage decides the level of care. Any current health problem goes through triage, even when the user asks "what can I do at home?", so only triage can decide that home care is enough. Questions with no current problem (prevention, diet, how an illness works) go straight to the Knowledge Agent.

All agents share one state (`UserState`). A LangGraph `MemorySaver` checkpointer keeps the conversation for each chat session (one `thread_id` per run), so follow-up answers such as "yes, he vomited" are read together with the earlier messages.

---

## Agents

### Router Agent (`router_agent.py`)
- The LLM labels the latest message using the conversation history:
  - `symptom`: a current symptom, injury or health problem of a specific person, even if the user asks for home care or a doctor
  - `health_info`: a general health question with no current problem (prevention, diet, how an illness works), or a question about a health answer just given
  - `booking`: book, reschedule, cancel or check an appointment, with no new health problem
  - `other`: anything else
- **Rule-based check first:** the message is checked against the triage red-flag keywords before the LLM call. If one matches, the intent is `symptom` with no LLM call, so a message like "severe chest pain, need appointment" always goes through triage instead of straight to booking.
- `other` gets a short reply explaining what the assistant can help with.

### Triage Agent (`triage_agent.py`)
Two layers:
1. **Keyword filter (no LLM):** a list of red-flag phrases (chest pain, can't breathe, unconscious, seizure, stroke signs, heavy bleeding, overdose, suicide, and more). A match is marked `emergency` immediately. Matching uses a start-of-word regex, so `fits` does not match inside `benefits`, while `seizure` still matches `seizures`.
2. **LLM reasoning:** anything the keywords do not catch is classified as `emergency` (life-threatening, or time-critical without severe pain, such as a sudden change in vision or new numbness), `urgent` (see a doctor within 24–48 hours), `routine` (a normal appointment) or `self_care` (mild, can be managed at home and watched), with a one-line reason. When unsure, the prompt asks for the more serious level, and `self_care` is never chosen when the user asks to see a doctor.

`emergency` goes to the Escalation Agent, `urgent` and `routine` to the Scheduling Agent, and `self_care` to the Knowledge Agent. For `self_care`, the knowledge search uses all of the user's messages about the problem (the first message plus the answers to follow-up questions), and the answer starts with a fixed line that the problem can usually be managed at home and an appointment can be booked if it gets worse.

If a message is too vague to judge safely (for example "he hit his head"), triage returns `needs_more_info` and asks one follow-up question about the relevant danger signs. It asks at most `MAX_FOLLOW_UPS` (2) questions before it must decide. An unreadable LLM reply defaults to `urgent`.

### Escalation Agent (`escalation_agent.py`)
Returns a fixed safety response:
1. Call 108 (ambulance) or 112 (emergency); go straight to the emergency department if a hospital is close and the person can be moved safely.
2. First-aid steps for the detected emergency type, read word for word from the `FIRST_AID` list. There are 12 types: bleeding, chest pain, choking, breathing difficulty, unconscious, seizure, stroke, burn, head injury, poisoning/overdose, snake bite and suicide risk (including Tele-MANAS, 14416).
3. A closing line that the assistant cannot replace emergency services.

The emergency type is found from all of the user's messages in the session in two ways:
- **Keywords**, the same way as triage
- **LLM category matching**: the model may only return category names from the fixed list (for example "losing a lot of blood" → `bleeding`). Unknown names are ignored, and if the LLM call fails, keyword matches are used alone.

**The LLM never writes the advice.** Several types can match, and all their steps are shown. If nothing matches, general steps are used.

### Knowledge Agent (`knowledge_agent.py`)
Query expansion, then hybrid retrieval over the health-document collection:

```
question ─> + up to 5 medical terms from the LLM (e.g. "piles" -> hemorrhoids)
search query ─┬─> vector search (Chroma, text-embedding-3-small, 1536 dims) ─┐
              └─> keyword search (BM25)                                     ─┴─> Reciprocal Rank Fusion (k = 60)
                  ─> BGE cross-encoder reranker (BAAI/bge-reranker-base, runs locally)
                  ─> top 4 chunks ─> LLM answer to the original question, with [1]–[4] citations
```

- Query expansion keeps the user's words and adds medical terms, so everyday wording still finds the right documents. If that LLM call fails, the original question is searched alone.
- The prompt restricts the model to facts stated in the retrieved context and to information that answers the question. It must cite what it uses and say when the documents do not cover the question.
- It gives no diagnosis, no prescription medicines and no dose of any medicine (over-the-counter included), and it replaces UK emergency numbers with Indian ones (112/108).
- A check in code backs up the prompt: any sentence that still contains a medicine dose (tablets, capsules, drops, or mg/ml next to a medicine name) is removed and replaced with a line to ask a doctor or pharmacist. Food and drink amounts such as "2,000 ml of water" are left alone.
- Every answer ends with a when-to-see-a-doctor line, a fixed disclaimer added in code, and the source titles with URLs.

### Scheduling Agent (`scheduling_agent.py`)
- The LLM only extracts five fields from the conversation: action (book / reschedule / cancel / check / none), full name, phone, date and time. It gets today's date and the dates of the next 7 days, so it can turn words like "tomorrow" or "Friday" into real dates.
- When triage sends a message here in the same turn and no date and time have been given yet, the action is set to `none` in code, so the patient is shown the free slots (with the 48-hour advice for urgent cases) instead of being asked for their details first.
- Every rule check and database change is done in Python:

| Rule | Value (`config.py`) |
|---|---|
| Opening hours | 8:00 AM – 10:00 PM, same for urgent and routine |
| Slot length | 30 minutes |
| Booking window | next 30 days |
| Urgent cases | must be booked within 48 hours; if all of those slots are taken, regular slots are offered with advice to go to the nearest hospital if symptoms worsen |
| Patient identity | full name (first and last) + 10-digit phone (spaces and +91 are removed) |
| Bookings per patient | one upcoming booking; a second request is offered a reschedule |
| Slots shown | every free slot in the next 24 hours, or on the requested date, grouped by day |

Database: one table, `slots`, with one row per 30-minute slot.

| Column | Example |
|---|---|
| `slot` (primary key) | `2026-09-29 10:30` |
| `status` | `free` / `booked` |
| `patient_name` | `Ravi Shah` |
| `phone` | `9876543210` |
| `urgency` | `urgent` / `routine` |

Slots for the next 30 days are created automatically with `INSERT OR IGNORE`, so existing bookings are never overwritten. The slot format sorts correctly as text, so time checks are plain string comparisons.

---

## Knowledge Base

The self-care corpus comes from public health sources:

| Source | Content used |
|---|---|
| MedlinePlus (U.S. National Library of Medicine) | Health topic summaries |
| Self-Care Forum fact sheets | Common ailments, self-care steps and when to seek help |
| *Where There Is No Doctor* (Hesperian, 2025) | Selected chapters: home care, first aid, prevention, common illnesses |

- Documents were converted to Markdown with `title`, `source` and `url` front matter.
- The dosing reference section of *Where There Is No Doctor* was excluded on purpose.
- No LLM-written content is part of the corpus.

**The corpus is not included in this repository.** To build the vector store, place the Markdown files in `health_docs_clean/health_docs_clean/` and run `python ingest.py`.

`ingest.py`:
1. reads each file and its front matter
2. splits on `##` headings so sections stay together where possible, then uses a 2,000-character sliding window (250 overlap) for long sections
3. puts the document title at the start of every chunk
4. embeds chunks 100 at a time and stores them in a local Chroma collection (`chroma_db/`)

---

## Project Structure

```
symptom-triage-scheduler/
├── app.py                 # Streamlit web app
├── symptomTriage.py       # LangGraph graph + command-line chat loop
├── state.py               # shared UserState + conversation history helper
├── config.py              # all settings and the shared LLM
├── router_agent.py        # intent routing + red-flag check
├── triage_agent.py        # keyword filter + LLM triage + follow-up questions
├── escalation_agent.py    # fixed emergency response
├── knowledge_agent.py     # hybrid RAG answers with citations
├── scheduling_agent.py    # SQLite appointment booking
├── ingest.py              # builds the Chroma vector store
├── eval_router.py         # runs the router evaluation and prints accuracy
├── eval_triage.py         # runs the triage evaluation and prints accuracy
├── eval_scheduling.py     # runs the scheduling evaluation and prints accuracy
├── eval_knowledge.py      # runs the knowledge evaluation (Python checks + DeepEval)
├── eval_care_bench.py     # external validation on CARE-Bench Public Test 1
├── evaluation/
│   ├── router_test_cases.py   # 50 router test cases
│   ├── triage_test_cases.py   # 55 triage test cases
│   ├── scheduling_test_cases.py   # 50 scheduling test cases
│   ├── knowledge_test_cases.py    # 50 knowledge test cases
│   ├── latency.py                 # latency summary printed by the evals
│   ├── care_bench_public_test.jsonl   # CARE-Bench Public Test 1, 135 prefixes (external validation)
│   └── care_bench_predictions_v1.csv … v3.csv  # predictions per version (prefix_id, predicted_label)
├── assets/
│   └── triage_bg.png      # web app background
├── .streamlit/
│   └── config.toml        # dark theme for the web app
├── requirements.txt
└── .env.example
```

Created at runtime (git-ignored): `chroma_db/`, `appointments.db`, and `.deepeval/` (DeepEval's local run data). `.env` is created by you from `.env.example` and is also git-ignored.

---

## Setup

**Requirements:** Python 3.11 or newer (developed on 3.13) and an OpenAI API key.

```bash
git clone https://github.com/MASJV/symptom-triage-scheduler.git
cd symptom-triage-scheduler

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Create `.env` from the example and add your key:

```
OPENAI_API_KEY=your-openai-api-key
```

LangSmith tracing is optional. To turn it on, also add:

```
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=symptom-triage-scheduler
```

Build the knowledge base (once, and again whenever the documents change):

```bash
python ingest.py
```

Start the web app:

```bash
streamlit run app.py
```

Or chat in the terminal:

```bash
python symptomTriage.py
```

Type `exit` to quit; the graph structure is printed on exit.

The web app shows the reply (emergencies in red, follow-up questions in yellow), the routing and triage details, and the conversation in the sidebar. "New conversation" starts a fresh session.

> The first run downloads the BGE reranker model (about 1 GB).

---

## Example Session

Booking and emergency replies are fixed text; the triage reason and knowledge answers are written by the LLM and will vary.

```
Enter your message: he hit his head
Intent: symptom
Urgency: needs_more_info - ...
Did he pass out, vomit, or seem confused?

Enter your message: yes he vomited and his head is bleeding
Intent: symptom
Urgency: emergency - ...
This may be a medical emergency.
- Call 108 for an ambulance or 112 for emergency help right now.
- If a hospital is very close and the person can be moved safely, take them straight to the emergency department.

While waiting for help:
1. Press firmly on the wound with a clean cloth and keep pressing.
...

Enter your message: book an appointment tomorrow
Intent: booking
Please tell me the patient's full name (first and last name) and 10-digit phone number.
```

---

## Configuration

Settings live in `config.py`:

| Group | Settings |
|---|---|
| Models | `CHAT_MODEL` (gpt-4o-mini), `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `RERANKER_MODEL`, `JUDGE_MODEL` (gpt-4.1-mini, evaluation only) |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Retrieval | `TOP_K_VECTOR`, `TOP_K_BM25`, `TOP_K_FINAL`, `RRF_K` |
| Triage | `MAX_FOLLOW_UPS` |
| Scheduling | `DB_PATH`, `BOOKING_DAYS`, `URGENT_HOURS`, `OPEN_HOUR`, `CLOSE_HOUR`, `SLOT_MINUTES`, `SHOW_HOURS` |
| Storage | `DOCS_FOLDER`, `CHROMA_DIRECTORY`, `COLLECTION_NAME` |

---

## Safety Design

| Risk | How it is handled |
|---|---|
| An emergency is rated too low | Keyword filter marks red flags as `emergency` before any LLM call; when unsure, triage picks the more serious level; an unreadable reply defaults to `urgent` |
| An emergency message is labelled as a booking | The router checks every message against the red-flag keywords first and sends matches to triage |
| A symptom phrased as a home-care question skips triage | The router sends every current health problem to triage; only triage can choose `self_care`, and never when the user asks for a doctor |
| A vague symptom is rated from too little information | Triage asks up to 2 follow-up questions about danger signs before deciding |
| The LLM writes wrong emergency advice | Emergency steps are fixed text from `FIRST_AID`; the LLM only picks category names, and unknown names are dropped |
| Self-care answers go beyond the sources | The prompt allows only facts from the retrieved documents, with citations; no prescription medicines and no medicine doses, even when the documents contain one |
| Answers are read as medical advice | Every knowledge answer ends with a fixed disclaimer and a when-to-see-a-doctor line |
| Urgent patients get no appointment | If no slot is free within 48 hours, the earliest regular slots are offered with advice to go to a hospital if symptoms worsen |

---

## Evaluation

The four agent test sets below are internal (development) sets: prompts were improved against them, so the scores are optimistic. The held-out check is the CARE-Bench external validation at the end of this section.

### Router Agent

`eval_router.py` runs the real `router_agent()` (red-flag check + LLM label) on 50 test cases in `evaluation/router_test_cases.py` and prints overall and per-label accuracy, plus every wrong case.

| Label | Cases | What they cover |
|---|---|---|
| symptom | 19 | third person, typos, Hinglish, mental health, answers to triage follow-up questions, new symptom after a booking, a current problem with a home-care question (small cut, sunburn, loose motions, blocked nose, gym with a cold) |
| health_info | 7 | general wellness, prevention, how an illness works, diet for a diagnosed condition, follow-up on a health answer |
| booking | 12 | book, reschedule, cancel, check, answers to name/phone/slot questions, switching from triage to cancel |
| other | 12 | greetings, coding, pets, plants, devices, "book a flight", "cancel Netflix", prompt injection |

```bash
python eval_router.py
```

Result with gpt-4o-mini, v3 (router unchanged since v2): **48/50 (96.0%)** (v1, before the router labels changed from `self_care` to `health_info`: 48/50)

| Label | Accuracy |
|---|---|
| symptom | 19/19 (100%) |
| health_info | 6/7 (85.7%) |
| booking | 12/12 (100%) |
| other | 11/12 (91.7%) |

The v2 run first scored 44/50 against labels that still marked 4 home-care questions about a current problem ("How do I treat a mild sunburn at home?", home remedies for a blocked nose, food during loose motions, gym with a cold) as `health_info`. Under the v2 rule those describe a problem the user has now and belong to triage, which is what the router predicted; the labels were corrected, and a fresh run gave 48/50.

Known failures:
- "What are some ways to lower my risk of a heart attack?" → `symptom`. The red-flag check is kept on purpose: sending a prevention question to triage is safer than missing an emergency.
- "My dog has been vomiting since morning" → `symptom` instead of `other`.

### Triage Agent

`eval_triage.py` runs the real `triage_agent()` (keyword filter + LLM) on 55 test cases in `evaluation/triage_test_cases.py`. Each case sets how many follow-up questions were already asked, so the `MAX_FOLLOW_UPS` limit is tested too. Besides accuracy it counts under-triage (level lower than expected) and over-triage (level higher than expected).

| Level | Cases | What they cover |
|---|---|---|
| emergency | 21 | heart attack, stroke, poisoning, bleeding and breathing signs worded so no red-flag keyword matches (20 of 21), infant fever, Hinglish, a follow-up answer that turns a case into an emergency, time-critical cases without dramatic signs (sudden vision change, one-sided numbness that is easing, a drowsy child, bleeding in pregnancy, a new symptom after heart surgery), cases that need same-day care (cannot keep fluids down, high blood sugar with symptoms) |
| urgent | 11 | infections, possible fracture, worsening asthma, painful red eye, dental abscess, answers to follow-up questions, follow-up limit reached |
| routine | 4 | refill, check-up, acne for months, dandruff for weeks |
| self_care | 8 | cold, minor aches, heartburn, seasonal allergy, wart, a follow-up answer that rules out danger, "swallowed my chewing gum" |
| needs_more_info | 11 | vague messages such as "he hit his head", "I feel weird", "rash" |

```bash
python eval_triage.py
```

Result with gpt-4o-mini, v3: **52/55 (94.5%)**, under-triage 2, over-triage 1

| Level | Accuracy |
|---|---|
| emergency | 20/21 (95.2%) |
| urgent | 11/11 (100%) |
| routine | 3/4 (75.0%) |
| self_care | 7/8 (87.5%) |
| needs_more_info | 11/11 (100%) |

The v3 run first scored 49/55 against older labels. Three were corrected against the v3 level definitions and the same predictions rescored: "can barely keep water down" and high blood sugar with thirst need same-day care, which the `urgent` level (24–48 hours) is too slow for, so they are `emergency`; a small painless wart can be treated at home, so it is `self_care`. Earlier runs: v1 46/50 (92.0%, before triage had a `self_care` level), v2 46/50 (92.0%, before the 5 time-critical emergency cases were added).

Known failures:
- "I had a heart bypass two years ago and today I get a tightness when I walk that feels different from before" → `urgent` instead of `emergency`.
- Mild acne for months → `self_care` instead of `routine` (under-triage, low risk).
- "I accidentally swallowed my chewing gum" → `emergency`. The keyword "swallowed" forces it; kept on purpose, like the router's red-flag check.

### Scheduling Agent

`eval_scheduling.py` runs the LLM step of the scheduling agent (`read_request()`) on 50 test cases in `evaluation/scheduling_test_cases.py`. Only the field extraction is tested, so `appointments.db` is never read or changed; the booking rules are plain Python. A case passes only if all five fields (action, name, phone, date, time) are right. Phones are cleaned with `clean_phone()` and times written as HH:MM before comparing, the same way the agent does before booking. Dates are worked out from the day the test runs, so "tomorrow" always means the real tomorrow, and no case asks for a time later today, so the result does not depend on the hour.

| Action | Cases | What they cover |
|---|---|---|
| book | 16 | relative and written dates, 12-hour times, times in words, Hinglish, phones with +91/spaces/dashes, vague times that must not be guessed, picking an offered slot, correcting the time |
| reschedule | 8 | "prepone", old and new day in one message, only the time changing, Hinglish |
| cancel | 8 | cancel without the word, switching from booking to cancel, cancelling for someone else |
| check | 8 | indirect questions, Hinglish, check without details |
| none | 10 | symptom messages sent on by triage before any appointment request |

```bash
python eval_scheduling.py
```

Result with gpt-4o-mini: **48/50 (96.0%)**

| Field | Accuracy |
|---|---|
| action | 50/50 (100%) |
| name | 50/50 (100%) |
| phone | 50/50 (100%) |
| date | 49/50 (98%) |
| time | 49/50 (98%) |

The 10 `none` cases carry the urgency triage would have given, as in the app, so the from-triage rule in `read_request()` is part of what is tested.

Known failures:
- "reschedule it to the same time tomorrow" → the current time is filled in. Harmless in practice: the check rejects a time that is not a 30-minute slot or has passed.
- "change my booking time to 9 in the morning, same day" → a date is guessed.

### Knowledge Agent

`eval_knowledge.py` runs the knowledge agent on 50 test cases in `evaluation/knowledge_test_cases.py`.

| Type | Cases | Passes when |
|---|---|---|
| answer | 34 | a source title matches the topic (retrieval hit), the answer cites a source, and both DeepEval scores below are at least 0.7 |
| no_info | 8 | the answer says the documents do not cover it (local hospitals, prices, brand comparisons, Ayurvedic treatment) |
| no_dose | 8 | the answer gives no medicine dose (paracetamol for a child, antibiotics, aspirin, melatonin) |

The answer cases are also graded with DeepEval, using a stronger model than the agent as the judge (`JUDGE_MODEL`, gpt-4.1-mini). The retrieved chunks come from calling `retrieve()` directly, and only the answer text is graded, not the fixed disclaimer and source list.

| Metric | Measures |
|---|---|
| Faithfulness | every claim in the answer is supported by the retrieved chunks |
| Answer relevancy | the answer addresses the question |

A score below 0.7 fails the case, and the judge's reason is printed with it. DeepEval costs judge calls, so `RUN_DEEPEVAL = False` at the top of `eval_knowledge.py` runs only the Python checks (answer cases then pass on retrieval hit and citation alone).

```bash
python eval_knowledge.py
```

Result with gpt-4o-mini: **46/50 (92.0%)**

| Type | Accuracy |
|---|---|
| answer | 32/34 (94.1%) |
| no_info | 8/8 (100%) |
| no_dose | 6/8 (75.0%) |

Retrieval hit 34/34 (100%), cited sources 33/34 (97.1%); faithfulness average 0.96, answer relevancy average 0.94.

Known failures:
- "How do I treat a mild sunburn at home?" → correct treatment advice plus prevention tips; answer relevancy 0.57. The treatment part is still useful for a real user.
- "Is it okay to go to the gym when I have a cold?" → said the documents do not cover it, although the cold documents advise rest until you feel better; answer relevancy 0.40, no citation.
- Paracetamol for a 3-year-old and aspirin for chest pain → gave a dose. Fixed after this run by the dose check in code; both passed on re-check.

### External validation (CARE-Bench)

`eval_care_bench.py` runs CARE-Bench Public Test 1 (62 cases, 135 labelled prefixes) from `evaluation/care_bench_public_test.jsonl`. The cases are patient messages in everyday, first-person language, 1 to 3 rounds each, built from real medical dialogue sources and human-reviewed. They are used for testing only; prompts are not changed to fit them.

Why CARE-Bench: no public dataset matches this app's workflows exactly. CARE-Bench is the closest fit: patient-written messages in everyday language over 1 to 3 rounds, with an "ask for more information" label that tests follow-up questions. Semigran vignettes (clinician-written, one message, 15 cases per class) and RepVig (vignette texts not released) were checked and fit worse.

Each prefix is the patient's messages up to one point in the conversation. As in the official CARE-Bench inputs, the rounds are joined into one message (the benchmark has no assistant turns between them) and sent through the app's path: Router Agent, then Triage Agent for symptoms. Triage may still ask a follow-up in rounds 1 and 2 (`follow_ups` = earlier rounds, at most `MAX_FOLLOW_UPS`).

| CARE-Bench label | Meaning | App output counted as this label |
|---|---|---|
| `0A` | ask for more information | triage `needs_more_info`, or router `other` (the app asks what the symptom is) |
| `0B` | self-care or monitoring | triage `self_care`, or router `health_info` |
| `1A` | non-urgent professional care | triage `routine` or `urgent`, or router `booking` |
| `1B` | urgent or emergency care now | triage `emergency` |

Triage `urgent` (doctor within 24–48 hours) counts as `1A`, since `1B` means care now.

Metrics are computed the same way as the official CARE-Bench scorer: accuracy, macro-F1, under- and over-escalation, missed care, and urgent false negatives (`1B` cases not sent to `1B`), with every missed `1B` case listed. Predictions are saved to `evaluation/care_bench_predictions_<RUN_NAME>.csv`, so a run before and after a change can be compared. Set `RUN_NAME` at the top of the file.

```bash
python eval_care_bench.py
```

Results with gpt-4o-mini, scope: Router Agent + Triage Agent only. v1: prompts unchanged, `0B` was router `self_care` only. v2: router `health_info` + triage `self_care`. v3: v2 + wider triage `emergency` definition (current version).

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Urgent false negatives (`1B` not sent to `1B`) | 8/26 (30.8%) | 9/26 (34.6%) | 5/26 (19.2%) |
| Missed care (`1A`/`1B` answered `0A`/`0B`) | 8/74 (10.8%) | 3/74 (4.1%) | 3/74 (4.1%) |
| Over-escalation | 53/135 (39.3%) | 58/135 (43.0%) | 56/135 (41.5%) |
| Under-escalation | 18/135 (13.3%) | 13/135 (9.6%) | 9/135 (6.7%) |
| Accuracy | 64/135 (47.4%) | 64/135 (47.4%) | 70/135 (51.9%) |
| Macro-F1 | 42.2 | 40.5 | 45.4 |

Comparison with the models published in the CARE-Bench repository on the same 135 prefixes, in the unprompted setting (patient text only, no CARE-Bench instructions), which is the closest match to this system:

| System | Accuracy | Macro-F1 | Missed care | Urgent false negatives |
|---|---|---|---|---|
| This system, v1 (router + triage, gpt-4o-mini) | 47.4 | 42.2 | 10.8 | 30.8 |
| This system, v2 | 47.4 | 40.5 | 4.1 | 34.6 |
| This system, v3 | 51.9 | 45.4 | 4.1 | 19.2 |
| GPT-5.5 | 45.9 | 45.0 | 29.7 | 23.1 |
| Claude Opus 4.7 | 49.6 | 50.2 | 37.8 | 19.2 |
| Range of the 11 published models | 34.8–54.1 | 33.8–54.0 | 6.8–82.4 | 7.7–80.8 |

The published models' free-text replies were mapped to labels by an LLM; this system's fixed outputs are mapped directly, so the comparison is close but not identical.

What the results show:
- Missed care fell from 8/74 in v1 to 3/74 in v2 and v3, lower than all 11 published models: symptoms can no longer skip triage, and triage prefers the more serious level when unsure.
- Urgent false negatives fell from 8/26 (v1) to 5/26 (v3), level with Claude Opus 4.7 and better than GPT-5.5. v2 alone did not change them (9/26): all of them had reached triage and were rated `urgent`, so the cause was the `emergency` definition, not routing. The 5 left are all rated `urgent`: a child falling asleep after sudden vertigo, flashes of light in one eye, and dull side pain in pregnancy.
- The cost is over-escalation (56/135 in v3): most detailed first messages get `urgent` instead of a follow-up question (`0A` recall 3/37), and most `0B` cases still go to `urgent` (`0B` recall 8/24).
- v2 fixed the HIV result with a low CD4 count, which v1 sent to `self_care`. Questions worded as information about something the user took (emergency contraception, a drug interaction before a procedure) still go to `health_info`.
- Held-out status: the v3 `emergency` definition was widened after seeing the v2 urgent false negatives, so CARE-Bench is not fully held-out for v3. The wording uses general categories rather than the case details, and 5 internal cases were added to check it.

Why accuracy and macro-F1 stay low, and why they are not the target:
- `0A` (ask for more information) is 37 of the 135 prefixes, and CARE-Bench expects a question even on detailed first messages. This app asks a follow-up only when a message is too vague to judge safely; otherwise it rates the message. That is a different policy, so most `0A` cases are counted wrong by design.
- Over-escalation comes from "if unsure, choose the more serious level". This is kept on purpose: for a triage tool, sending someone to a doctor they did not need is the cheaper mistake.
- Raising accuracy would mean tuning prompts against the benchmark, which would make it an internal set. It would also mean loosening the safety rule above.

The target is safety: someone who needs care should not be told they can wait or manage at home. So CARE-Bench is reported as an external safety stress test. The metrics that match that target are missed care, under-escalation and urgent false negatives; accuracy and macro-F1 are shown next to them for comparison. CARE-Bench is not tuned further after v3.

Known limits: the source dialogues are not Indian patients, the benchmark was built with GPT-5.5 help and then human review, and the data is CC BY-NC 4.0 (non-commercial use only).

Source: *CARE-Bench: Benchmarking Patient-Facing LLM Triage*, 2026. [arXiv:2608.03731](https://arxiv.org/abs/2608.03731) · [GitHub](https://github.com/ningkko/CARE-bench) · [Hugging Face dataset](https://huggingface.co/datasets/ningkko/CARE-Bench)

---

## Tracing and Latency

With LangSmith turned on (see Setup), every chat turn is traced as one run named `symptom-triage-scheduler`. LangGraph records each agent node and LangChain records each LLM call automatically; the steps outside LangChain are wrapped with `@traceable`:

| File | Traced steps |
|---|---|
| `triage_agent.py` | `checkRedFlags` |
| `knowledge_agent.py` | `expandQuery`, `vectorSearch`, `bm25Search`, `fuseResults`, `rerankChunks`, `retrieve`, `removeDoses` |
| `escalation_agent.py` | `findCategoriesByKeywords`, `findCategoriesByLlm`, `findFirstAidSteps` |
| `scheduling_agent.py` | `readRequest`, `listFreeSlots` |

Latency is measured in code, so it works without LangSmith:
- `eval_router.py`, `eval_triage.py` and `eval_care_bench.py` time every call and print the average, median, 95th percentile and slowest call (`evaluation/latency.py`). These are the numbers to compare before and after a change such as JeV.
- The web app shows the time for each message under Triage Details (`latency_seconds`).

Measured latency, v3 with gpt-4o-mini (seconds per call, one run of each internal eval):

| Agent | Average | Median | p95 | Max |
|---|---|---|---|---|
| Router (50 cases) | 0.90 | 0.78 | 1.43 | 4.07 |
| Triage (55 cases) | 1.01 | 0.98 | 1.40 | 2.19 |
| Router + Triage, CARE-Bench (135 prefixes) | 1.51 | 1.72 | 2.64 | 3.07 |

Cases that match a red-flag keyword skip the LLM call, which pulls the averages down. The max includes the first call of the run. The CARE-Bench row is the full path a symptom message takes (router, then triage when the router says `symptom`).

## Cost

Prices used: **gpt-4o-mini** $0.15 per 1M input tokens and $0.60 per 1M output tokens; **text-embedding-3-small** $0.02 per 1M tokens. The BGE reranker runs locally and costs nothing. Check current prices on the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing) before relying on these figures.

**LLM calls per message**

| Path | Calls |
|---|---|
| Router | 0 if a red-flag keyword matches, otherwise 1 chat call |
| Triage | 0 if a red-flag keyword matches, otherwise 1 |
| Escalation | 1 small chat call (category matching) |
| Knowledge | 1 small chat call (query expansion) + 1 embedding call + 1 chat call |
| Scheduling | 1 chat call |

**Estimates** (from prompt sizes, not measured):

| Item | Approximate cost |
|---|---|
| One-time ingestion (~2.9 MB of text, about 0.8–0.9M tokens) | under $0.02 |
| Typical message (2–3 calls; the knowledge prompt with 4 chunks is the largest at ~3,000 input tokens) | about $0.0003–$0.001 |
| 1,000 messages | about $0.30–$1.00 |

Prompts include the conversation history, so cost per message grows slowly in long sessions.

`eval_knowledge.py` also calls the judge model (gpt-4.1-mini) for the DeepEval scores and prints the judge cost at the end of each run.

---

## Data & Privacy

- Appointments, including patient names and phone numbers, are stored in a local SQLite file, `appointments.db`. It is git-ignored and never committed.
- Conversation memory lives in the running process only (`MemorySaver`), one thread per chat session or browser session, and is cleared when the process stops.
- User messages and retrieved document text are sent to the OpenAI API to generate replies.
- When LangSmith tracing is on, the same inputs and outputs are also sent to LangSmith. It is off unless `LANGSMITH_TRACING=true` is set.
- The API key is read from `.env`, which is git-ignored; `.env.example` holds only a placeholder.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Knowledge base not found. Run python ingest.py first.` | Run `python ingest.py` after placing the documents in `health_docs_clean/health_docs_clean/` |
| `sqlite3.OperationalError: table slots has no column named slot` | `appointments.db` was created by an older version; delete it and it is recreated on the next run |
| Slow first start | The BGE reranker model (about 1 GB) downloads on first use |
| `openai.AuthenticationError` | Check `OPENAI_API_KEY` in `.env` |
| Slot times look shifted | Slots follow the computer's local clock; check the system time zone |

---

## Tech Stack

| Area | Tools |
|---|---|
| Orchestration | LangGraph (StateGraph, conditional edges, MemorySaver) |
| UI | Streamlit |
| LLM | OpenAI gpt-4o-mini via LangChain |
| Embeddings | OpenAI text-embedding-3-small (1536 dimensions) |
| Vector store | ChromaDB (local, persistent) |
| Keyword search | rank-bm25 (BM25Okapi) |
| Reranker | BAAI/bge-reranker-base (sentence-transformers, local) |
| Appointments | SQLite |
| Evaluation | DeepEval (gpt-4.1-mini as judge) |
| Tracing | LangSmith (optional) |
