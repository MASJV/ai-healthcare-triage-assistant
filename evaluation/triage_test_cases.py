"""
triage_test_cases.py

55 test cases for the Triage Agent.
Each case has:
    scenario   - short tag of what the case checks
    history    - earlier messages as (role, text); role is "user" or "assistant"
    message    - the latest user message triage has to judge
    follow_ups - follow-up questions already asked in this session
                 (at MAX_FOLLOW_UPS triage must decide, it cannot ask again)
    expected   - the correct level: emergency / urgent / routine / self_care / needs_more_info
"""

HEAD_INJURY_QUESTION = "Did he pass out, vomit, or seem confused after the fall?"


TEST_CASES = [

    # ---------- emergency (21) ----------
    {
        "scenario": "heart attack signs, no red-flag keyword",
        "history": [],
        "message": "my father has a crushing pressure in his chest that is spreading to his left arm",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "breathing trouble, no red-flag keyword",
        "history": [],
        "message": "I can't catch my breath even sitting still and my lips look bluish",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "stroke signs, no red-flag keyword",
        "history": [],
        "message": "my wife's face looks droopy on one side and she can't lift her right arm",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "poisoning, no red-flag keyword",
        "history": [],
        "message": "my 3 year old drank some toilet cleaner by mistake",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "infant fever with danger signs",
        "history": [],
        "message": "my 2 month old baby has a fever of 101, is very floppy and not feeding",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "bleeding, no red-flag keyword",
        "history": [],
        "message": "cut my hand on broken glass and the blood is spurting out",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "sudden worst headache",
        "history": [],
        "message": "the worst headache of my life came on suddenly a few minutes ago",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "collapsed, no red-flag keyword",
        "history": [],
        "message": "my grandfather collapsed and isn't responding when we shake him",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "meningitis signs",
        "history": [],
        "message": "stiff neck, high fever and a rash that doesn't fade when I press a glass on it",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "severe abdominal pain with vomiting",
        "history": [],
        "message": "severe pain in my lower right belly, I can't stand straight and I keep vomiting",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "red-flag keyword",
        "history": [],
        "message": "she had a seizure and is still shaking",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "pill ingestion, no red-flag keyword",
        "history": [],
        "message": "I took a whole strip of sleeping pills an hour ago",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "hinglish chest pain",
        "history": [],
        "message": "papa ko seene mein bahut tez dard ho raha hai aur bahut paseena aa raha hai",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "answer to follow-up turns it into an emergency",
        "history": [
            ("user", "my son fell from the stairs and hit his head"),
            ("assistant", HEAD_INJURY_QUESTION)
        ],
        "message": "yes he threw up twice and he is very sleepy",
        "follow_ups": 1,
        "expected": "emergency"
    },
    {
        "scenario": "sudden vision change, no red-flag keyword",
        "history": [],
        "message": "since this morning a dark shadow like a curtain is covering part of the vision in my right eye",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "one-sided numbness that is easing",
        "history": [],
        "message": "my father's left hand and the left side of his face went numb an hour ago, he says it is getting better now",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "drowsy child, no dramatic signs",
        "history": [],
        "message": "my 3 year old has had loose motions for two days and today he is very drowsy and hard to wake up",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "bleeding in pregnancy",
        "history": [],
        "message": "I'm 32 weeks pregnant, I've been bleeding for an hour and have strong pain in my belly",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "new symptom after past heart surgery",
        "history": [],
        "message": "I had a heart bypass two years ago and today I get a tightness when I walk that feels different from before",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "cannot keep fluids down, needs same-day care",
        "history": [],
        "message": "vomiting and loose motions for 2 days, I can barely keep water down",
        "follow_ups": 0,
        "expected": "emergency"
    },
    {
        "scenario": "high blood sugar with symptoms, needs same-day care",
        "history": [],
        "message": "I'm diabetic, my sugar has been above 350 for two days and I'm very thirsty",
        "follow_ups": 0,
        "expected": "emergency"
    },

    # ---------- urgent (11) ----------
    {
        "scenario": "fever not settling",
        "history": [],
        "message": "I've had a fever of 102 for four days now and it's not coming down",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "urine infection spreading",
        "history": [],
        "message": "burning when I pee and pain in my lower back since yesterday, I feel feverish",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "ear infection",
        "history": [],
        "message": "my ear has been hurting for 3 days and now there's yellow discharge coming out",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "infected wound",
        "history": [],
        "message": "a cut on my leg from last week is now red, swollen, warm and has pus",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "possible fracture",
        "history": [],
        "message": "twisted my ankle yesterday, it's very swollen and I can't put any weight on it",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "asthma getting worse",
        "history": [],
        "message": "my inhaler isn't helping as much this week and I'm wheezing more at night",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "painful red eye",
        "history": [],
        "message": "my eye is red and painful since this morning and bright light hurts it",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "dental abscess",
        "history": [],
        "message": "tooth pain for a week and now my jaw is swollen",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "hinglish fever and vomiting",
        "history": [],
        "message": "teen din se tez bukhar hai aur ulti ho rahi hai",
        "follow_ups": 0,
        "expected": "urgent"
    },
    {
        "scenario": "answer to follow-up (child fever)",
        "history": [
            ("user", "my daughter has a fever"),
            ("assistant", "How high is the fever and how long has she had it? Is she drinking and passing urine normally?")
        ],
        "message": "it's 103 for 3 days now, she's drinking less than usual",
        "follow_ups": 1,
        "expected": "urgent"
    },
    {
        "scenario": "follow-up limit reached, must decide",
        "history": [
            ("user", "I have a cough"),
            ("assistant", "How long have you had it, and do you have fever or trouble breathing?"),
            ("user", "about 2 weeks, mild fever at night"),
            ("assistant", "Are you coughing anything up, and have you lost weight recently?")
        ],
        "message": "some yellow phlegm, not sure about the weight",
        "follow_ups": 2,
        "expected": "urgent"
    },

    # ---------- routine (4) ----------
    {
        "scenario": "medicine refill",
        "history": [],
        "message": "I need a follow-up for my blood pressure medicine refill, I feel fine",
        "follow_ups": 0,
        "expected": "routine"
    },
    {
        "scenario": "long-standing skin issue",
        "history": [],
        "message": "mild acne on my face for a few months",
        "follow_ups": 0,
        "expected": "routine"
    },
    {
        "scenario": "check-up",
        "history": [],
        "message": "I want a general health check-up, no problems right now",
        "follow_ups": 0,
        "expected": "routine"
    },
    {
        "scenario": "scalp issue",
        "history": [],
        "message": "dandruff and an itchy scalp for a few weeks",
        "follow_ups": 0,
        "expected": "routine"
    },

    # ---------- self_care (8) ----------
    {
        "scenario": "common cold",
        "history": [],
        "message": "runny nose and a mild sore throat since yesterday",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "minor ache",
        "history": [],
        "message": "my knee aches a little after running but it goes away with rest",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "occasional heartburn",
        "history": [],
        "message": "I get mild heartburn after spicy food sometimes",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "typos",
        "history": [],
        "message": "mild hedache after long screen time in the evenings",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "seasonal allergy",
        "history": [],
        "message": "sneezing and itchy eyes every morning this season",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "trap: red-flag word, harmless",
        "history": [],
        "message": "I accidentally swallowed my chewing gum",
        "follow_ups": 0,
        "expected": "self_care"
    },
    {
        "scenario": "answer to follow-up rules out danger",
        "history": [
            ("user", "my son hit his head on the table"),
            ("assistant", HEAD_INJURY_QUESTION)
        ],
        "message": "no, he cried for a minute and is playing normally, just a small bump",
        "follow_ups": 1,
        "expected": "self_care"
    },
    {
        "scenario": "wart",
        "history": [],
        "message": "small painless wart on my finger",
        "follow_ups": 0,
        "expected": "self_care"
    },

    # ---------- needs_more_info (11) ----------
    {
        "scenario": "vague head injury",
        "history": [],
        "message": "he hit his head",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague pain",
        "history": [],
        "message": "I have a pain",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague stomach",
        "history": [],
        "message": "my stomach hurts",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague baby",
        "history": [],
        "message": "my baby is not well",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague feeling",
        "history": [],
        "message": "I feel weird",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague fall",
        "history": [],
        "message": "my mother fell down",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "fever, no details",
        "history": [],
        "message": "I have a fever",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "one word",
        "history": [],
        "message": "rash",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague eye",
        "history": [],
        "message": "something is wrong with my eye",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "vague chest",
        "history": [],
        "message": "my chest feels a bit off",
        "follow_ups": 0,
        "expected": "needs_more_info"
    },
    {
        "scenario": "one word",
        "history": [],
        "message": "dizzy",
        "follow_ups": 0,
        "expected": "needs_more_info"
    }
]
