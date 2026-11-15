# Development History

How the router and triage design changed from v1 to v4, and what each change did to the results. The [README](README.md) describes the final version (v4).

| Version | Decision model | Change |
|---|---|---|
| v1 | gpt-4o-mini | First version. The router had a `self_care` label that sent home-care questions straight to the Knowledge Agent; triage had `emergency`, `urgent`, `routine` and `needs_more_info` |
| v2 | gpt-4o-mini | Router and triage split: the router picks the workflow, triage picks the care level |
| v3 | gpt-4o-mini | Wider `emergency` definition in triage for time-critical problems without dramatic signs |
| v4 | Jev | Jev makes the router and triage decisions; gpt-4o-mini still writes all text |

---

## v1 → v2: router and triage split

`self_care` was used both as a router intent (send to the Knowledge Agent) and, implicitly, as a care outcome. A message such as "I have a mild cold, what can I do at home?" is both a symptom and a self-care request, so the router had to pick between two overlapping labels. Loosening the router let serious symptoms phrased as home-care questions skip triage (an HIV result with a low CD4 count, late emergency contraception); tightening it sent mild cases to an appointment, because triage had no self-care level. Prompt wording could only move this trade-off, not remove it.

| Change | v1 | v2 |
|---|---|---|
| Router labels | `symptom` / `self_care` / `booking` / `other` | `symptom` / `health_info` / `booking` / `other`; any current health problem is `symptom`, even with "what can I do at home?" |
| Triage levels | `emergency` / `urgent` / `routine` / `needs_more_info` | adds `self_care`, routed to the Knowledge Agent; never chosen when the user asks for a doctor |
| Knowledge Agent from triage | — | searches with the user's messages about the problem and opens with a fixed "manage at home for now, book if it gets worse" line |
| CARE-Bench `0B` mapping | router `self_care` | triage `self_care` or router `health_info` |

Effect on CARE-Bench: missed care fell from 8/74 to 3/74 and under-escalation from 18 to 13, because symptoms could no longer skip triage. The HIV result with a low CD4 count, which v1 sent to self-care, was fixed. Urgent false negatives did not improve (8/26 → 9/26): all of them had reached triage and were rated `urgent`, so the cause was the `emergency` definition, not routing.

The router labels were checked against the new rule after the first v2 run: 4 home-care questions about a problem the user has now (mild sunburn, blocked nose, food during loose motions, gym with a cold) were relabelled from `health_info` to `symptom`, and the router score went from 44/50 to 48/50 on the same predictions.

## v2 → v3: emergency definition

`emergency` now also covers problems that are time-critical without severe pain: a sudden change in vision, new numbness or weakness, a child who is unusually drowsy or hard to wake, bleeding or strong pain in pregnancy, and a new or different symptom in someone with a serious past heart or brain condition. Five internal emergency cases were added for these, worded so no red-flag keyword matches (triage set 50 → 55 cases).

Effect on CARE-Bench: urgent false negatives fell from 9/26 to 5/26 and under-escalation from 13 to 9, with accuracy and macro-F1 up. The 5 left were all rated `urgent`: a child falling asleep after sudden vertigo, flashes of light in one eye, and dull side pain in pregnancy.

Held-out status: this change was prompted by the v2 urgent false negatives, so CARE-Bench is not fully held-out from v3 on. The wording uses general categories rather than the case details. CARE-Bench was frozen after the v3 run and is reported as a safety stress test from then on.

The triage labels were also checked against the v3 level definitions: "can barely keep water down" and high blood sugar with thirst need same-day care, which `urgent` (24–48 hours) is too slow for, so they became `emergency`; a small painless wart became `self_care`.

## v3 → v4: Jev

Jev (TypeSafe AI's decision model, `typesafe/jev-1.13` through OpenRouter) answers one `choice` question per decision, using the same label and level definitions as the gpt-4o-mini prompts, and returns a probability for each option.

- Router: if `symptom` has 0.30 or more without being the top label, the message still goes to triage. Red-flag keyword messages are routed by Jev too, so prevention questions such as "how do I lower my risk of a heart attack?" can reach `health_info`; triage keeps the keyword filter.
- Triage: Jev's top choice is the level. Jev does not write text, so gpt-4o-mini writes the follow-up questions.
- Failure: an error or 10-second timeout falls back to gpt-4o-mini. An earlier run had a 30-second timeout fall back to a fixed `urgent`, which is why the fallback became a real LLM decision.
- Router label change: "ok can I just see a doctor about it then?" after a cold answer was relabelled from `booking` to `symptom` (a current problem plus a request for a doctor, the v2 rule), so v3 scores 47/50 on the current labels.

Two Jev experiments were tried and dropped: a 0.30 threshold for `emergency` in triage caused over-triage, and a scored variant of the triage question dropped triage accuracy to 47%.

---

## Results by version

### Internal test sets

| Agent | v1 | v2 | v3 | v4 (Jev) |
|---|---|---|---|---|
| Router | 48/50 (96.0%) | 48/50 (96.0%) | 47/50 (94.0%) | 49/50 (98.0%) |
| Triage | 46/50 (92.0%) | 46/50 (92.0%) | 52/55 (94.5%) | 52/55 (94.5%) |

Each router score is on the labels of its own version: v1 had `self_care` instead of `health_info`, and v3 and v4 are on the current labels. Triage v1 and v2 used 50 cases (before the 5 time-critical emergency cases were added).

Per label, v3 against v4:

| Router label | v3 | v4 (Jev) |
|---|---|---|
| symptom | 19/20 | 20/20 |
| health_info | 6/7 | 7/7 |
| booking | 11/11 | 11/11 |
| other | 11/12 | 11/12 |

| Triage level | v3 | v4 (Jev) |
|---|---|---|
| emergency | 20/21 | 21/21 |
| urgent | 11/11 | 9/11 |
| routine | 3/4 | 4/4 |
| self_care | 7/8 | 7/8 |
| needs_more_info | 11/11 | 11/11 |
| Under-triage / over-triage | 2 / 1 | 1 / 2 |

v3 known failures: the heart-bypass case → `urgent` instead of `emergency`; mild acne for months → `self_care` instead of `routine`; the heart-attack prevention question → `symptom`, because the keyword set `symptom` before the LLM call. All three pass with Jev.

### CARE-Bench (router + triage, 135 prefixes)

| Metric | v1 | v2 | v3 | v4 (Jev) | v4, 3 runs |
|---|---|---|---|---|---|
| Urgent false negatives (`1B` not sent to `1B`) | 8/26 (30.8%) | 9/26 (34.6%) | 5/26 (19.2%) | 4/26 (15.4%) | 4/26 in all 3 |
| Missed care (`1A`/`1B` answered `0A`/`0B`) | 8/74 (10.8%) | 3/74 (4.1%) | 3/74 (4.1%) | 5/74 (6.8%) | 3–5/74 |
| Over-escalation | 53/135 (39.3%) | 58/135 (43.0%) | 56/135 (41.5%) | 54/135 (40.0%) | 53–54 |
| Under-escalation | 18/135 (13.3%) | 13/135 (9.6%) | 9/135 (6.7%) | 9/135 (6.7%) | 7–9 |
| `0A` recall (follow-up questions) | 8/37 | 2/37 | 3/37 | 8/37 | — |
| Accuracy | 64/135 (47.4%) | 64/135 (47.4%) | 70/135 (51.9%) | 72/135 (53.3%) | 53.3–54.8% |
| Macro-F1 | 42.2 | 40.5 | 45.4 | 48.6 | 48.6–49.7 |

Missed care moved to 3–5/74 with Jev because it asks a follow-up question more often, which CARE-Bench counts as `0A`; in two cases it asked instead of deciding, including the child who fell asleep after sudden vertigo. Predictions for every version are in `evaluation/care_bench/care_bench_predictions_<version>.csv`.

### Latency (seconds per call, one run of each eval)

| Agent | Version | Average | Median | p95 | Max |
|---|---|---|---|---|---|
| Router (50 cases) | v3, gpt-4o-mini | 0.90 | 0.78 | 1.43 | 4.07 |
| Router (50 cases) | v4, Jev | 0.69 | 0.62 | 0.97 | 1.34 |
| Triage (55 cases) | v3, gpt-4o-mini | 1.01 | 0.98 | 1.40 | 2.19 |
| Triage (55 cases) | v4, Jev | 0.86 | 0.71 | 1.62 | 2.03 |
| Router + Triage, CARE-Bench (135 prefixes) | v3, gpt-4o-mini | 1.51 | 1.72 | 2.64 | 3.07 |
| Router + Triage, CARE-Bench (135 prefixes) | v4, Jev | 1.61 | 1.65 | 2.78 | 5.51 |

Jev cut the median decision time by about 20% in the router and 27% in triage. End to end the gain is small, because follow-up questions are still written by gpt-4o-mini, and the CARE-Bench median moved between 1.20 and 1.65 seconds across the three v4 runs. With gpt-4o-mini, red-flag keyword cases skip the router's LLM call, which pulls its averages down.
