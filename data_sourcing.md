# Knowledge Base: How the Documents Were Chosen

This note records how the self-care knowledge base for the RAG pipeline was built: what we looked for, what we tried, what we dropped, and why.

## Goal

Answer general self-care health questions (common ailments, home care, when to see a doctor) from trusted documents, with citations. Research papers were out of scope. The content had to be free to reuse and small enough to store and embed locally.

## Sources considered

| Source | Decision | Reason |
|---|---|---|
| MedlinePlus (NIH) | **Used** | Public domain, ~1,000 topics, single XML bulk download |
| Self-Care Forum fact sheets | **Used** | Built for self-care: what to do yourself + red flags for common ailments |
| Where There Is No Doctor (Hesperian, 2025) | **Used** (selected chapters) | Home care, first aid, prevention; relevant to Indian conditions |
| NHS Website Content API | Tried, dropped | See below |
| CDC, WHO fact sheets | Dropped | Mostly epidemiology rather than self-care; no bulk download |
| openFDA / DailyMed drug labels | Dropped | Regulatory text, too dense for self-care questions |
| Mayo Clinic, Cleveland Clinic | Not used | Terms of use don't allow scraping |
| LLM-written summaries | Rejected | No verifiable source; risk of made-up medical facts |

## The NHS attempt

NHS content was the first choice: each condition page has clear "things you can do yourself" and "when to get help" sections. We registered on the NHS developer hub and created an Integration test app. The API key was accepted by the gateway, but the integration environment's backend returned "Access Denied", which is outside our control. Production access requires a formal onboarding review and a signed connection agreement by a legal signatory, which is too heavy for a student project. We dropped NHS, and its download code was later removed from the downloader.

## How the documents were collected

`download_health_docs.py` fetches everything in one run:
- MedlinePlus: the newest compressed health-topic XML, detected automatically from the listing page (the file name changes daily)
- Self-Care Forum: each fact-sheet page is read to find its current PDF link (file names carry version dates)
- Where There Is No Doctor: 16 self-care-relevant chapters. Injections, midwifery, family planning and the medicine kit were skipped.

Raw download: ~31 MB.

## Trimming

`trim_health_docs.py` converts everything to Markdown with `title`, `source` and `url` front matter, and removes only non-content:
- MedlinePlus: Spanish topics, link lists, MeSH codes, site records
- Self-Care Forum: page footers, sponsor line, and the website-name lines under "Where to find out more". The first version cut that whole section, but in two-column PDFs it also held red-flag warnings. It now removes only the link lines, and a line-by-line diff confirmed nothing else was lost.
- Where There Is No Doctor: page headers, page numbers, chapter labels, one postal address
- **Green Pages excluded on purpose:** a prescription-dosing reference (antibiotics, injections, seizure drugs) meant for health workers where no doctor is available. A self-care assistant should not give those doses.

Result: 1,056 files, ~2.9 MB of text.

## Known limitations

- Self-Care Forum sheets use UK services ("call 999 / 111"). The knowledge prompt maps these to Indian equivalents (112 / 108).
- Where There Is No Doctor is written for low-resource settings; some advice assumes no clinic nearby.
- Two-column PDFs give slightly out-of-order text in places. The content is complete.
- The folic acid fact sheet did not download (26 of 27 sheets).
- Gaps remain for over-the-counter medicine questions and mental health.

## Chunking and embeddings(as built)

`ingest.py` splits each document on `##` headings, so sections such as "when to get help" stay whole, and splits long sections with a 2,000-character window and 250-character overlap. The document title is added to the start of every chunk. Embeddings use text-embedding-3-small at its full 1,536 dimensions (512 was the first plan), stored in a local Chroma collection (`chroma_db/`).