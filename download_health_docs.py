"""
One-shot downloader for the self-care RAG corpus.
Setup:     pip install requests beautifulsoup4
           put  NHS_API_KEY=your_key  in a .env file next to this script (add .env to .gitignore)
Run:       python download_health_docs.py              # everything
           python download_health_docs.py --nhs-test   # NHS: only 5 pages, to check output first
Output:    health_docs/{medlineplus,selfcareforum,hesperian_wtnd,nhs}/
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

OUT = Path("health_docs")
HEADERS = {"User-Agent": "Mozilla/5.0 (health-rag-corpus; personal study project)"}
S = requests.Session()
S.headers.update(HEADERS)


def save(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip  {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = S.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"saved {dest.name} ({len(r.content) // 1024} KB)")
    time.sleep(0.5)


# 1) MedlinePlus: newest compressed health-topic XML (auto-detected from the listing page)
def medlineplus() -> None:
    html = S.get("https://medlineplus.gov/xml.html", timeout=60).text
    zips = re.findall(r'href="([^"]*mplus_topics_compressed_[\d-]+\.zip)"', html)
    if not zips:
        raise RuntimeError("MedlinePlus zip link not found")
    url = urljoin("https://medlineplus.gov/xml.html", zips[0])  # first = newest
    save(url, OUT / "medlineplus" / Path(url).name)


# 2) Self-Care Forum fact sheets: each page links one PDF
SCF_PAGES = [
    "lower-back-pain", "eczema", "heartburn-and-indigestion", "fever-in-children",
    "constipation-in-adults", "headache-and-migraine", "cough-in-adults", "acne-spots",
    "sprains-and-strains", "sore-throat", "middle-ear-infection-in-children",
    "common-cold-in-adults", "acute-sinusitis-in-adults", "urine-symptoms-in-men",
    "venous-disease-looking-after-your-legs-2", "boosting-your-mood", "menopause",
    "urinary-incontinence-leakage-in-adults", "long-covid-and-post-covid-syndrome-in-adults",
    "the-power-of-self-care", "conjunctivitis-pink-red-eye", "vitamin-d", "folic-acid",
    "strep-a-scarlet-fever", "winter-wellness", "elementor-10488", "moving-more",
]


def selfcareforum() -> None:
    for slug in SCF_PAGES:
        page = f"https://www.selfcareforum.org/{slug}/"
        try:
            soup = BeautifulSoup(S.get(page, timeout=60).text, "html.parser")
            pdfs = [
                a["href"] for a in soup.find_all("a", href=True)
                if a["href"].lower().endswith(".pdf")
                and "wp-content/uploads" in a["href"]
                and "production-manual" not in a["href"].lower()
            ]
            if not pdfs:
                print(f"MISS  no PDF on {page}")
                continue
            url = urljoin(page, pdfs[0])
            save(url, OUT / "selfcareforum" / f"{slug}.pdf")
        except Exception as e:
            print(f"FAIL  {slug}: {e}")


# 3) Where There Is No Doctor (2025) — self-care-relevant chapters only
WTND = "https://hesperian.org/wp-content/uploads/pdf/en_wtnd_2025/en_wtnd_2025_{}.pdf"
WTND_CHAPTERS = {
    "01": "home_cures", "02": "sicknesses_often_confused", "04": "caring_for_sick_person",
    "05": "healing_without_medicines", "06": "right_wrong_use_of_medicines",
    "07": "antibiotics", "10": "first_aid", "11": "nutrition", "12": "prevention",
    "13": "common_sicknesses", "15": "skin_problems", "16": "eyes",
    "17": "teeth_gums_mouth", "18": "urinary_and_genitals", "21": "children",
    "22": "older_people", "gp": "green_pages_medicines",
}


def hesperian() -> None:
    for code, name in WTND_CHAPTERS.items():
        try:
            save(WTND.format(code), OUT / "hesperian_wtnd" / f"wtnd_{code}_{name}.pdf")
        except Exception as e:
            print(f"FAIL  wtnd {code}: {e}")


# 4) NHS Website Content API v2 (Integration test env) -> one .md per page
NHS_BASE = "https://int.api.service.nhs.uk/nhs-website-content"
NHS_SECTIONS = ("conditions", "symptoms", "medicines", "live-well", "mental-health", "womens-health")
SKIP_GENRES = {"Procedure", "Test"}
SELF_CARE_ASPECTS = {"SelfCareHealthAspect", "MedicalHelpNonurgentHealthAspect",
                     "MedicalHelpUrgentHealthAspect", "MedicalHelpEmergencyHealthAspect"}


def load_env() -> None:
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def to_int(url: str) -> str:
    # manifest/pagination may return production URLs; keep every call on the integration env
    return url.replace("https://api.service.nhs.uk/nhs-website-content", NHS_BASE)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(" ", strip=False)).strip()


def walk(node, out: list, depth: int = 0) -> None:
    """Collect (heading, text) from nested mainEntityOfPage / hasPart elements."""
    if isinstance(node, list):
        for n in node:
            walk(n, out, depth)
        return
    if not isinstance(node, dict):
        return
    aspect = node.get("hasHealthAspect")
    if isinstance(aspect, str):
        aspect = aspect.rstrip("/").split("/")[-1]  # e.g. SelfCareHealthAspect
    head = node.get("headline") or node.get("description") if aspect else node.get("headline")
    text = html_to_text(node.get("text", ""))
    if head or text:
        label = f"{head} [{aspect}]" if aspect and head else (head or "")
        out.append((label.strip(), text))
    for key in ("mainEntityOfPage", "hasPart"):
        if key in node:
            walk(node[key], out, depth + 1)


def nhs_page_to_md(data: dict) -> str:
    parts: list = []
    walk(data.get("hasPart") or [], parts)
    walk(data.get("mainEntityOfPage") or [], parts)
    web = data.get("webpage") or data.get("url", "").replace(
        "https://api.service.nhs.uk", "https://www.nhs.uk")
    front = {
        "title": data.get("name", ""),
        "url": web,
        "source": "NHS website (www.nhs.uk)",
        "genre": data.get("genre", []),
        "last_reviewed": (data.get("lastReviewed") or [""])[0] if isinstance(
            data.get("lastReviewed"), list) else data.get("lastReviewed", ""),
    }
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front.items()] + ["---", ""]
    if data.get("description"):
        lines += [data["description"], ""]
    seen = set()
    for head, text in parts:
        if (head, text) in seen or not (head or text):
            continue
        seen.add((head, text))
        if head:
            lines.append(f"## {head}")
        if text:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def nhs(test: bool = False) -> None:
    load_env()
    key = os.getenv("NHS_API_KEY")
    if not key:
        print("SKIP  NHS: set NHS_API_KEY in .env")
        return
    api = requests.Session()
    api.headers.update({"apikey": key, "accept": "application/json", **HEADERS})

    def get(url: str) -> dict:
        for attempt in range(5):
            r = api.get(to_int(url), timeout=60)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            time.sleep(0.55)  # stays under 120 req/min
            return r.json()
        raise RuntimeError(f"rate-limited: {url}")

    done = 0
    next_url = f"{NHS_BASE}/manifest/pages/?page=1"
    while next_url:
        manifest = get(next_url)
        for p in manifest.get("results", []):
            section = p["url"].split("/nhs-website-content/")[-1].split("/")[0]
            if section not in NHS_SECTIONS:
                continue
            # self-care filter: skip procedures/tests, and condition/symptom pages with no self-care or when-to-get-help section
            if set(p.get("genre") or []) & SKIP_GENRES:
                continue
            if section in ("conditions", "symptoms") and not set(p.get("healthAspects") or []) & SELF_CARE_ASPECTS:
                continue
            slug = p["url"].rstrip("/").split("/nhs-website-content/")[-1].replace("/", "__")
            dest = OUT / "nhs" / section / f"{slug}.md"
            if dest.exists():
                continue
            try:
                url = p["url"] + ("?modules=true" if p.get("modules") else "")
                md = nhs_page_to_md(get(url))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(md, encoding="utf-8")
                done += 1
                print(f"saved nhs/{section}/{dest.name}")
            except Exception as e:
                print(f"FAIL  {p['url']}: {e}")
            if test and done >= 5:
                print("NHS test run done: open health_docs/nhs/ and check the .md files")
                return
        next_url = manifest.get("pagination", {}).get("next")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhs-test", action="store_true", help="download only 5 NHS pages")
    args = ap.parse_args()
    if args.nhs_test:
        nhs(test=True)
    else:
        medlineplus()
        selfcareforum()
        hesperian()
        # nhs()  # disabled: NHS integration env is blocked; re-enable once a working key exists
    print("\nDone ->", OUT.resolve())