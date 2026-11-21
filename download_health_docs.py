"""
One-shot downloader for the self-care RAG corpus.
Setup:     pip install requests beautifulsoup4
Run:       python download_health_docs.py
Output:    health_docs/{medlineplus,selfcareforum,hesperian_wtnd}/
"""
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


if __name__ == "__main__":
    medlineplus()
    selfcareforum()
    hesperian()
    print("\nDone ->", OUT.resolve())