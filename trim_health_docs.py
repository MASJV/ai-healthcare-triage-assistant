"""
Trim the downloaded health corpus to clean, source-tagged Markdown for RAG.
Needs: poppler's `pdftotext` on PATH (Windows: install poppler, or use the
already-trimmed output instead of re-running).
Run:   python trim_health_docs.py            (reads health_docs/, writes health_docs_clean/)

What is removed (nothing else is changed):
  MedlinePlus      Spanish topics, link lists, MeSH codes, site records (keeps title, also-called, full summary)
  Self Care Forum  "Where to find out more" heading + website-name lines, page footers, sponsor line
  WTND             running page headers, chapter-number labels, contact addresses for other books
                   Green Pages (prescription dosing tables for health workers) -> excluded, see README note
"""
import html
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SRC = Path("health_docs")
OUT = Path("health_docs_clean")


def front(title: str, source: str, url: str) -> str:
    esc = lambda s: s.replace('"', "'")
    return f'---\ntitle: "{esc(title)}"\nsource: "{source}"\nurl: "{url}"\n---\n\n'


def pdf_text(p: Path) -> str:
    return subprocess.run(["pdftotext", "-enc", "UTF-8", str(p), "-"],
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout


def tidy(text: str) -> str:
    text = text.replace("\f", "\n").replace(" ", " ").replace("�", "•")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# ---------------------------------------------------------------- MedlinePlus
def html_to_text(h: str) -> str:
    h = re.sub(r"<li[^>]*>", "\n- ", h)
    h = re.sub(r"</(p|ul|ol|li|h\d)>", "\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    return html.unescape(h)


def medlineplus() -> int:
    z = next((SRC / "medlineplus").glob("mplus_topics_compressed_*.zip"))
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xml"))
        root = ET.fromstring(zf.read(name))
    out = OUT / "medlineplus"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in root.findall("health-topic"):
        if t.get("language") != "English":
            continue
        fs = t.find("full-summary")
        if fs is None or not (fs.text or "").strip():
            continue
        body = html_to_text(fs.text)
        body = re.sub(r"\n\s*NIH:.*$", "", body.strip())  # trailing institute credit line
        also = [a.text.strip() for a in t.findall("also-called") if a.text]
        md = front(t.get("title"), "MedlinePlus (medlineplus.gov)", t.get("url"))
        md += f"# {t.get('title')}\n\n"
        if also:
            md += f"Also called: {', '.join(also)}\n\n"
        md += body
        slug = re.sub(r"[^a-z0-9]+", "-", t.get("title").lower()).strip("-")
        (out / f"{slug}.md").write_text(tidy(md), encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------- Self Care Forum
# "Where to find out more" is often interleaved with red-flag text (two-column PDFs), so only the
# heading and the pure link/website lines are dropped -- never the surrounding content.
SCF_FOOTER = re.compile(
    r"^(Where to find out more|Where to find more information|Where can I go for more information\?|"
    r"Fact Sheet No.*|(Produced by )?The Self[- ]Care Forum\.? www\.selfcareforum\.org|Produced by the Self Care Forum.*|"
    r".*Next review due.*|Reviewed .*20\d\d.*|Self-Care|Self|Forum|Care Forum|Helping people take care of themselves|"
    r"[.• \t]*|This fact sheet was developed with .*)[ \t]*$",
    re.I | re.M)
SCF_LINK = re.compile(
    r"^[•� \t]*("
    r"NHS[ ]?(111|England|Website|website|Inform|Direct|NI|24)[^.\n]{0,30}|NI Direct[^.\n]{0,40}|Phone First service[^.\n]{0,40}|"
    r"Migraine Trust|Clusterbusters|BASH \(.*\)|The British Association of Dermatologists.*|Legsmatter website|"
    r"Vascular Society for Great Britain and Ireland|Leg Ulcer (support groups|Forum):?|The Lindsay Leg Club Foundation|"
    r"Or,? visit .*(websites?|country)[^.\n]*:?|For coronavirus information go to:|"
    r"For more information go to the Patient\.co\.uk website|here\. or NHS England website here\.|"
    r"For information on Scarlet Fever visit our website\.)[ \t]*$",
    re.M)
SCF_HEAD = re.compile(r"^(Useful facts|What can I expect to happen\?|What can I do.*|Treatment|When to .*)$", re.M)


def selfcareforum() -> int:
    out = OUT / "selfcareforum"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((SRC / "selfcareforum").glob("*.pdf")):
        text = pdf_text(p)
        text = SCF_FOOTER.sub("", text)
        text = SCF_LINK.sub("", text)
        text = SCF_HEAD.sub(lambda mm: f"## {mm.group(1)}" if len(mm.group(1)) < 70 else mm.group(1), text)
        title = next((l.strip() for l in text.splitlines() if l.strip()), p.stem)
        md = front(title, "Self-Care Forum fact sheet", f"https://www.selfcareforum.org/{p.stem}/") + tidy(text)
        (out / f"{p.stem}.md").write_text(md, encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------- Where There Is No Doctor
WTND_URL = "https://hesperian.org/wp-content/uploads/pdf/en_wtnd_2025/en_wtnd_2025_{}.pdf"
WTND_RUNNING = re.compile(r"^(Where There Is No Doctor 2025[ \t]*\d*|\d+[ \t]*Where There Is No Doctor 2025|\d{1,3}|CHAPTER)[ \t]*$",
                          re.M)
WTND_CAPS = re.compile(r"^([A-Z][A-Z ,’'()\-/&]{7,60})$", re.M)
# trailing postal/contact block for other organisations (e.g. end of ch. 21)
WTND_CONTACT = re.compile(r"\n(Or write to:|For more information, write to:).*\Z", re.S)


def wtnd() -> int:
    out = OUT / "hesperian_wtnd"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((SRC / "hesperian_wtnd").glob("wtnd_*.pdf")):
        code = p.stem.split("_")[1]
        if code == "gp":  # Green Pages: prescription dosing reference, excluded on purpose
            continue
        text = pdf_text(p)
        text = WTND_RUNNING.sub("", text)
        text = WTND_CONTACT.sub("\n", text)
        text = WTND_CAPS.sub(lambda m: f"## {m.group(1).title()}", text)
        title = "Where There Is No Doctor — " + " ".join(p.stem.split("_")[2:]).replace("-", " ").title()
        md = front(title, "Where There Is No Doctor (Hesperian, 2025)", WTND_URL.format(code)) + tidy(text)
        (out / f"{p.stem}.md").write_text(md, encoding="utf-8")
        n += 1
    return n


if __name__ == "__main__":
    counts = {"medlineplus": medlineplus(), "selfcareforum": selfcareforum(), "hesperian_wtnd": wtnd()}
    print(counts, "->", OUT.resolve())
