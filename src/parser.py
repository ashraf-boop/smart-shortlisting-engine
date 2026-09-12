"""
src/parser.py
-------------
Fixes vs. the previous version:
  A1. OCR fallback: if PyMuPDF returns near-empty text (scanned/image PDF),
      fall back to pdf2image + pytesseract rasterization + OCR.
  A2. Phrase-aware tokenization: multi-word / punctuation-bearing skills
      ("Node.js", "C++", "REST API", "CI/CD") are protected with placeholder
      tokens BEFORE punctuation stripping, then restored, so they never get
      shredded into "Node" + "js" or "C" + "++" -> "C".
  A3. Term normalization: synonyms ("ReactJS" -> "react", "Amazon Web
      Services" -> "aws") collapse to one canonical token via
      skills_data.TERM_NORMALIZATION_MAP.
  C2. Section anchors: SECTION_HEADER_ALIASES (including fuzzy ones like
      "My Background") drive section splitting, and skills are tagged with
      the section they were found in for section-weighted scoring later.
"""

import os
import re
import hashlib
import docx
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.skills_data import (
    CANONICAL_SKILLS,
    TERM_NORMALIZATION_MAP,
    MULTI_WORD_PHRASES,
    SECTION_HEADER_ALIASES,
)

# Minimum extracted-character threshold below which we assume the PDF is
# scanned/image-based and trigger the OCR fallback path.
OCR_FALLBACK_CHAR_THRESHOLD = 40

# Optional explicit paths to the Tesseract/Poppler binaries. Leave both as
# None to rely on PATH (works fine on most Linux/Mac setups). On Windows,
# `where`-based PATH lookups can be unreliable even when PATH itself is
# correctly configured — if you hit "tesseract is not installed" or
# "poppler not found" errors despite having both installed and on PATH,
# hardcode the two paths here instead of fighting PATH further, e.g.:
#   TESSERACT_CMD = r"C:\Users\yourname\tesseract-ocr\tesseract.exe"
#   POPPLER_PATH = r"C:\Users\yourname\poppler-26.07.0\Library\bin"
TESSERACT_CMD = r"C:\Users\ashraf\tesseract-ocr\tesseract.exe"
POPPLER_PATH = r"C:\Users\ashraf\poppler-26.07.0\Library\bin"

_HEADER_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z &/]{2,45}:?$")


# ---------------------------------------------------------------------------
# STAGE 1: RAW TEXT EXTRACTION (with OCR fallback)
# ---------------------------------------------------------------------------

def _extract_from_docx(docx_path: str) -> str:
    """Extracts text paragraphs and table contents from Word documents."""
    import docx
    doc = docx.Document(docx_path)
    full_text = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
            
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    full_text.append(cell.text)
                    
    return "\n".join(full_text)

def _extract_from_txt(txt_path: str) -> str:
    """Extracts raw text from .txt files with UTF-8 encoding fallback."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1") as f:
            return f.read()

def _extract_with_pymupdf(pdf_path: str) -> str:
    import fitz  # PyMuPDF
    chunks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" mode (not "blocks"/"dict") gives a reasonable reading
            # order for single/near-single column resumes without us having
            # to hand-roll column detection.
            chunks.append(page.get_text("text"))
    return "\n".join(chunks)


def _extract_with_ocr(pdf_path: str) -> str:
    """
    Fallback for scanned/image-only PDFs. Requires pdf2image (+ poppler) and
    pytesseract (+ tesseract-ocr binary) to be installed. Raises a clear
    error if the OCR stack isn't available rather than silently returning
    empty text.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR fallback needed (near-empty text extracted) but pdf2image/"
            "pytesseract are not installed. Install with: "
            "pip install pdf2image pytesseract --break-system-packages "
            "(plus system packages poppler-utils and tesseract-ocr)."
        ) from exc

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    convert_kwargs = {"dpi": 300}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH # type: ignore[operator]

    images = convert_from_path(pdf_path, **convert_kwargs) # type: ignore[operator]
    ocr_chunks = [pytesseract.image_to_string(img) for img in images]
    return "\n".join(ocr_chunks)


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """
    Returns (text, extraction_method) where extraction_method is
    'pymupdf' or 'ocr', so downstream logging/QA can flag low-confidence
    OCR extractions if needed.
    """
    text = ""
    try:
        text = _extract_with_pymupdf(pdf_path)
    except Exception:
        text = ""

    if len(text.strip()) < OCR_FALLBACK_CHAR_THRESHOLD:
        ocr_text = _extract_with_ocr(pdf_path)
        if len(ocr_text.strip()) > len(text.strip()):
            return ocr_text, "ocr"

    if not text.strip():
        raise ValueError(f"No extractable text found in: {pdf_path}")
    return text, "pymupdf"

def extract_text_from_file(file_path: str) -> Tuple[str, str]:
    """
    Main file-routing entrypoint for .pdf, .docx, and .txt files.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = _extract_from_docx(file_path)
        if not text.strip():
            raise ValueError(f"No extractable text found in DOCX: {file_path}")
        return text, "python-docx"
    elif ext == ".txt":
        text = _extract_from_txt(file_path)
        if not text.strip():
            raise ValueError(f"No extractable text found in TXT: {file_path}")
        return text, "txt_reader"
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file: {file_path}")


# ---------------------------------------------------------------------------
# STAGE 2: WHITESPACE / LINE-BREAK CLEANING
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)          # rejoin hyphen-broken words
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# STAGE 3: PHRASE PROTECTION + TERM NORMALIZATION
# ---------------------------------------------------------------------------
#
# Naive tokenizers split on non-alphanumeric characters, which destroys
# tokens like "Node.js", "C++", "CI/CD", "REST API". We protect these BEFORE
# any punctuation-sensitive step by swapping them for a single safe
# placeholder token (e.g. "node.js" -> "node__js__phrase"), then swap back
# at the end. Longest phrases are matched first (MULTI_WORD_PHRASES is
# pre-sorted longest-first) so "spring boot" is caught before "spring".

def _phrase_to_placeholder(phrase: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return f"__PHRASE_{safe}__"


_PLACEHOLDER_TO_CANONICAL: Dict[str, str] = {}
for _phrase in MULTI_WORD_PHRASES:
    canonical = TERM_NORMALIZATION_MAP.get(_phrase.lower(), _phrase.lower())
    _PLACEHOLDER_TO_CANONICAL[_phrase_to_placeholder(_phrase)] = canonical


def protect_and_normalize(lower_text: str) -> str:
    """
    Input must already be lower-cased. Replaces every occurrence of a known
    multi-word/punctuation-bearing skill phrase with a placeholder token
    that survives punctuation stripping, and simultaneously applies
    synonym normalization (both single- and multi-word synonyms).
    """
    working = lower_text
    for phrase in MULTI_WORD_PHRASES:

        if len(phrase) <= 2 and phrase.isalnum():
            pattern = r"\b" + re.escape(phrase) + r"\b"
        else:
            # For longer or special-character terms ("c++", "node.js", "rest api"), use lookaround boundaries
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"

        placeholder = _phrase_to_placeholder(phrase)
        working = re.sub(pattern, f" {placeholder} ", working)
    return working


def restore_placeholders(text: str) -> str:
    """Swap placeholders back to their canonical (normalized) skill string."""
    for placeholder, canonical in _PLACEHOLDER_TO_CANONICAL.items():
        text = text.replace(placeholder, canonical)
    return text


def build_clean_text(raw_text: str) -> str:
    """
    Produces the text used everywhere downstream for matching: lower-cased,
    phrase-protected, synonym-normalized, punctuation-safe.
    """
    lowered = raw_text.lower()
    protected = protect_and_normalize(lowered)
    restored = restore_placeholders(protected)
    # Collapse the extra spaces introduced by phrase substitution.
    return re.sub(r"\s+", " ", restored).strip()


# ---------------------------------------------------------------------------
# STAGE 4: SECTION SPLITTING (section-anchor aware)
# ---------------------------------------------------------------------------

def split_into_sections(normalized_raw_text: str) -> Dict[str, str]:
    """
    Splits on header lines, mapping via SECTION_HEADER_ALIASES (which
    includes fuzzy variants like "My Background"). Falls back to a single
    'body' bucket for anything before the first recognized header or when no
    headers are detected at all.
    """
    lines = normalized_raw_text.split("\n")
    sections: Dict[str, List[str]] = {"body": []}
    current = "body"

    for line in lines:
        stripped = line.strip()
        candidate = stripped.rstrip(":").lower()
        if stripped and _HEADER_LINE_RE.match(stripped) and candidate in SECTION_HEADER_ALIASES:
            current = SECTION_HEADER_ALIASES[candidate]
            sections.setdefault(current, [])
            continue
        sections[current].append(line)

    return {name: "\n".join(lines_).strip() for name, lines_ in sections.items() if "\n".join(lines_).strip()}


# ---------------------------------------------------------------------------
# STAGE 5: SECTION-AWARE SKILL EXTRACTION
# ---------------------------------------------------------------------------

def extract_skills_by_section(sections: Dict[str, str]) -> Dict[str, Set[str]]:
    """
    Returns {section_name: {canonical_skill, ...}} by running the same
    phrase-protect + normalize pipeline on each section independently, then
    checking which canonical skills from CANONICAL_SKILLS are present.
    """
    canonical_set = set(CANONICAL_SKILLS) | set(TERM_NORMALIZATION_MAP.values())
    result: Dict[str, Set[str]] = {}
    for name, text in sections.items():
        clean = build_clean_text(text)
        found = set()
        for skill in canonical_set:
            pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
            if re.search(pattern, clean):
                found.add(skill)
        if found:
            result[name] = found
    return result


def flatten_skills(skills_by_section: Dict[str, Set[str]]) -> Set[str]:
    all_skills: Set[str] = set()
    for skill_set in skills_by_section.values():
        all_skills |= skill_set
    return all_skills


# ---------------------------------------------------------------------------
# CACHING (avoids re-parsing/re-OCRing unchanged PDFs on repeated runs)
# ---------------------------------------------------------------------------

def _file_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def process_pdf(pdf_path: str, cache_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Full pipeline for one PDF. If cache_dir is given, results are memoized
    on disk keyed by file content hash, so unchanged resumes skip
    re-extraction (and re-OCR, which is the expensive part) on reruns.
    """
    import json

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = Path(cache_dir) / f"{_file_hash(pdf_path)}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["skills"] = set(cached["skills"])
            cached["skills_by_section"] = {k: set(v) for k, v in cached["skills_by_section"].items()}
            return cached

    raw_text, method = extract_text_from_file(pdf_path)
    normalized_raw = normalize_whitespace(raw_text)
    sections = split_into_sections(normalized_raw)
    skills_by_section = extract_skills_by_section(sections)
    skills = flatten_skills(skills_by_section)
    clean_text = build_clean_text(normalized_raw)

    result = {
        "filename": os.path.basename(pdf_path),
        "extraction_method": method,
        "raw_text": normalized_raw,
        "clean_text": clean_text,
        "sections": sections,
        "skills": skills,
        "skills_by_section": skills_by_section,
    }

    if cache_path:
        serializable = dict(result)
        serializable["skills"] = sorted(skills)
        serializable["skills_by_section"] = {k: sorted(v) for k, v in skills_by_section.items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    return result


def process_resume_directory(resume_dir: str, cache_dir: Optional[str] = None) -> List[Dict[str, object]]:
    """Scans directory for .pdf, .docx, and .txt candidate resumes."""
    results = []
    files = []
    
    # Collect all supported resume formats
    for ext in ("*.pdf", "*.docx", "*.txt"):
        files.extend(Path(resume_dir).glob(ext))
        
    for path in sorted(files):
        try:
            results.append(process_pdf(str(path), cache_dir=cache_dir))
        except Exception as exc:
            print(f"[WARN] Skipping '{path.name}': {exc}")
            
    return results