"""
src/semantic_engine.py
-----------------------
Fixes vs. the previous version:
  B2. Document-length dilution: instead of embedding the ENTIRE resume as
      one vector (where a long, detailed resume's signal gets averaged/
      diluted across lots of irrelevant tokens, letting short resumes win
      on cosine similarity by accident), we:
        1. Split each resume into small section-tagged chunks.
        2. Embed every chunk independently.
        3. Score the resume as a SECTION-WEIGHTED AVERAGE OF ITS TOP-K
           best-matching chunks against the JD, not a single whole-document
           vector. This means a 3-page resume with one dense, highly
           relevant "Experience" section scores on that strength, rather
           than having it drowned out by unrelated "Extracurricular" text.
  Performance: batch encoding (encode_texts encodes all chunks across all
      candidates in one call) and on-disk embedding caching keyed by a hash
      of the text, so re-running the pipeline on unchanged resumes/JD
      doesn't re-run the (relatively expensive) transformer forward pass.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Chunks from these sections are considered more diagnostic of true fit
# than generic body text, so their similarity contributes more to the
# candidate's final semantic score.
SECTION_WEIGHTS = {
    "experience": 1.35,
    "skills": 1.35,
    "projects": 1.15,
    "summary": 1.05,
    "education": 0.9,
    "certifications": 1.0,
    "achievements": 0.9,
    "extracurricular": 0.7,
    "body": 1.0,
}

TOP_K_CHUNKS = 5  # how many best-matching chunks per resume feed the final average

_MODEL_CACHE = {"model": None}


def get_model():
    if _MODEL_CACHE["model"] is None:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE["model"] = SentenceTransformer(MODEL_NAME) # type: ignore[operator]
    return _MODEL_CACHE["model"]


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_sections(sections: Dict[str, str], sentences_per_chunk: int = 3) -> List[Tuple[str, str]]:
    """
    Returns a list of (section_name, chunk_text) pairs. Grouping a few
    sentences per chunk (rather than one embedding per sentence) keeps
    enough local context for the embedding to be meaningful while still
    being much finer-grained than "one vector for the whole resume".
    """
    chunks: List[Tuple[str, str]] = []
    for section_name, text in sections.items():
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        for i in range(0, len(sentences), sentences_per_chunk):
            group = " ".join(sentences[i:i + sentences_per_chunk])
            if len(group) >= 15:  # skip near-empty fragments (e.g. bare headers)
                chunks.append((section_name, group))
    return chunks or [("body", text) for text in sections.values() if text.strip()]


# ---------------------------------------------------------------------------
# CACHED EMBEDDING
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def embed_texts_cached(texts: List[str], cache_dir: str = None) -> np.ndarray: # type: ignore[operator]
    """
    Batch-embeds `texts`, using a per-text on-disk cache so repeated runs
    (e.g. iterating on ranker weights during development) don't re-embed
    text that hasn't changed. Cache is a flat directory of
    "<md5(text)>.json" files, each holding the embedding as a float list.
    """
    if not texts:
        return np.zeros((0, 384))

    if not cache_dir:
        model = get_model()
        return np.array(model.encode(texts, batch_size=32, show_progress_bar=False))

    os.makedirs(cache_dir, exist_ok=True)
    embeddings: List[np.ndarray] = [None] * len(texts) # type: ignore[operator]
    to_encode_idx: List[int] = []
    to_encode_text: List[str] = []

    for i, text in enumerate(texts):
        cache_path = Path(cache_dir) / f"{_text_hash(text)}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                embeddings[i] = np.array(json.load(f))
        else:
            to_encode_idx.append(i)
            to_encode_text.append(text)

    if to_encode_text:
        model = get_model()
        fresh = model.encode(to_encode_text, batch_size=32, show_progress_bar=False)
        for idx, text, vec in zip(to_encode_idx, to_encode_text, fresh):
            embeddings[idx] = np.array(vec)
            cache_path = Path(cache_dir) / f"{_text_hash(text)}.json"
            with open(cache_path, "w") as f:
                json.dump(vec.tolist(), f)

    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def compute_semantic_scores(
    jd_sections: Dict[str, str],
    resumes_sections: List[Dict[str, str]],
    cache_dir: str = None, # type: ignore[operator]
) -> List[float]:
    """
    jd_sections: the JD's {section_name: text} dict.
    resumes_sections: list of each resume's {section_name: text} dict, in
        the order scores should be returned.

    Returns one length-dilution-resistant semantic score per resume, in
    [0, 1] (cosine similarities are ~[-1, 1] but practically positive for
    related professional text; we clip for safety).
    """
    # JD: chunk it too (JDs can have distinct "requirements" vs "about us"
    # sections) and represent it as the mean of its chunk embeddings.
    jd_chunks = [text for _, text in chunk_sections(jd_sections)]
    jd_embeddings = embed_texts_cached(jd_chunks, cache_dir)
    jd_vector = jd_embeddings.mean(axis=0, keepdims=True)

    # Flatten all resume chunks into one big batch for efficient encoding,
    # tracking chunk -> resume index and chunk -> section weight.
    all_chunks: List[str] = []
    chunk_owner: List[int] = []
    chunk_weight: List[float] = []

    per_resume_chunks: List[List[Tuple[str, str]]] = []
    for resume_sections in resumes_sections:
        chunks = chunk_sections(resume_sections)
        per_resume_chunks.append(chunks)
        for section_name, chunk_text in chunks:
            all_chunks.append(chunk_text)
            chunk_owner.append(len(per_resume_chunks) - 1)
            chunk_weight.append(SECTION_WEIGHTS.get(section_name, 1.0))

    if not all_chunks:
        return [0.0] * len(resumes_sections)

    chunk_embeddings = embed_texts_cached(all_chunks, cache_dir)
    similarities = _cosine(chunk_embeddings, jd_vector).flatten()  # one sim per chunk

    scores = [0.0] * len(resumes_sections)
    owner_arr = np.array(chunk_owner)
    weight_arr = np.array(chunk_weight)

    for resume_idx in range(len(resumes_sections)):
        mask = owner_arr == resume_idx
        if not mask.any():
            continue
        sims = similarities[mask]
        weights = weight_arr[mask]
        weighted = sims * weights
        # Take the top-K weighted chunk scores (length-dilution fix: we
        # don't average over the WHOLE resume, just its strongest,
        # highest-priority-section evidence).
        top_k_indices = np.argsort(weighted)[-TOP_K_CHUNKS:]
        top_k_sims = sims[top_k_indices]
        top_k_weights = weights[top_k_indices]

        score = float(np.sum(top_k_sims * top_k_weights) / np.sum(top_k_weights))
        scores[resume_idx] = max(0.0, min(1.0, score))

    return scores