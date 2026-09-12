"""
src/keyword_engine.py
----------------------
Fixes vs. the previous version:
  B1 (partial). Synonym awareness comes for free here because both JD and
      resume clean_text have already been through parser.build_clean_text(),
      which normalizes synonyms to a single canonical form before this
      module ever sees the text.
  New: token_pattern keeps "c++", "node.js", "ci/cd" as single tokens (no
      more "Node.js" -> "Node"+"js"), and ngram_range=(1, 2) lets bigrams
      like "rest api" / "machine learning" be captured as coherent features
      even though they contain a space.
  New: adds a hard-skill Jaccard intersection ratio on top of TF-IDF cosine,
      which is a harder, more literal signal than TF-IDF alone (a resume
      that mentions "python" fifteen times vs once still just gets 1 point
      in the skill-set intersection — this counters TF-IDF's tendency to
      reward keyword stuffing).
"""

from typing import Dict, List, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Keeps tokens like "c++", "node.js", "ci/cd", "c#" intact instead of being
# shredded by the default \w+ token pattern.
TOKEN_PATTERN = r"\.?[a-zA-Z0-9\+\#/]+(?:\.[a-zA-Z0-9\+\#/]+)*"


def tfidf_cosine_scores(jd_clean_text: str, resume_clean_texts: List[str]) -> np.ndarray:
    """
    Fits TF-IDF jointly over [JD, *resumes] (so the vocabulary and IDF
    weights are shared/comparable), then returns the cosine similarity of
    each resume against the JD as a 1D array aligned with resume_clean_texts.
    """
    corpus = [jd_clean_text] + resume_clean_texts
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,   # dampens keyword-stuffing (log-scaled term freq)
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(corpus)
    jd_vector = matrix[0:1] # type: ignore[operator]
    resume_vectors = matrix[1:] # type: ignore[operator]
    if resume_vectors.shape[0] == 0:
        return np.array([])
    return cosine_similarity(jd_vector, resume_vectors).flatten()


def jaccard_skill_score(jd_skills: Set[str], resume_skills: Set[str]) -> float:
    """
    Intersection-over-JD-required-skills ratio (not classic symmetric
    Jaccard) — we specifically want "what fraction of what the JD asked for
    does this candidate have", not penalized by extra unrelated skills the
    candidate happens to list. Classic Jaccard (|A∩B| / |A∪B|) is available
    via `strict=True` if you want candidates penalized for irrelevant noise.
    """
    if not jd_skills:
        return 0.0
    return len(jd_skills & resume_skills) / len(jd_skills)


def classic_jaccard(jd_skills: Set[str], resume_skills: Set[str]) -> float:
    union = jd_skills | resume_skills
    if not union:
        return 0.0
    return len(jd_skills & resume_skills) / len(union)


def hard_requirement_penalty(jd_skills: Set[str], resume_skills: Set[str], required_skills: Set[str]) -> float:
    """
    `required_skills` is a subset of jd_skills the JD explicitly marks as
    mandatory (see ranker.identify_required_skills). Returns a multiplier in
    (0, 1]: 1.0 if every required skill is present, decaying toward a floor
    as more required skills are missing. This lets ranker.py apply a harder
    penalty for missing hard requirements than for missing "nice to have"
    skills, without hand-waving a static formula that always over- or
    under-penalizes.
    """
    if not required_skills:
        return 1.0
    missing = required_skills - resume_skills
    if not missing:
        return 1.0
    fraction_missing = len(missing) / len(required_skills)
    floor = 0.4  # never zero out a candidate entirely on this signal alone
    return 1.0 - (1.0 - floor) * fraction_missing


def compute_keyword_scores(
    jd_clean_text: str,
    jd_skills: Set[str],
    resumes: List[Dict[str, object]],
    required_skills: Set[str],
) -> List[Dict[str, float]]:
    """
    Batch-computes all keyword-side scores for a list of parsed resume dicts
    (as produced by parser.process_pdf). Returns one dict of component
    scores per resume, in the same order as `resumes`.
    """
    resume_texts = [r["clean_text"] for r in resumes]
    cosine_scores = tfidf_cosine_scores(jd_clean_text, resume_texts) # type: ignore[operator]

    scored = []
    for i, resume in enumerate(resumes):
        resume_skills = resume["skills"]
        scored.append({
            "tfidf_cosine": float(cosine_scores[i]) if len(cosine_scores) else 0.0,
            "skill_jaccard": jaccard_skill_score(jd_skills, resume_skills), # type: ignore[operator]
            "hard_requirement_multiplier": hard_requirement_penalty(jd_skills, resume_skills, required_skills), # type: ignore[operator]
        })
    return scored