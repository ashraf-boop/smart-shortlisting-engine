"""
src/ranker.py
--------------
Fixes vs. the previous version:
  B3. Static weighting fails when a JD strictly requires hard
      certifications/skills. We now (1) detect which JD skills are flagged
      as "hard requirements" (near words like "required", "must have",
      "certified") via skills_data.HARD_REQUIREMENT_MARKERS, and (2) apply
      keyword_engine.hard_requirement_penalty as a MULTIPLIER on top of the
      blended score, so a candidate missing a mandatory item is capped well
      below one who's merely weaker on general semantic fit — no amount of
      unrelated semantic similarity can fully compensate for a missing hard
      requirement.
  Dynamic weighting: weights are a configurable dict (not a hardcoded
      0.4/0.6 split), defaulting to a 3-way TF-IDF / Jaccard / Semantic
      blend that a Streamlit slider (or CLI flag) can override at runtime.
"""

import re
from typing import Dict, List, Set

from skills_data import CANONICAL_SKILLS, TERM_NORMALIZATION_MAP, HARD_REQUIREMENT_MARKERS
import keyword_engine
import semantic_engine

DEFAULT_WEIGHTS = {
    "tfidf_cosine": 0.30,
    "skill_jaccard": 0.30,
    "semantic": 0.40,
}

_REQUIREMENT_WINDOW_CHARS = 60  # how close a "required"/"must have" marker must be to a skill mention


def identify_required_skills(jd_clean_text: str, jd_skills: Set[str]) -> Set[str]:
    """
    Flags which JD skills are "hard requirements" by checking whether a
    requirement marker (from HARD_REQUIREMENT_MARKERS) appears within a
    small character window of the skill mention in the JD text. This is a
    heuristic, not a guarantee — teams can swap in a dependency-parse-based
    version later without touching the rest of the pipeline, since ranker.py
    only depends on the returned Set[str] contract.
    """
    canonical_set = set(CANONICAL_SKILLS) | set(TERM_NORMALIZATION_MAP.values())
    required: Set[str] = set()

    for skill in jd_skills & canonical_set:
        for match in re.finditer(re.escape(skill), jd_clean_text):
            start = max(0, match.start() - _REQUIREMENT_WINDOW_CHARS)
            end = min(len(jd_clean_text), match.end() + _REQUIREMENT_WINDOW_CHARS)
            window = jd_clean_text[start:end].lower()
            if any(marker in window for marker in HARD_REQUIREMENT_MARKERS):
                required.add(skill)
                break

    return required


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {k: v / total for k, v in weights.items()}


def rank_candidates(
    jd_data: Dict[str, object],
    resumes: List[Dict[str, object]],
    weights: Dict[str, float] = None, # type: ignore[operator]
    semantic_cache_dir: str = None, # type: ignore[operator]
) -> List[Dict[str, object]]:
    """
    jd_data: parser.process_pdf(...) output for the Job Description.
    resumes: list of parser.process_pdf(...) outputs for each candidate.
    weights: optional override for {"tfidf_cosine", "skill_jaccard",
        "semantic"} — e.g. from a Streamlit slider. Auto-normalized to sum
        to 1 so partial overrides (e.g. only bumping "semantic") are safe.

    Returns resumes sorted best-fit-first, each augmented with:
        "component_scores": {...}          raw sub-scores before weighting
        "required_skills_missing": set      hard requirements not met
        "final_score": float in [0, 1]
    """
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)

    jd_skills: Set[str] = jd_data["skills"] # type: ignore[operator]
    required_skills = identify_required_skills(jd_data["clean_text"], jd_skills) # type: ignore[operator]

    keyword_scores = keyword_engine.compute_keyword_scores(
        jd_clean_text=jd_data["clean_text"], # type: ignore[operator]
        jd_skills=jd_skills,
        resumes=resumes,
        required_skills=required_skills,
    )
    semantic_scores = semantic_engine.compute_semantic_scores(
        jd_sections=jd_data["sections"], # type: ignore[operator]
        resumes_sections=[r["sections"] for r in resumes], # type: ignore[operator]
        cache_dir=semantic_cache_dir,
    )

    ranked = []
    for resume, kw_score, sem_score in zip(resumes, keyword_scores, semantic_scores):
        blended = (
            weights["tfidf_cosine"] * kw_score["tfidf_cosine"]
            + weights["skill_jaccard"] * kw_score["skill_jaccard"]
            + weights["semantic"] * sem_score
        )
        final_score = blended * kw_score["hard_requirement_multiplier"]

        ranked.append({
            **resume,
            "component_scores": {
                "tfidf_cosine": round(kw_score["tfidf_cosine"], 4),
                "skill_jaccard": round(kw_score["skill_jaccard"], 4),
                "semantic": round(sem_score, 4),
                "hard_requirement_multiplier": round(kw_score["hard_requirement_multiplier"], 4),
            },
            "required_skills": required_skills,
            "required_skills_missing": required_skills - resume["skills"], # type: ignore[operator]
            "final_score": round(final_score, 4),
        })

    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked

def calculate_auto_weights(jd_text: str, jd_skills: list) -> dict:
    """
    Dynamically infers optimal matching weights based on JD structure and skill density.
    """
    total_words = len(jd_text.split())
    skill_count = len(jd_skills)
    
    # Calculate skill density (skills per 100 words)
    skill_density = (skill_count / max(total_words, 1)) * 100

    # High skill density (Tool-heavy JD) -> Boost Jaccard
    if skill_density > 3.0:
        return {"tfidf_cosine": 0.20, "skill_jaccard": 0.50, "semantic": 0.30, "mode": "Tool-Heavy / Technical"}
    
    # Low skill density (Conceptual JD) -> Boost Semantic
    elif skill_density < 1.0:
        return {"tfidf_cosine": 0.20, "skill_jaccard": 0.20, "semantic": 0.60, "mode": "Conceptual / Narrative"}
    
    # Balanced default
    else:
        return {"tfidf_cosine": 0.30, "skill_jaccard": 0.35, "semantic": 0.35, "mode": "Balanced Alignment"}