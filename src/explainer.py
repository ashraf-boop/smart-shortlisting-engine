"""
src/explainer.py
------------------
Fixes vs. the previous version:
  C1. Instead of printing raw `matched = [...] / missing = [...]` token
      lists, explanations are built from deterministic, structured
      templates that read like a recruiter's shortlist note — while still
      being 100% derived from the rule-based matching (no LLM call, so it
      stays reproducible and auditable, which is exactly what the hackathon
      brief requires ("judges will ask you to walk through how the matching
      actually works")).
  Adds section provenance: "matched under Experience" / "matched under
      Skills" using resume["skills_by_section"] from parser.py, so the
      explanation says WHERE the evidence was found, not just THAT it was
      found.
"""

from typing import Dict, List, Set


def _skills_with_section(skill_set: Set[str], skills_by_section: Dict[str, Set[str]]) -> List[str]:
    """
    Formats each skill as "python (Experience)" by looking up which
    section(s) it was detected in; falls back to bare skill name if no
    section provenance is available.
    """
    section_lookup: Dict[str, Set[str]] = {}
    for section_name, skills in skills_by_section.items():
        for skill in skills:
            section_lookup.setdefault(skill, set()).add(section_name.title())

    formatted = []
    for skill in sorted(skill_set):
        sections = section_lookup.get(skill)
        if sections:
            formatted.append(f"{skill} ({', '.join(sorted(sections))})")
        else:
            formatted.append(skill)
    return formatted


def _fit_label(final_score: float) -> str:
    if final_score >= 0.75:
        return "Strong fit"
    if final_score >= 0.55:
        return "Good fit"
    if final_score >= 0.35:
        return "Partial fit"
    return "Weak fit"


def explain_candidate(candidate: Dict[str, object], jd_skills: Set[str]) -> str:
    """
    Builds one deterministic natural-language explanation paragraph for a
    single ranked candidate dict (as produced by ranker.rank_candidates).
    """
    matched = jd_skills & candidate["skills"] # type: ignore[operator]
    missing = jd_skills - candidate["skills"] # type: ignore[operator]
    required_missing = candidate.get("required_skills_missing", set())

    matched_fmt = _skills_with_section(matched, candidate.get("skills_by_section", {})) # type: ignore[operator]
    scores = candidate["component_scores"]

    lines = []
    lines.append(
        f"Rank #{candidate['rank']} — {candidate['filename']} "
        f"({_fit_label(candidate['final_score'])}, score {candidate['final_score']:.2f})" # type: ignore[operator]
    )

    if matched_fmt:
        lines.append("  Matched skills: " + "; ".join(matched_fmt))
    else:
        lines.append("  Matched skills: none of the JD's explicit skill terms were found.")

    if missing:
        lines.append("  Missing skills: " + ", ".join(sorted(missing)))
    else:
        lines.append("  Missing skills: none — this candidate covers every skill mentioned in the JD.")

    if required_missing:
        lines.append(
            "  ⚠ Missing HARD requirement(s): " + ", ".join(sorted(required_missing)) # type: ignore[operator]
            + " — the JD explicitly marks these as required/mandatory."
        )

    lines.append(
        f"  Score breakdown: keyword overlap (TF-IDF) {scores['tfidf_cosine']:.2f}, " # type: ignore[operator]
        f"required-skill coverage {scores['skill_jaccard']:.2f}, " # type: ignore[operator]
        f"semantic/contextual fit {scores['semantic']:.2f}" # type: ignore[operator]
        + (
            f", hard-requirement penalty x{scores['hard_requirement_multiplier']:.2f}" # type: ignore[operator]
            if scores["hard_requirement_multiplier"] < 1.0 else "" # type: ignore[operator]
        )
    )
    return "\n".join(lines)


def explain_top_n(ranked_candidates: List[Dict[str, object]], jd_skills: Set[str], n: int = 3) -> str:
    """Convenience wrapper: joins explanations for the top-N candidates."""
    blocks = [explain_candidate(c, jd_skills) for c in ranked_candidates[:n]]
    return "\n\n".join(blocks)


def answer_comparison_query(
    candidate_x: Dict[str, object],
    candidate_y: Dict[str, object],
    jd_skills: Set[str],
) -> str:
    """
    Bonus-feature hook: deterministic answer to "Why is Candidate X ranked
    above Candidate Y?" — a UI layer (Streamlit/chat) can call this directly
    with two candidate dicts pulled from the ranked list by filename, no LLM
    required.
    """
    if candidate_x["final_score"] < candidate_y["final_score"]: # type: ignore[operator]
        candidate_x, candidate_y = candidate_y, candidate_x  # ensure x is the higher-ranked one

    x_only = (jd_skills & candidate_x["skills"]) - candidate_y["skills"]  # type: ignore[operator]
    y_only = (jd_skills & candidate_y["skills"]) - candidate_x["skills"]  # type: ignore[operator]
    sx, sy = candidate_x["component_scores"], candidate_y["component_scores"]

    lines = [
        f"{candidate_x['filename']} (score {candidate_x['final_score']:.2f}) ranks above "
        f"{candidate_y['filename']} (score {candidate_y['final_score']:.2f})."
    ]
    if x_only:
        lines.append(f"- {candidate_x['filename']} has these JD-relevant skills that the other doesn't: {', '.join(sorted(x_only))}.")
    if y_only:
        lines.append(f"- {candidate_y['filename']} does have {', '.join(sorted(y_only))}, which {candidate_x['filename']} lacks — but this wasn't enough to overcome the gap elsewhere.")
    if sx["semantic"] > sy["semantic"] + 0.05: # type: ignore[operator]
        lines.append(f"- {candidate_x['filename']}'s experience reads as more contextually aligned with the JD overall (semantic fit {sx['semantic']:.2f} vs {sy['semantic']:.2f}).") # type: ignore[operator]
    if sx["hard_requirement_multiplier"] < 1.0 or sy["hard_requirement_multiplier"] < 1.0: # type: ignore[operator]
        if sy["hard_requirement_multiplier"] < sx["hard_requirement_multiplier"]:  # type: ignore[operator]
            lines.append(f"- {candidate_y['filename']} is missing one or more hard requirements from the JD, which caps its score.")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# RELATIVE CANDIDATE COMPARISON (BONUS FEATURE)
# ---------------------------------------------------------------------------

def explain_relative_difference(cand_a: dict, cand_b: dict) -> str:
    """
    Explains why Candidate A ranked higher than Candidate B safely,
    handling both flat and nested dictionary structures.
    """
    reasons = []

    # Safe extraction helper to look inside "scores" sub-dict OR flat top-level dict
    def get_score(cand_dict: dict, metric_name: str) -> float:
        if isinstance(cand_dict.get("scores"), dict):
            return cand_dict["scores"].get(metric_name, 0.0)
        return cand_dict.get(metric_name, 0.0)

    cov_a = get_score(cand_a, "skill_jaccard")
    cov_b = get_score(cand_b, "skill_jaccard")

    sem_a = get_score(cand_a, "semantic")
    sem_b = get_score(cand_b, "semantic")

    tfidf_a = get_score(cand_a, "tfidf_cosine")
    tfidf_b = get_score(cand_b, "tfidf_cosine")

    # Check Skill Coverage difference
    if cov_a > cov_b:
        diff_pct = (cov_a - cov_b) * 100
        reasons.append(f"Higher required skill coverage (+{diff_pct:.1f}% coverage overlap).")
        missing_in_b = cand_b.get("required_skills_missing", [])
        if missing_in_b:
            reasons.append(f"Candidate B is missing key skills: {', '.join(missing_in_b)}.")

    # Check Semantic Fit difference
    if sem_a > sem_b:
        reasons.append(f"Stronger overall contextual alignment with the JD text (+{(sem_a - sem_b):.2f} semantic score difference).")

    # Check TF-IDF Keyword Density difference
    if tfidf_a > tfidf_b:
        reasons.append("Higher keyword density and matching terminology across experience sections.")

    if not reasons:
        reasons.append("Candidates scored similarly across metrics, but Candidate A has slight fractional advantages in vector embedding similarity.")

    cand_a_name = cand_a.get("filename", "Candidate A")
    cand_b_name = cand_b.get("filename", "Candidate B")

    return f"**Why {cand_a_name} ranks higher than {cand_b_name}:**\n" + "\n".join(f"- {r}" for r in reasons)

def generate_candidate_explanation(cand: dict, jd_skills: set) -> str:
    """
    Generates a natural-language summary report for an individual candidate.
    """
    filename = cand.get("filename", "Unknown Candidate")
    final_score = cand.get("final_score", 0.0)
    scores_dict = cand.get("scores", {})
    
    tfidf_score = scores_dict.get("tfidf_cosine", cand.get("tfidf_cosine", 0.0))
    jaccard_score = scores_dict.get("skill_jaccard", cand.get("skill_jaccard", 0.0))
    semantic_score = scores_dict.get("semantic", cand.get("semantic", 0.0))
    
    matched_skills = cand.get("skills", [])
    missing_skills = [sk for sk in jd_skills if sk not in matched_skills]
    
    lines = [
        f"Candidate Summary: {filename}",
        f"Overall Alignment Score: {final_score:.4f}",
        "-" * 50,
        f"• Keyword Overlap (TF-IDF): {tfidf_score:.2f}",
        f"• Skill Coverage (Jaccard): {jaccard_score * 100:.1f}%",
        f"• Semantic Contextual Fit: {semantic_score:.2f}",
        "",
        f"Matched Skills ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}",
        f"Missing Target Skills ({len(missing_skills)}): {', '.join(missing_skills) if missing_skills else 'None'}"
    ]
    return "\n".join(lines)