"""
app.py
------
CLI orchestrator for the offline Smart Shortlisting Engine.

Usage:
    python app.py --jd data/raw/Sample_JD.pdf --resumes data/raw/ \
        --weight-tfidf 0.3 --weight-jaccard 0.3 --weight-semantic 0.4

All matching happens locally via src/keyword_engine.py (TF-IDF + Jaccard)
and src/semantic_engine.py (sentence-transformers) — no external LLM API
calls are made anywhere in the ranking path.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import parser as pdf_parser  # noqa: E402  (src/parser.py) # type: ignore[operator]
import ranker  # noqa: E402 # type: ignore[operator]
import explainer  # noqa: E402 # type: ignore[operator]


def _serializable(ranked):
    """Converts sets to sorted lists so the ranking is JSON-dumpable."""
    out = []
    for c in ranked:
        c = dict(c)
        for key in ("skills", "required_skills", "required_skills_missing"):
            if key in c and isinstance(c[key], set):
                c[key] = sorted(c[key])
        c["skills_by_section"] = {k: sorted(v) for k, v in c.get("skills_by_section", {}).items()}
        out.append(c)
    return out


def main():
    parser_cli = argparse.ArgumentParser(description="Offline Smart Shortlisting Engine")
    parser_cli.add_argument("--jd", required=True, help="Path to Job Description PDF")
    parser_cli.add_argument("--resumes", required=True, help="Directory of candidate resume PDFs")
    parser_cli.add_argument("--out", default="data/processed/ranked_candidates.json")
    parser_cli.add_argument("--cache-dir", default="data/processed/.cache", help="Parse + embedding cache")
    parser_cli.add_argument("--weight-tfidf", type=float, default=0.30)
    parser_cli.add_argument("--weight-jaccard", type=float, default=0.30)
    parser_cli.add_argument("--weight-semantic", type=float, default=0.40)
    parser_cli.add_argument("--top-n", type=int, default=3)
    args = parser_cli.parse_args()

    print(f"Parsing JD: {args.jd}")
    jd_data = pdf_parser.process_pdf(args.jd, cache_dir=f"{args.cache_dir}/parsed")

    print(f"Parsing resumes in: {args.resumes}")
    resumes = pdf_parser.process_resume_directory(args.resumes, cache_dir=f"{args.cache_dir}/parsed")
    print(f"  -> {len(resumes)} resumes parsed successfully.")

    weights = {
        "tfidf_cosine": args.weight_tfidf,
        "skill_jaccard": args.weight_jaccard,
        "semantic": args.weight_semantic,
    }

    print("Ranking candidates (TF-IDF + Jaccard + semantic, hybrid, offline)...")
    ranked = ranker.rank_candidates(
        jd_data=jd_data,
        resumes=resumes,
        weights=weights,
        semantic_cache_dir=f"{args.cache_dir}/embeddings",
    )

    print("\n" + "=" * 70)
    print(f"TOP {args.top_n} CANDIDATE EXPLANATIONS")
    print("=" * 70)
    print(explainer.explain_top_n(ranked, jd_data["skills"], n=args.top_n))

    print("\n" + "=" * 70)
    print("FULL RANKING")
    print("=" * 70)
    for c in ranked:
        print(f"  #{c['rank']:>2}  {c['final_score']:.4f}  {c['filename']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "job_description": {**jd_data, "skills": sorted(jd_data["skills"]),
                                 "skills_by_section": {k: sorted(v) for k, v in jd_data["skills_by_section"].items()}},
            "ranked_candidates": _serializable(ranked),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results written to: {args.out}")


if __name__ == "__main__":
    main()