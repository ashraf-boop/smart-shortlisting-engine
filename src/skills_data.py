"""
skills_data.py
--------------
Single source of truth for the skill vocabulary, multi-word phrase list,
synonym/normalization map, and section-header aliases. Every other module
imports from here so the whole pipeline stays consistent (a skill detected
in the JD is guaranteed to be spelled the same way when detected in a
resume).
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical skill vocabulary. Keys are the canonical form used everywhere
# downstream (scoring, explanations). Extend this for your domain.
# ---------------------------------------------------------------------------
CANONICAL_SKILLS: List[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "html", "css",
    "react", "angular", "vue", "next.js", "redux", "tailwind", "bootstrap",
    "node.js", "express", "django", "flask", "fastapi", "spring boot",
    "rest api", "graphql", "microservices", "ci/cd",
    "mongodb", "mysql", "postgresql", "sqlite", "redis", "firebase", "dynamodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "git", "github", "linux",
    "machine learning", "deep learning", "nlp", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "spacy", "data analysis", "data visualization", "power bi", "tableau",
    "agile", "scrum", "unit testing", "object oriented programming", "data structures",
    "algorithms", "system design",
]

# ---------------------------------------------------------------------------
# Synonym / alias -> canonical form. Anything on the left normalizes to the
# canonical form on the right before matching, so "AWS" and "Amazon Web
# Services" collapse to a single token instead of being treated as unrelated.
# ---------------------------------------------------------------------------
TERM_NORMALIZATION_MAP: Dict[str, str] = {
    "reactjs": "react",
    "react js": "react",
    "amazon web services": "aws",
    "nodejs": "node.js",
    "node js": "node.js",
    "expressjs": "express",
    "express js": "express",
    "vuejs": "vue",
    "vue js": "vue",
    "nextjs": "next.js",
    "next js": "next.js",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "postgres": "postgresql",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "continuous integration": "ci/cd",
    "continuous integration/continuous deployment": "ci/cd",
    "continuous deployment": "ci/cd",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "ml": "machine learning",
    "dl": "deep learning",
    "oop": "object oriented programming",
    "object-oriented programming": "object oriented programming",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "spring": "spring boot",
    "springboot": "spring boot",
    "postgre sql": "postgresql",
    "js": "javascript",
    "ts": "typescript",
}

# Multi-word (and punctuation-bearing) phrases that must survive naive
# whitespace/punctuation tokenization intact. Sorted longest-first so greedy
# phrase protection matches the most specific phrase before a shorter
# substring of it.
MULTI_WORD_PHRASES: List[str] = sorted(
    set(CANONICAL_SKILLS) | set(TERM_NORMALIZATION_MAP.keys()),
    key=len,
    reverse=True,
)

# Certifications / "hard requirement" style tokens. If a JD mentions these,
# the ranker treats their absence in a resume as a harder penalty than a
# missing general skill (see ranker.py).
HARD_REQUIREMENT_MARKERS: List[str] = [
    "required skill", "required experience", "must have experience",
    "must be proficient in", "mandatory requirement", "aws certified",
    "pmp", "certified", "certification", "must have", "required", "mandatory"
]

# Section header aliases -> canonical section name. Includes loose/fuzzy
# variants like "my background" that a strict regex would otherwise miss.
SECTION_HEADER_ALIASES: Dict[str, str] = {
    "work experience": "experience", "professional experience": "experience",
    "employment history": "experience", "experience": "experience",
    "internship experience": "experience", "career history": "experience",
    "technical skills": "skills", "skills & tools": "skills",
    "core competencies": "skills", "skills": "skills", "key skills": "skills",
    "education": "education", "academic background": "education",
    "academic qualifications": "education",
    "projects": "projects", "academic projects": "projects",
    "personal projects": "projects", "key projects": "projects",
    "certifications": "certifications", "certifications & courses": "certifications",
    "licenses & certifications": "certifications",
    "achievements": "achievements", "accomplishments": "achievements",
    "extracurricular": "extracurricular", "activities": "extracurricular",
    "summary": "summary", "objective": "summary", "profile": "summary",
    "about me": "summary", "my background": "summary", "background": "summary",
}