# Resume Match Checker

> An offline, privacy-first, high-capacity resume screening tool built with Streamlit, Plotly, PyMuPDF, and SentenceTransformers.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![Offline](https://img.shields.io/badge/LLM-None%20required-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Built for hackathons and recruiting workflows, this application takes a batch of resumes and a job description, scores each candidate using a multi-metric hybrid ranking approach, checks the job description for biased or overly rigid wording, and shows clear, plain-language explanations and visual breakdowns for every match — all running locally, with no data ever leaving your machine.

## Why this exists

Most ATS tools fall into one of two traps: legacy keyword-matching systems that penalize good candidates for phrasing things differently, or cloud-based LLM tools that introduce API cost, latency, and privacy risk from sending candidate PII to a third party. Resume Match Checker takes a third path — a hybrid, fully local scoring engine that's fast, explainable, and never sends a single byte off your machine.

## Key Features

**High-Capacity Batch Selection**
Dynamic selection controls with real-time filename search and file-type filters (.pdf, .docx, .txt), plus a scrollable multi-column grid for handling large batches of resumes.

**Hybrid Scoring Engine (no LLM required)**
- **Keyword Match** — measures direct keyword and term overlap with the job description (TF-IDF cosine similarity).
- **Skill Coverage** — measures how many of the job's required skills appear in the resume (Jaccard overlap).
- **Context Match** — uses local SentenceTransformers embeddings to judge overall fit beyond exact keyword matches.

**Flexible Weighting**
- **Auto-recommended** — analyzes the job description's skill density and automatically tunes the weighting (e.g., leaning on skill coverage for tool-heavy roles, or context match for more conceptual roles).
- **Manual sliders** — lets you set the weighting yourself.

**Visual Analytics (Plotly)**
- **Match composition (stacked bar)** — shows how much of each candidate's score comes from keyword match, skill coverage, and context match.
- **Candidate comparison (radar chart)** — compares top candidates side by side across all three metrics.
- **Skill match grid (heatmap)** — shows at a glance which required skills each candidate does or doesn't have.

**Explanations & Review**
- **Head-to-head comparison** — explains in plain language why one candidate ranked above another.
- **Job description review** — flags wording in the job post that may be unclear, overly rigid, or unintentionally exclusionary.
- **Standalone resume review** — reviews skills and resume sections even when no job description has been uploaded yet.

## Tech Stack & Dependencies

- **Frontend / UI:** Streamlit, Plotly Express & Graph Objects
- **Document Parsers:** PyMuPDF (fitz), python-docx, reportlab
- **NLP & Scoring:** Scikit-learn (TF-IDF), SentenceTransformers (all-MiniLM-L6-v2), NumPy, Pandas
- **Environment:** Python 3.9+ (Windows / Linux / macOS)

## Project Structure

```
smart-shortlisting-engine/
├── app.py                    # Main Streamlit UI & dashboard
├── src/
│   ├── parser.py             # PDF/DOCX/TXT ingestion & parsing logic
│   ├── ranker.py             # Hybrid scoring engine & auto-weight optimization
│   └── explainer.py          # Head-to-head comparisons, reports & bias review
├── data/
│   ├── raw/                  # Sample job descriptions & resumes
│   └── processed/.cache/     # Local cache for parsed documents & embeddings
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Quick Start & Installation

### 1. Clone the repository & set up a virtual environment

```powershell
# Windows (PowerShell)
git clone https://github.com/ashraf-boop/smart-shortlisting-engine.git
cd smart-shortlisting-engine

python -m venv venv
.\venv\Scripts\activate
```

```bash
# macOS / Linux
git clone https://github.com/ashraf-boop/smart-shortlisting-engine.git
cd smart-shortlisting-engine

python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate a sample job description PDF (optional)

```bash
python make_jd.py
```

### 4. Launch the application

```bash
streamlit run app.py
```

## How to Use It

1. **Upload the job description** — drag and drop a job description file (.pdf, .docx, or .txt) into the left upload slot.
2. **Upload resumes** — add as many candidate resumes as you like into the right upload slot.
3. **Choose which resumes to include** — use the filename search or file-type filter to select or deselect resumes in the grid.
4. **Choose a weighting strategy** — leave "Auto-recommended" on in the sidebar, or switch to manual sliders to set your own weighting.
5. **Check the resumes** — click "Check resumes against the job description" to see the results, head-to-head comparisons, visual breakdowns, and individual reviews.

## How Scoring Works

Each candidate's Overall Match is a weighted combination of three independent signals:

```
S_final = w1 * S_tfidf + w2 * S_jaccard + w3 * S_semantic
```

where `w1 + w2 + w3 = 1.0`. When "Auto-recommended" weighting is on, the app measures the job description's skill density (recognized technical skills as a share of total words) and shifts the weights accordingly — favoring **Skill Coverage** for tool-heavy postings, and **Context Match** for more conceptual, narrative-style postings.

## Roadmap

- [ ] Export ranked results and comparison reports to PDF/CSV
- [ ] Support additional resume formats (e.g., LinkedIn PDF exports)
- [ ] Configurable skill taxonomy per industry
- [ ] Optional multi-language resume support

## Contributing

Issues and pull requests are welcome. If you're proposing a larger change, please open an issue first to discuss what you'd like to change.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Shaik Abdul Ashraf**
[GitHub](https://github.com/ashraf-boop) · [LinkedIn](https://www.linkedin.com/in/shaik-abdul-ashraf/)
