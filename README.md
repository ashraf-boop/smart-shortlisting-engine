# Resume Match Checker

An offline, privacy-first, high-capacity resume screening tool built with Streamlit, Plotly, PyMuPDF, and SentenceTransformers.

Built for hackathons and recruiting workflows, this application takes a batch of resumes and a job description, scores each candidate using a multi-metric hybrid ranking approach, checks the job description for biased or overly rigid wording, and shows clear, plain-language explanations and visual breakdowns for every match.

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
resume-match-checker/
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
git clone https://github.com/ashraf-boop/resume-match-checker.git
cd resume-match-checker

python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate a sample job description PDF (optional)

```powershell
python make_jd.py
```

### 4. Launch the application

```powershell
streamlit run app.py
```

## How to Use It

1. **Upload the job description** — drag and drop a job description file (.pdf, .docx, or .txt) into the left upload slot.
2. **Upload resumes** — add as many candidate resumes as you like into the right upload slot.
3. **Choose which resumes to include** — use the filename search or file-type filter to select or deselect resumes in the grid.
4. **Choose a weighting strategy** — leave "Auto-recommended" on in the sidebar, or switch to manual sliders to set your own weighting.
5. **Check the resumes** — click "Check resumes against the job description" to see the results, head-to-head comparisons, visual breakdowns, and individual reviews.
