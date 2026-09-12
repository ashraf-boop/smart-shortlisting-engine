import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src modules are resolvable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import explainer  # type: ignore
import parser as doc_parser  # type: ignore
import ranker  # type: ignore


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resume Match Checker",
    layout="wide",
)

# ---------------------------------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    .section-divider {
        margin: 2rem 0 1rem 0;
        border-top: 1px solid #e5e7eb;
    }
    .candidate-card {
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .candidate-name {
        color: #f3f4f6;
        font-weight: 600;
    }
    .candidate-meta {
        color: #b8bcc4;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high { background-color: #dcfce7; color: #15803d; }
    .score-mid  { background-color: #fef9c3; color: #a16207; }
    .score-low  { background-color: #fee2e2; color: #b91c1c; }
    .rank-pill {
        display: inline-block;
        background-color: #f3f4f6;
        color: #374151;
        border-radius: 8px;
        padding: 0.1rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Resume Match Checker")
st.markdown(
    '<p class="app-subtitle">Upload a job description and a batch of resumes to see how well '
    'each candidate matches — by keywords, required skills, and overall context.</p>',
    unsafe_allow_html=True,
)

cache_dir = "data/processed/.cache"


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS FOR ROBUST SCORE RESOLUTION
# ---------------------------------------------------------------------------

def get_score(candidate, score_name, default=0.0):
    """
    Safely retrieves a sub-score across nested ('scores', 'raw_scores', 'metrics')
    or flat dictionary structures.
    """
    if not isinstance(candidate, dict):
        return default

    for container_key in ["scores", "raw_scores", "metrics", "component_scores"]:
        container = candidate.get(container_key)
        if isinstance(container, dict) and score_name in container:
            try:
                return float(container[score_name])
            except (TypeError, ValueError):
                pass

    try:
        return float(candidate.get(score_name, default))
    except (TypeError, ValueError):
        return default


def get_candidate_filename(candidate):
    if not isinstance(candidate, dict):
        return "Unknown"
    return candidate.get("filename", candidate.get("name", "Unknown"))


def get_final_score(candidate):
    if not isinstance(candidate, dict):
        return 0.0
    try:
        return float(candidate.get("final_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_rank(candidate, fallback="-"):
    if not isinstance(candidate, dict):
        return fallback
    return candidate.get("rank", fallback)


def get_candidate_skills(candidate):
    if not isinstance(candidate, dict):
        return []
    skills = candidate.get("skills", [])
    if skills is None:
        return []
    if isinstance(skills, str):
        return [skills]
    try:
        return list(skills)
    except TypeError:
        return []


def score_badge_class(pct):
    """Return a CSS class name for a 0-100 match percentage."""
    if pct >= 75:
        return "score-high"
    if pct >= 50:
        return "score-mid"
    return "score-low"


def score_badge_html(pct):
    return f'<span class="score-badge {score_badge_class(pct)}">{pct:.0f}% match</span>'


# ---------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------------------------

st.sidebar.header("Matching Settings")

weight_mode = st.sidebar.radio(
    "How should candidates be scored?",
    [
        "Auto-recommended (based on the job description)",
        "Set weights manually",
    ],
)

if weight_mode == "Auto-recommended (based on the job description)":
    st.sidebar.caption(
        "Weighting is calculated automatically from the job description once it's uploaded."
    )
    weights = {
        "tfidf_cosine": 0.30,
        "skill_jaccard": 0.35,
        "semantic": 0.35,
    }
else:
    w_tfidf = st.sidebar.slider("Keyword Match importance", 0.0, 1.0, 0.30, 0.05)
    w_jaccard = st.sidebar.slider("Skill Coverage importance", 0.0, 1.0, 0.30, 0.05)
    w_semantic = st.sidebar.slider("Context Match importance", 0.0, 1.0, 0.40, 0.05)

    total_w = w_tfidf + w_jaccard + w_semantic

    if total_w > 0:
        weights = {
            "tfidf_cosine": w_tfidf / total_w,
            "skill_jaccard": w_jaccard / total_w,
            "semantic": w_semantic / total_w,
        }
    else:
        weights = {
            "tfidf_cosine": 0.30,
            "skill_jaccard": 0.30,
            "semantic": 0.40,
        }


# ---------------------------------------------------------------------------
# 1. UPLOAD & BATCH INGESTION
# ---------------------------------------------------------------------------

st.subheader("1. Upload the job description and resumes")

col1, col2 = st.columns([1, 2])

with col1:
    jd_file = st.file_uploader(
        "Job description (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        key="jd",
    )

with col2:
    resume_files = st.file_uploader(
        "Resumes (.pdf, .docx, .txt) — upload as many as you like",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resumes",
    )

selected_resumes = []


# ---------------------------------------------------------------------------
# 2. HIGH-CAPACITY CANDIDATE BATCH SELECTION
# ---------------------------------------------------------------------------

if resume_files:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader(f"2. Choose which resumes to include ({len(resume_files)} uploaded)")

    ctrl_col1, ctrl_col2, search_col, filter_col = st.columns([1, 1, 2, 1.5])

    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = {f.name: True for f in resume_files}

    current_names = {f.name for f in resume_files}
    st.session_state["selected_files"] = {
        name: selected
        for name, selected in st.session_state["selected_files"].items()
        if name in current_names
    }

    for f in resume_files:
        if f.name not in st.session_state["selected_files"]:
            st.session_state["selected_files"][f.name] = True

    with ctrl_col1:
        if st.button("Select all", use_container_width=True):
            st.session_state["selected_files"] = {f.name: True for f in resume_files}
            st.rerun()

    with ctrl_col2:
        if st.button("Deselect all", use_container_width=True):
            st.session_state["selected_files"] = {f.name: False for f in resume_files}
            st.rerun()

    with search_col:
        search_query = st.text_input(
            "Search by filename",
            "",
            placeholder="Type a candidate name...",
        )

    with filter_col:
        ext_filter = st.selectbox(
            "File type",
            ["All types", ".pdf", ".docx", ".txt"],
        )

    filtered_files = []
    for rf in resume_files:
        matches_search = search_query.lower() in rf.name.lower()
        matches_ext = (ext_filter == "All types") or rf.name.lower().endswith(ext_filter)
        if matches_search and matches_ext:
            filtered_files.append(rf)

    st.caption(f"Showing {len(filtered_files)} of {len(resume_files)} resumes")

    with st.container(height=260):
        grid_cols = st.columns(4)
        for idx, rf in enumerate(filtered_files):
            with grid_cols[idx % 4]:
                is_checked = st.checkbox(
                    rf.name,
                    value=st.session_state["selected_files"].get(rf.name, True),
                    key=f"chk_{rf.name}",
                )
                st.session_state["selected_files"][rf.name] = is_checked

    selected_resumes = [
        rf for rf in resume_files
        if st.session_state["selected_files"].get(rf.name, False)
    ]

    st.info(f"{len(selected_resumes)} of {len(resume_files)} resumes selected for evaluation.")


# ---------------------------------------------------------------------------
# 3. PIPELINE EXECUTION
# ---------------------------------------------------------------------------

if jd_file and selected_resumes:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button(f"Check {len(selected_resumes)} resumes against the job description", type="primary"):
        with st.spinner("Reading documents and comparing against the job description..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Process JD
                jd_path = os.path.join(tmpdir, jd_file.name)
                with open(jd_path, "wb") as f:
                    f.write(jd_file.getvalue())

                jd_data = doc_parser.process_pdf(
                    jd_path,
                    cache_dir=f"{cache_dir}/parsed",
                )

                # Execute Auto-Weight Calculation if enabled
                if weight_mode == "Auto-recommended (based on the job description)" and hasattr(
                    ranker, "calculate_auto_weights"
                ):
                    try:
                        auto_res = ranker.calculate_auto_weights(
                            jd_data.get("raw_text", ""),
                            jd_data.get("skills", [])
                        )
                        weights = {
                            "tfidf_cosine": auto_res.get("tfidf_cosine", 0.30),
                            "skill_jaccard": auto_res.get("skill_jaccard", 0.35),
                            "semantic": auto_res.get("semantic", 0.35),
                        }
                    except Exception as e:
                        st.warning(f"Using default weights — couldn't calculate custom weights ({e}).")

                # Process Resumes
                parsed_resumes = []
                for rf in selected_resumes:
                    r_path = os.path.join(tmpdir, rf.name)
                    with open(r_path, "wb") as f:
                        f.write(rf.getvalue())

                    parsed_data = doc_parser.process_pdf(
                        r_path,
                        cache_dir=f"{cache_dir}/parsed",
                    )
                    parsed_resumes.append(parsed_data)

                # Rank Candidates
                ranked = ranker.rank_candidates(
                    jd_data=jd_data,
                    resumes=parsed_resumes,
                    weights=weights,
                    semantic_cache_dir=f"{cache_dir}/embeddings",
                )

                if ranked is None or not isinstance(ranked, list):
                    ranked = list(ranked) if ranked else []

                st.session_state["ranked"] = ranked
                st.session_state["jd_skills"] = jd_data.get("skills", [])
                st.session_state["jd_raw_text"] = jd_data.get("raw_text", "")

                st.success(f"Done — {len(ranked)} resumes checked.")


# ---------------------------------------------------------------------------
# 4. RESULTS, LEADERBOARD & GRAPHICAL ANALYTICS
# ---------------------------------------------------------------------------

if "ranked" in st.session_state:
    ranked = st.session_state["ranked"]
    jd_skills = st.session_state.get("jd_skills", [])

    if jd_skills is None:
        jd_skills = []

    # JD BIAS AUDIT
    if hasattr(explainer, "analyze_jd_bias_and_rigidity") and "jd_raw_text" in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Job description review")
        st.caption("Flags wording in the job post that may be unclear, overly rigid, or unintentionally exclusionary.")
        try:
            flags = explainer.analyze_jd_bias_and_rigidity(st.session_state["jd_raw_text"])
            if flags:
                for flag in flags:
                    st.write(flag)
            else:
                st.success("No issues found with the job description wording.")
        except Exception as e:
            st.warning(f"Couldn't complete the job description review: {e}")

    # LEADERBOARD
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Results")

    if not ranked:
        st.warning("No results to show.")
    else:
        # Score-card style summary for each candidate
        for c in ranked:
            if not isinstance(c, dict):
                continue

            final_score = get_final_score(c)
            pct = max(0.0, min(final_score, 1.0)) * 100 if final_score <= 1.0 else final_score
            filename = get_candidate_filename(c)
            rank = get_rank(c)
            skill_pct = get_score(c, "skill_jaccard") * 100
            skills = get_candidate_skills(c)

            st.markdown(
                f"""
                <div class="candidate-card">
                    <span class="rank-pill">#{rank}</span>
                    <strong>{filename}</strong>
                    &nbsp;&nbsp;{score_badge_html(pct)}
                    <div style="color:#6b7280; font-size:0.85rem; margin-top:0.4rem;">
                        Skill coverage: {skill_pct:.0f}% &nbsp;·&nbsp;
                        Matched skills: {", ".join(map(str, skills)) if skills else "None detected"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View as a table"):
            table_data = []
            for c in ranked:
                if not isinstance(c, dict):
                    continue

                tfidf_val = get_score(c, "tfidf_cosine")
                jaccard_val = get_score(c, "skill_jaccard")
                semantic_val = get_score(c, "semantic")
                candidate_skills = get_candidate_skills(c)

                table_data.append({
                    "Rank": f"#{get_rank(c)}",
                    "Candidate": get_candidate_filename(c),
                    "Overall Match": round(get_final_score(c), 4),
                    "Keyword Match": round(tfidf_val, 2),
                    "Skill Coverage": f"{round(jaccard_val * 100)}%",
                    "Context Match": round(semantic_val, 2),
                    "Matched Skills": ", ".join(map(str, candidate_skills)),
                })

            if table_data:
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # RELATIVE COMPARISON
    if len(ranked) > 1:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("Why the top candidate ranked higher")

        top_cand = ranked[0]
        top_filename = get_candidate_filename(top_cand)
        top_final_score = get_final_score(top_cand)
        top_rank = get_rank(top_cand, 1)  # type: ignore

        st.subheader(f"#{top_rank}: {top_filename} — Overall Match {top_final_score:.2f}")

        for other_cand in ranked[1:]:
            other_filename = get_candidate_filename(other_cand)
            other_rank = get_rank(other_cand, "?")

            with st.expander(f"{top_filename} vs. {other_filename}"):
                if hasattr(explainer, "explain_relative_difference"):
                    try:
                        explanation = explainer.explain_relative_difference(top_cand, other_cand)
                        if explanation:
                            st.markdown(explanation)
                    except Exception as e:
                        st.warning(f"Couldn't generate a comparison: {e}")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**{top_filename}**")
                    st.metric("Skill Coverage", f"{get_score(top_cand, 'skill_jaccard') * 100:.0f}%")
                    st.metric("Context Match", f"{get_score(top_cand, 'semantic'):.2f}")
                    st.metric("Keyword Match", f"{get_score(top_cand, 'tfidf_cosine'):.2f}")

                with col_b:
                    st.markdown(f"**{other_filename}**")
                    st.metric("Skill Coverage", f"{get_score(other_cand, 'skill_jaccard') * 100:.0f}%")
                    st.metric("Context Match", f"{get_score(other_cand, 'semantic'):.2f}")
                    st.metric("Keyword Match", f"{get_score(other_cand, 'tfidf_cosine'):.2f}")

    # 5. RANKING ANALYTICS & SCORE BREAKDOWN
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Score breakdown")

    tab1, tab2, tab3 = st.tabs([
        "Match composition",
        "Candidate comparison",
        "Skill match grid",
    ])

    with tab1:
        st.subheader("What each candidate's score is made of")
        st.caption("Shows how much each candidate's overall score comes from keyword matches, required skills, and overall context alignment.")

        chart_data = []
        for c in ranked:
            if not isinstance(c, dict):
                continue
            chart_data.append({
                "Candidate": get_candidate_filename(c),
                "Keyword Match": get_score(c, "tfidf_cosine") * weights["tfidf_cosine"],
                "Skill Coverage": get_score(c, "skill_jaccard") * weights["skill_jaccard"],
                "Context Match": get_score(c, "semantic") * weights["semantic"],
            })

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig_bar = px.bar(
                df_chart,
                x="Candidate",
                y=["Keyword Match", "Skill Coverage", "Context Match"],
                title="Score breakdown by candidate",
                labels={"value": "Contribution to overall score", "variable": "Component"},
                barmode="stack",
                height=400,
            )
            fig_bar.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig_bar, use_container_width=True, key="analytics_stacked_bar_plot")

    with tab2:
        st.subheader("Top candidates side by side")
        st.caption("Compares the top matches across keyword match, skill coverage, and context match.")

        categories = ["Keyword Match", "Skill Coverage", "Context Match"]
        fig_radar = go.Figure()

        top_n = min(3, len(ranked))
        for idx in range(top_n):
            c = ranked[idx]
            if not isinstance(c, dict):
                continue

            tfidf_v = get_score(c, "tfidf_cosine")
            jaccard_v = get_score(c, "skill_jaccard")
            semantic_v = get_score(c, "semantic")
            candidate_rank = get_rank(c, idx + 1)  # type: ignore
            candidate_filename = get_candidate_filename(c)

            fig_radar.add_trace(go.Scatterpolar(
                r=[tfidf_v, jaccard_v, semantic_v, tfidf_v],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"#{candidate_rank} {candidate_filename}",
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="analytics_radar_plot")

    with tab3:
        st.subheader("Which required skills each candidate has")
        st.caption("Green means the skill was found in the resume; dark means it wasn't.")

        try:
            all_jd_skills = sorted(list(jd_skills))
        except TypeError:
            all_jd_skills = []

        if not all_jd_skills:
            all_skills_set = set()
            for c in ranked:
                all_skills_set.update(get_candidate_skills(c))
            all_jd_skills = sorted(list(all_skills_set))

        if all_jd_skills:
            matrix_rows = []
            candidates_list = []
            for c in ranked:
                candidates_list.append(get_candidate_filename(c))
                cand_skills = set(get_candidate_skills(c))
                matrix_rows.append([1 if sk in cand_skills else 0 for sk in all_jd_skills])

            fig_heatmap = px.imshow(
                matrix_rows,
                x=all_jd_skills,
                y=candidates_list,
                color_continuous_scale=["#262730", "#00CC96"],
                labels=dict(x="Skill", y="Candidate", color="Present"),
                height=max(300, 150 + len(ranked) * 40),
            )
            fig_heatmap.update_xaxes(side="top")
            st.plotly_chart(fig_heatmap, use_container_width=True, key="analytics_heatmap_plot")
        else:
            st.info("No skills were identified to compare.")

# ---------------------------------------------------------------------------
# 6. STANDALONE AUDIT MODE (NO JD PROVIDED)
# ---------------------------------------------------------------------------

elif resume_files and not jd_file:
    st.info(
        "No job description uploaded yet. You can still review resumes on their own "
        "to see what skills and sections were detected."
    )

    if selected_resumes and st.button(f"Review {len(selected_resumes)} resumes"):
        with st.spinner("Reading resumes..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                audited = []
                for rf in selected_resumes:
                    r_path = os.path.join(tmpdir, rf.name)
                    with open(r_path, "wb") as f:
                        f.write(rf.getvalue())
                    data = doc_parser.process_pdf(r_path, cache_dir=f"{cache_dir}/parsed")
                    audited.append(data)

                audit_table = []
                for res in audited:
                    skills = res.get("skills", [])
                    sections = res.get("sections", {})
                    detected_skills = ", ".join(sorted(map(str, skills))) if skills else "None"
                    sections_identified = ", ".join(list(sections.keys())) if sections else "None"

                    audit_table.append({
                        "Filename": res.get("filename", "Unknown"),
                        "How it was read": res.get("extraction_method", "Unknown"),
                        "Detected Skills": detected_skills,
                        "Sections Found": sections_identified,
                    })

                if audit_table:
                    st.dataframe(pd.DataFrame(audit_table), use_container_width=True)