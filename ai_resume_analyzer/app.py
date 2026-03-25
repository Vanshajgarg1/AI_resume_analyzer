"""
app.py
------
Streamlit UI for the AI Resume Analyzer.
Supports: standard analysis and Job Description comparison mode.

Run with:
    streamlit run app.py
"""

import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from docx import Document
from resume_parser import extract_resume_text
from analyzer import ResumeAnalyzer
from nlp_extractor import extract_skills_nlp, extract_sections, compute_keyword_density
from report_generator import generate_markdown_report, generate_txt_report
from utils import (
    clean_resume_text,
    truncate_text,
    validate_resume_text,
    validate_job_role,
    get_ats_color,
    get_ats_label,
    estimate_token_count,
    highlight_keywords_in_text,
    compute_skill_match_percent,
    safe_int,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0c29 40%, #1a1040 70%, #0d1b2a 100%);
    background-attachment: fixed;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(139,92,246,0.35); }

/* ── Section titles ── */
.sec-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

/* ── Metric big number ── */
.big-score {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.2rem;
}

/* ── Pills ── */
.pill          { display:inline-block; background:rgba(139,92,246,.18); border:1px solid rgba(139,92,246,.45); color:#c4b5fd; border-radius:30px; padding:.22rem .78rem; margin:.18rem; font-size:.8rem; font-weight:500; }
.pill-green    { background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#86efac; }
.pill-red      { background:rgba(239,68,68,.15); border:1px solid rgba(239,68,68,.4); color:#fca5a5; }
.pill-yellow   { background:rgba(234,179,8,.15); border:1px solid rgba(234,179,8,.4); color:#fde68a; }
.pill-blue     { background:rgba(96,165,250,.15); border:1px solid rgba(96,165,250,.4); color:#bfdbfe; }

/* ── Sub-score grid ── */
.sub-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-top:0.8rem; }
.sub-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:0.9rem; text-align:center; }
.sub-val  { font-size:1.8rem; font-weight:800; }
.sub-lbl  { font-size:0.72rem; color:#94a3b8; margin-top:0.15rem; }

/* ── Hero ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub { text-align:center; color:#64748b; font-size:1rem; margin-bottom:1.5rem; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 14px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,10,26,0.9);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-weight:600; font-size:0.88rem; }
[data-testid="stTabs"] [aria-selected="true"] { color:#a78bfa !important; border-bottom-color:#a78bfa !important; }

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg,#7c3aed,#60a5fa); }

/* ── Keyword highlights ── */
mark { background: rgba(234,179,8,0.35); color:#fde68a; border-radius:3px; padding:0 2px; }

/* ── JD match banner ── */
.jd-banner {
    background: linear-gradient(90deg, rgba(96,165,250,0.12), rgba(139,92,246,0.12));
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Before/after tip card ── */
.tip-card { background: rgba(255,255,255,0.03); border-left: 3px solid #7c3aed; border-radius:0 10px 10px 0; padding:.7rem 1rem; margin-bottom:.6rem; }

hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pills(items, style=""):
    css = f"pill {f'pill-{style}' if style else ''}".strip()
    return " ".join(f'<span class="{css}">{i}</span>' for i in items)


def card(content_html: str, extra_style: str = ""):
    return f'<div class="card" style="{extra_style}">{content_html}</div>'


def sub_score_card(label: str, value: int, color: str = "#a78bfa"):
    return f"""
    <div class="sub-card">
        <div class="sub-val" style="color:{color};">{value}</div>
        <div class="sub-lbl">{label}<br><small style="color:#475569;">/ 25</small></div>
    </div>"""


def render_ats_gauge(score: int):
    color = get_ats_color(score)
    label = get_ats_label(score)
    pct   = score / 100
    st.markdown(f"""
    <div class="card">
        <div class="sec-title">🎯 ATS Compatibility Score</div>
        <div class="big-score" style="color:{color};">{score}<span style="font-size:1.4rem;font-weight:400;color:#475569;">/100</span></div>
        <div style="color:{color};margin-bottom:.6rem;font-weight:600;">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct)


def render_skill_match_bar(pct: float, label: str = "Skill Match"):
    color = get_ats_color(int(pct))
    st.markdown(f"""
    <div style="margin-bottom:.4rem;">
        <span style="font-size:.85rem;color:#94a3b8;">{label}</span>
        <span style="float:right;font-weight:700;color:{color};">{pct:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct / 100)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key_input = st.text_input(
        "Groq API Key", type="password", placeholder="gsk_...",
        help="Overrides the .env file for this session. Get one free at console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    model_choice = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="llama-3.3-70b-versatile: best quality. llama3-8b-8192: fastest.",
    )

    st.markdown("---")
    st.markdown("### 📊 Analysis History")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-3:])):
            ats = h.get("ats_score", 0)
            role = h.get("job_role", "—")
            color = get_ats_color(ats)
            st.markdown(
                f'<div style="font-size:.8rem;padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.06);">'
                f'<span style="color:{color};font-weight:700;">{ats}</span>'
                f' &nbsp;·&nbsp; <span style="color:#94a3b8;">{role}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No analyses yet.")

    st.markdown("---")
    st.markdown("### 📌 Tips")
    st.markdown("""
- Upload a **text-based** PDF or DOCX
- Be specific: *"Senior Data Scientist"* beats *"Data Science"*
- Paste the **actual JD** for a precise match score
- Aim for **ATS ≥ 70** before applying
""")
    st.markdown("---")
    st.markdown('<small style="color:#334155;">Built with LangChain & OpenAI</small>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload your resume · Enter a role · Get instant AI-powered feedback</div>', unsafe_allow_html=True)
st.divider()


# ---------------------------------------------------------------------------
# Input Section
# ---------------------------------------------------------------------------

col_up, col_mid, col_jd = st.columns([1.1, 1, 1.3], gap="medium")

with col_up:
    st.markdown("### 📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Drop file here",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}**")
        st.caption(f"{uploaded_file.size / 1024:.1f} KB")

with col_mid:
    st.markdown("### 🎯 Target Job Role")
    job_role = st.text_input(
        "Job Role", placeholder="e.g., GenAI Engineer, Data Scientist",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button(
        "🚀 Analyze Resume", type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

with col_jd:
    st.markdown("### 📋 Job Description *(optional)*")
    job_description = st.text_area(
        "Paste JD here",
        placeholder="Paste the job description here for JD-comparison scoring...",
        height=120,
        label_visibility="collapsed",
    )
    if job_description.strip():
        st.info("🔍 JD detected — comparison mode enabled")

st.divider()


# ---------------------------------------------------------------------------
# Analysis + Results
# ---------------------------------------------------------------------------

if analyze_btn and uploaded_file:

    role_err = validate_job_role(job_role)
    if role_err:
        st.error(f"❌ {role_err}")
        st.stop()

    # Extract text
    with st.spinner("📖 Extracting resume text..."):
        try:
            raw_text = extract_resume_text(uploaded_file)
        except (ValueError, RuntimeError) as e:
            st.error(f"❌ **Extraction Error:** {e}")
            st.stop()

    resume_text = clean_resume_text(raw_text)
    text_err = validate_resume_text(resume_text)
    if text_err:
        st.error(f"❌ {text_err}")
        st.stop()

    resume_text = truncate_text(resume_text)

    # NLP extraction (local, no API)
    with st.spinner("🔬 Running local NLP extraction..."):
        nlp_result   = extract_skills_nlp(resume_text)
        sections     = extract_sections(resume_text)
        nlp_skills   = nlp_result.get("taxonomy_matched", [])

    # AI Analysis
    jd_mode = bool(job_description and job_description.strip())
    spinner_msg = (
        "🤖 Comparing resume against JD with AI..." if jd_mode
        else "🤖 Analysing resume with AI (10-20 seconds)..."
    )

    with st.spinner(spinner_msg):
        try:
            analyzer = ResumeAnalyzer(model_name=model_choice)
            if jd_mode:
                result = analyzer.analyze_with_jd(resume_text, job_role, job_description)
            else:
                result = analyzer.analyze(resume_text, job_role)
        except EnvironmentError as e:
            st.error(f"🔑 **API Key Error:** {e}\n\nGet a free Groq key at https://console.groq.com")
            st.stop()
        except RuntimeError as e:
            st.error(f"🤖 **AI Error:** {e}")
            st.stop()

    # Save to history
    result["job_role"] = job_role
    result["resume_filename"] = uploaded_file.name
    st.session_state.history.append(result)

    # Merge NLP skills with AI-detected skills (deduplicated)
    ai_skills  = result.get("skills_detected", [])
    all_skills = sorted(set(ai_skills + nlp_skills))
    result["skills_detected"] = all_skills

    missing_skills = result.get("missing_skills", [])
    keywords_add   = result.get("keywords_to_add", [])
    ats_score      = safe_int(result.get("ats_score", 0))
    breakdown      = result.get("ats_breakdown", {})
    jd_score       = result.get("jd_match_score")

    # Keyword density for all detected + missing skills
    all_kws      = all_skills + missing_skills + keywords_add
    kw_density   = compute_keyword_density(resume_text, [s.lower() for s in all_kws])

    # -------------------------------------------------------------------------
    # Results Header
    # -------------------------------------------------------------------------

    st.markdown(f"## 📊 Analysis Results")

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.metric("🎯 ATS Score", f"{ats_score}/100", delta=None)
    with col_h2:
        skill_pct = compute_skill_match_percent(all_skills, missing_skills + all_skills)
        st.metric("✅ Skills Detected", len(all_skills))
    with col_h3:
        if jd_score is not None:
            st.metric("📋 JD Match Score", f"{safe_int(jd_score)}/100")
        else:
            st.metric("❌ Missing Skills", len(missing_skills))

    # JD match banner
    if jd_score is not None:
        jd_color = get_ats_color(safe_int(jd_score))
        jd_label = get_ats_label(safe_int(jd_score))
        st.markdown(f"""
        <div class="jd-banner">
            <span style="font-weight:700;color:{jd_color};font-size:1.4rem;">{jd_score}/100</span>
            &nbsp;&nbsp;<span style="color:#94a3b8;font-size:.9rem;">JD Match · {jd_label}</span>
            &nbsp;·&nbsp;<span style="color:#60a5fa;font-size:.85rem;">
            ✅ {len(result.get('jd_matched_keywords',[]))} matched &nbsp;
            ❌ {len(result.get('jd_missing_keywords',[]))} missing
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------

    tab_overview, tab_ats, tab_skills, tab_improve, tab_keywords, tab_preview = st.tabs([
        "📝 Overview",
        "🎯 ATS Score",
        "🔍 Skills",
        "💡 Improvements",
        "🔑 Keywords",
        "👁️ Resume Preview",
    ])

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    with tab_overview:
        name = result.get("candidate_name", "Not found")
        exp  = result.get("years_of_experience", "Unknown")
        overall = result.get("overall_evaluation", "N/A")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card(
                f'<div class="sec-title">👤 Candidate</div>'
                f'<p style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">{name}</p>'
                f'<p style="color:#94a3b8;font-size:.9rem;">Experience: {exp}</p>'
                f'<p style="color:#94a3b8;font-size:.9rem;">Target Role: <strong style="color:#c4b5fd;">{job_role}</strong></p>'
            ), unsafe_allow_html=True)

        with c2:
            st.markdown(card(
                f'<div class="sec-title">📊 Quick Stats</div>'
                f'<p style="color:#94a3b8;font-size:.85rem;">Skills detected: <strong style="color:#86efac;">{len(all_skills)}</strong></p>'
                f'<p style="color:#94a3b8;font-size:.85rem;">Missing skills: <strong style="color:#fca5a5;">{len(missing_skills)}</strong></p>'
                f'<p style="color:#94a3b8;font-size:.85rem;">Keywords to add: <strong style="color:#fde68a;">{len(keywords_add)}</strong></p>'
                f'<p style="color:#94a3b8;font-size:.85rem;">Sections detected: <strong style="color:#bfdbfe;">{len(sections)}</strong></p>'
            ), unsafe_allow_html=True)

        st.markdown(card(
            f'<div class="sec-title">📝 Overall Evaluation</div>'
            f'<p style="color:#e2e8f0;line-height:1.75;font-size:.95rem;">{overall}</p>'
        ), unsafe_allow_html=True)

        # Section feedback
        sec_fb = result.get("section_feedback", {})
        if sec_fb:
            icons = {"summary": "🗒️", "experience": "💼", "education": "🎓", "skills": "⚙️"}
            st.markdown('<div class="sec-title" style="margin-top:.8rem;">📂 Section Feedback</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            items = list(sec_fb.items())
            for idx, (sec, tip) in enumerate(items):
                icon = icons.get(sec, "📌")
                card_html = (
                    f'<div style="font-weight:600;color:#a78bfa;margin-bottom:.3rem;">{icon} {sec.title()}</div>'
                    f'<div style="color:#cbd5e1;font-size:.88rem;">{tip}</div>'
                )
                if idx % 2 == 0:
                    f1.markdown(card(card_html), unsafe_allow_html=True)
                else:
                    f2.markdown(card(card_html), unsafe_allow_html=True)

        # Strengths
        strengths = result.get("strengths", [])
        if strengths:
            st.markdown("#### 💪 Strengths")
            for s in strengths:
                st.markdown(f"- ✨ {s}")

        # Recommended titles
        alt_titles = result.get("recommended_job_titles", [])
        if alt_titles:
            st.markdown("#### 🧭 Other Roles You May Fit")
            st.markdown(pills(alt_titles, "green"), unsafe_allow_html=True)

    # ── Tab 2: ATS Score ─────────────────────────────────────────────────────
    with tab_ats:
        render_ats_gauge(ats_score)

        ats_expl = result.get("ats_score_explanation", "")
        if ats_expl:
            st.caption(ats_expl)

        # Sub-score breakdown
        if breakdown:
            st.markdown("#### Sub-Score Breakdown")
            sub_labels = {
                "keyword_match":       ("🔍 Keyword Match", "#60a5fa"),
                "formatting":          ("📐 Formatting", "#a78bfa"),
                "section_completeness":("📂 Section Completeness", "#34d399"),
                "contact_info":        ("📞 Contact Info", "#fb923c"),
            }
            s1, s2, s3, s4 = st.columns(4)
            cols = [s1, s2, s3, s4]
            for i, (key, (label, color)) in enumerate(sub_labels.items()):
                val = safe_int(breakdown.get(key, 0))
                cols[i].markdown(
                    f'<div class="sub-card">'
                    f'<div class="sub-val" style="color:{color};">{val}</div>'
                    f'<div class="sub-lbl">{label}<br><small style="color:#475569;">/25</small></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                cols[i].progress(val / 25)

        # JD keywords
        if jd_score is not None:
            st.divider()
            st.markdown("#### 📋 JD Keyword Analysis")
            jd_matched_kws = result.get("jd_matched_keywords", [])
            jd_missing_kws = result.get("jd_missing_keywords", [])

            c_m, c_mi = st.columns(2)
            with c_m:
                st.markdown(f"**✅ Matched Keywords** ({len(jd_matched_kws)})")
                if jd_matched_kws:
                    st.markdown(pills(jd_matched_kws, "green"), unsafe_allow_html=True)
                else:
                    st.caption("None matched.")
            with c_mi:
                st.markdown(f"**❌ Missing JD Keywords** ({len(jd_missing_kws)})")
                if jd_missing_kws:
                    st.markdown(pills(jd_missing_kws, "red"), unsafe_allow_html=True)
                else:
                    st.caption("All JD keywords present!")

    # ── Tab 3: Skills ────────────────────────────────────────────────────────
    with tab_skills:

        # Skill match bar
        if missing_skills:
            total_ref = len(all_skills) + len(missing_skills)
            match_pct = round(len(all_skills) / total_ref * 100, 1)
        else:
            match_pct = 100.0

        render_skill_match_bar(match_pct, f"Skill Match for {job_role}")

        st.divider()

        sk1, sk2 = st.columns(2)

        with sk1:
            st.markdown(f"#### ✅ Detected Skills ({len(all_skills)})")

            # Separate AI + NLP sources
            nlp_only = [s for s in nlp_skills if s not in ai_skills]
            if ai_skills:
                st.markdown(pills(sorted(set(ai_skills)), "green"), unsafe_allow_html=True)
            if nlp_only:
                st.markdown(
                    '<span style="font-size:.75rem;color:#64748b;">▸ Also detected by NLP:</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(pills(sorted(nlp_only), "blue"), unsafe_allow_html=True)

        with sk2:
            st.markdown(f"#### ❌ Missing Skills ({len(missing_skills)})")
            if missing_skills:
                # Colour-code by priority: first 3 = critical, next 3 = recommended, rest = nice
                critical    = missing_skills[:3]
                recommended = missing_skills[3:6]
                nice        = missing_skills[6:]

                if critical:
                    st.markdown('<small style="color:#fca5a5;">🔴 Critical</small>', unsafe_allow_html=True)
                    st.markdown(pills(critical, "red"), unsafe_allow_html=True)
                if recommended:
                    st.markdown('<small style="color:#fde68a;">🟡 Recommended</small>', unsafe_allow_html=True)
                    st.markdown(pills(recommended, "yellow"), unsafe_allow_html=True)
                if nice:
                    st.markdown('<small style="color:#bfdbfe;">🔵 Nice to have</small>', unsafe_allow_html=True)
                    st.markdown(pills(nice, "blue"), unsafe_allow_html=True)
            else:
                st.success("🎉 All required skills detected — great match!")

        # Keyword density chart
        if kw_density:
            st.divider()
            st.markdown("#### 📊 Keyword Frequency in Resume")
            top_kws = dict(list(kw_density.items())[:12])
            st.bar_chart(top_kws)

    # ── Tab 4: Improvements ──────────────────────────────────────────────────
    with tab_improve:
        improvements = result.get("suggested_improvements", [])
        if improvements:
            st.markdown("#### 🔧 Suggested Improvements")
            for i, imp in enumerate(improvements, 1):
                st.markdown(
                    f'<div class="tip-card"><strong style="color:#a78bfa;">#{i}</strong> &nbsp; {imp}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No major improvements needed!")

        # Keyword optimization tips
        kw_tips = result.get("keyword_optimization_tips", [])
        if kw_tips:
            st.divider()
            st.markdown("#### ✍️ Keyword Optimization Tips")
            st.caption("Replace weak phrasing with ATS-optimised language:")
            for tip in kw_tips:
                st.markdown(
                    f'<div class="tip-card">💡 {tip}</div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 5: Keywords ──────────────────────────────────────────────────────
    with tab_keywords:
        keywords_to_add = result.get("keywords_to_add", [])
        jd_missing_kws  = result.get("jd_missing_keywords", [])

        all_missing_kws = sorted(set(keywords_to_add + jd_missing_kws))

        if all_missing_kws:
            st.markdown("#### 🔑 ATS Keywords to Add to Your Resume")
            st.caption(
                "These keywords appear in job postings for your target role but are absent from your resume."
            )
            st.markdown(pills(all_missing_kws, "yellow"), unsafe_allow_html=True)

            st.divider()

            # Keyword priority table
            st.markdown("#### 📋 Keyword Priority Table")
            kw_rows = []
            for kw in all_missing_kws:
                source = []
                if kw in keywords_to_add: source.append("AI Analysis")
                if kw in jd_missing_kws:  source.append("Job Description")
                importance = "🔴 Critical" if kw in (jd_missing_kws[:5] if jd_missing_kws else []) else "🟡 Recommended"
                kw_rows.append({"Keyword": kw, "Source": " + ".join(source), "Importance": importance})

            if kw_rows:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(kw_rows),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.success("✅ No missing keywords detected — your resume is well-optimised!")

    # ── Tab 6: Resume Preview ────────────────────────────────────────────────
    with tab_preview:
        st.markdown("#### 👁️ Resume Text with Highlighted Keywords")
        st.caption("Detected skills and ATS keywords are highlighted below.")

        highlight_kws = list(set(all_skills + keywords_add + missing_skills))
        highlighted   = highlight_keywords_in_text(
            resume_text.replace("\n", "<br>"), highlight_kws
        )

        with st.expander("Click to show/hide highlighted resume", expanded=True):
            st.markdown(
                f'<div style="background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.07);'
                f'border-radius:12px;padding:1.4rem;font-size:.85rem;line-height:1.8;'
                f'color:#cbd5e1;max-height:500px;overflow-y:auto;">'
                f'{highlighted}</div>',
                unsafe_allow_html=True,
            )

        # Section map
        if sections:
            st.divider()
            st.markdown("#### 📂 Detected Resume Sections")
            sec_icons = {
                "summary": "🗒️", "experience": "💼", "education": "🎓",
                "skills": "⚙️", "projects": "🚀", "certifications": "🏅",
                "achievements": "🏆", "publications": "📄",
            }
            for sec_name, snippet in sections.items():
                icon = sec_icons.get(sec_name, "📌")
                with st.expander(f"{icon} {sec_name.title()}"):
                    st.text(snippet[:300] + ("..." if len(snippet) > 300 else ""))

    # -------------------------------------------------------------------------
    # Download Section
    # -------------------------------------------------------------------------

    st.divider()
    st.markdown("#### ⬇️ Download Reports")

    md_report  = generate_markdown_report(result, job_role, uploaded_file.name)
    txt_report = generate_txt_report(result, job_role, uploaded_file.name)
    json_data  = json.dumps(result, indent=2)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "📄 Markdown Report (.md)",
            data=md_report,
            file_name=f"resume_analysis_{job_role.replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📃 Text Report (.txt)",
            data=txt_report,
            file_name=f"resume_analysis_{job_role.replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "🗂️ Raw JSON (.json)",
            data=json_data,
            file_name=f"resume_analysis_{job_role.replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.success("✅ Analysis complete! Use the tabs above to explore your results.")


elif not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#334155;">
        <div style="font-size:5rem;margin-bottom:1rem;">📄</div>
        <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;">Upload your resume to get started</div>
        <div style="font-size:.95rem;margin-top:.5rem;color:#64748b;">Supports PDF and DOCX · Instant AI analysis · No data stored</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🎯", "ATS Scoring", "Detailed ATS score with sub-category breakdown"),
        ("🔍", "NLP Extraction", "Local skill detection using spaCy taxonomy"),
        ("📋", "JD Comparison", "Match your resume against any job description"),
        ("⬇️", "Export Reports", "Download MD, TXT, or JSON reports")
    ]
    for col, (icon, title, desc) in zip([f1,f2,f3,f4], features):
        col.markdown(card(
            f'<div style="font-size:2rem;margin-bottom:.5rem;">{icon}</div>'
            f'<div style="font-weight:700;color:#e2e8f0;margin-bottom:.3rem;">{title}</div>'
            f'<div style="font-size:.82rem;color:#64748b;">{desc}</div>'
        ), unsafe_allow_html=True)
