
import re
from urllib.parse import quote_plus

import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SmartATS | AI Career Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# LIGHT LAVENDER + BLUE DESIGN
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #f5f0ff 0%,
            #eef5ff 52%,
            #fbfcff 100%
        );
        color: #262747;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eee8ff 0%, #e5efff 100%);
        border-right: 1px solid #dedcf8;
    }

    [data-testid="stSidebar"] * {
        color: #29264d;
    }

    .hero {
        background: linear-gradient(115deg, #6555ee 0%, #5689f5 58%, #42b9dc 100%);
        padding: 34px;
        border-radius: 24px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 32px rgba(80, 103, 230, 0.19);
    }

    .hero-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        opacity: 0.95;
    }

    .hero h1 {
        color: white !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        margin: 10px 0 !important;
    }

    .hero p {
        color: white !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 800;
        color: #302e60;
        margin: 12px 0 6px 0;
    }

    .section-subtitle {
        color: #747895;
        font-size: 14px;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e5e5fa;
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 5px 18px rgba(64, 72, 140, 0.05);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e5fa;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(64, 72, 140, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #737895;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #5c4ce5;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(110deg, #6555ee, #398eea);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        color: white;
        border: none;
        box-shadow: 0 6px 18px rgba(89, 91, 230, 0.25);
        transform: translateY(-1px);
    }

    .stTextArea textarea {
        background: white;
        border: 1px solid #dfe3f7;
        border-radius: 12px;
        color: #262747;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #b5b0f4;
        border-radius: 14px;
        padding: 10px;
    }

    .skill {
        display: inline-block;
        padding: 7px 11px;
        margin: 4px;
        border-radius: 9px;
        font-size: 13px;
        font-weight: 600;
    }

    .skill-good {
        background: #e3f8ed;
        color: #167647;
    }

    .skill-missing {
        background: #fff0e7;
        color: #b85422;
    }

    .job-card {
        background: white;
        border: 1px solid #e4e5fa;
        border-radius: 16px;
        padding: 19px;
        margin-bottom: 14px;
        box-shadow: 0 5px 18px rgba(64, 72, 140, 0.05);
    }

    .job-card h3 {
        color: #343064;
        margin: 0 0 8px 0;
        font-size: 19px;
    }

    .job-card p {
        color: #6f7593;
        font-size: 14px;
    }

    .score-pill {
        display: inline-block;
        background: #ece9ff;
        color: #5744d8;
        border-radius: 10px;
        padding: 7px 11px;
        font-weight: 800;
        font-size: 13px;
    }

    .small-note {
        color: #7a809d;
        font-size: 12px;
    }

    .footer {
        text-align: center;
        color: #858ba8;
        font-size: 12px;
        margin-top: 36px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SKILL DATABASE
# =========================================================
SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql",
    "postgresql", "mongodb", "machine learning",
    "deep learning", "artificial intelligence",
    "data science", "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "nlp",
    "natural language processing", "computer vision",
    "statistics", "excel", "power bi", "tableau",
    "matplotlib", "seaborn", "flask", "django", "fastapi",
    "streamlit", "html", "css", "javascript", "react",
    "node.js", "git", "github", "docker", "aws", "azure",
    "linux", "rest api", "data structures", "algorithms",
    "oop", "problem solving", "communication", "teamwork",
    "cloud computing", "data visualization", "etl",
    "data engineering", "spark", "hadoop", "pyspark",
    "cybersecurity", "networking", "testing",
    "software development", "firebase", "flutter",
    "android development", "kotlin", "ui ux",
    "project management", "powerpoint", "leadership",
    "generative ai", "large language models", "llm",
    "prompt engineering", "retrieval augmented generation",
    "rag", "api integration", "deployment",
    "model evaluation", "computer networks",
]


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


# =========================================================
# JOB ROLE DATABASE
# Edit or add roles here whenever you want.
# =========================================================
JOB_ROLES = [
    {
        "title": "Python Developer Intern",
        "category": "Software Development",
        "skills": [
            "python", "oop", "data structures",
            "algorithms", "git", "rest api",
            "problem solving",
        ],
        "description": "Build Python applications, automate tasks, and work with APIs.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "AI / ML Intern",
        "category": "Artificial Intelligence",
        "skills": [
            "python", "machine learning", "pandas", "numpy",
            "scikit-learn", "statistics", "model evaluation",
        ],
        "description": "Build and evaluate machine learning models and practical AI solutions.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Data Analyst Intern",
        "category": "Data Analytics",
        "skills": [
            "python", "sql", "excel", "power bi",
            "statistics", "data analysis",
            "data visualization", "pandas",
        ],
        "description": "Analyze datasets, find patterns, and communicate insights with reports.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Data Science Intern",
        "category": "Data Science",
        "skills": [
            "python", "sql", "statistics", "pandas",
            "numpy", "machine learning",
            "data visualization", "scikit-learn",
        ],
        "description": "Explore datasets, build predictive models, and communicate findings.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Generative AI Intern",
        "category": "Generative AI",
        "skills": [
            "python", "generative ai", "large language models",
            "llm", "prompt engineering", "api integration",
            "rest api", "rag",
        ],
        "description": "Create AI-powered applications using language models and APIs.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "NLP Intern",
        "category": "Artificial Intelligence",
        "skills": [
            "python", "nlp", "natural language processing",
            "machine learning", "deep learning",
            "pandas", "scikit-learn",
        ],
        "description": "Work on text classification, language processing, and text-based AI.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Computer Vision Intern",
        "category": "Artificial Intelligence",
        "skills": [
            "python", "computer vision", "deep learning",
            "machine learning", "numpy", "pytorch",
            "tensorflow",
        ],
        "description": "Build applications that analyze images and visual data.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Data Engineer Intern",
        "category": "Data Engineering",
        "skills": [
            "python", "sql", "etl", "data engineering",
            "pandas", "spark", "data structures",
        ],
        "description": "Prepare, transform, and manage data pipelines for analytics and AI.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Full Stack Developer Intern",
        "category": "Web Development",
        "skills": [
            "html", "css", "javascript", "react",
            "node.js", "rest api", "git", "mongodb",
        ],
        "description": "Build web interfaces and connect them to backend services.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Flutter Developer Intern",
        "category": "Mobile Development",
        "skills": [
            "flutter", "firebase", "api integration",
            "git", "android development",
        ],
        "description": "Develop mobile applications and integrate APIs or Firebase services.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Cloud / DevOps Intern",
        "category": "Cloud Engineering",
        "skills": [
            "linux", "git", "docker", "aws",
            "azure", "cloud computing", "deployment",
        ],
        "description": "Learn cloud deployment, development workflows, and infrastructure tools.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Cybersecurity Intern",
        "category": "Cybersecurity",
        "skills": [
            "cybersecurity", "networking", "linux",
            "computer networks", "python",
        ],
        "description": "Learn security fundamentals, network concepts, and defensive practices.",
        "level": "Entry-level / Internship",
    },
    {
        "title": "Software Testing Intern",
        "category": "Software Quality",
        "skills": [
            "python", "testing", "problem solving",
            "software development", "git",
        ],
        "description": "Test applications, identify defects, and help improve software quality.",
        "level": "Entry-level / Internship",
    },
]


# =========================================================
# LOAD AI MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# READ RESUME PDF
# =========================================================
def extract_resume_text(uploaded_file):
    text_parts = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")

    return "\n".join(text_parts).strip()


# =========================================================
# MATCH RESUME TO JOB DESCRIPTION
# =========================================================
def calculate_similarity(text1, text2, model):
    embeddings = model.encode(
        [text1, text2],
        normalize_embeddings=True,
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    return round(max(0, min(100, float(similarity) * 100)))


# =========================================================
# RECOMMEND BEST-FIT JOB ROLES
# =========================================================
def recommend_jobs(resume_text, resume_skills):
    recommendations = []

    for role in JOB_ROLES:
        role_skills = role["skills"]

        matched = sorted(set(resume_skills) & set(role_skills))
        missing = sorted(set(role_skills) - set(resume_skills))

        if role_skills:
            skill_coverage = len(matched) / len(role_skills)
        else:
            skill_coverage = 0

        # Role-specific text relevance and skill coverage
        role_text = (
            role["title"] + " " +
            role["description"] + " " +
            " ".join(role_skills)
        )

        resume_words = set(re.findall(r"[a-zA-Z+#.]+", resume_text.lower()))
        role_words = set(re.findall(r"[a-zA-Z+#.]+", role_text.lower()))

        if role_words:
            word_overlap = len(resume_words & role_words) / len(role_words)
        else:
            word_overlap = 0

        # Skill coverage is the main factor.
        # Text overlap is only a small secondary signal.
        score = round(
            100 * (0.85 * skill_coverage + 0.15 * min(word_overlap, 1))
        )

        recommendations.append({
            "title": role["title"],
            "category": role["category"],
            "description": role["description"],
            "level": role["level"],
            "score": score,
            "matched": matched,
            "missing": missing,
        })

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 20px 0;">
            <div style="font-size:38px;">🤖</div>
            <div style="font-size:26px; font-weight:800; color:#5041c8;">
                SmartATS
            </div>
            <div style="font-size:12px; color:#65668c;">
                AI CAREER ASSISTANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🧭 Your workspace")
    st.markdown("🏠 Resume Analyzer")
    st.markdown("📊 Match Analytics")
    st.markdown("💼 Best-fit Jobs")
    st.markdown("🎯 Skill Gap Analysis")
    st.markdown("📚 Learning Resources")

    st.markdown("---")
    st.markdown("### ✨ What SmartATS does")
    st.caption("• Resume and job description matching")
    st.caption("• Skills detected in your resume")
    st.caption("• Job role recommendations")
    st.caption("• Suggested skills to learn")

    st.markdown("---")
    st.caption("Designed for students and aspiring tech professionals.")


# =========================================================
# HERO HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">✦ YOUR AI-POWERED CAREER COMPANION</div>
        <h1>SmartATS 🤖</h1>
        <p style="font-size:19px; font-weight:600;">
            Smarter Resumes. Better Opportunities.
        </p>
        <p style="margin-top:12px;">
            Discover your strengths, close skill gaps, and find career roles
            that align with your resume.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">🚀 Analyze your resume</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Upload a resume and paste a job description to analyze your fit and discover other possible roles.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# INPUT CARDS
# =========================================================
left, right = st.columns([0.9, 1.1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("### 📄 Your Resume")
        st.caption("Upload a text-based PDF for best results.")

        uploaded_file = st.file_uploader(
            "Choose your resume PDF",
            type=["pdf"],
        )

        st.markdown(
            '<p class="small-note">Your resume text is processed by this app for analysis.</p>',
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        st.markdown("### 🎯 Target Job Description")
        st.caption("Optional for job-role suggestions; required for match analysis.")

        job_description = st.text_area(
            "Paste job description",
            height=190,
            placeholder=(
                "Paste the responsibilities and requirements of the job here..."
            ),
        )

analyze_clicked = st.button(
    "✨ Analyze Resume & Find Best-fit Jobs",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PROCESS
# =========================================================
if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload your resume PDF first.")
    else:
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_resume_text(uploaded_file)

            if not resume_text:
                st.error(
                    "No readable text was found. Please upload a text-based PDF instead of a scanned image."
                )
                st.stop()

            resume_skills = extract_skills(resume_text)

            with st.spinner("Finding suitable career roles..."):
                job_recommendations = recommend_jobs(
                    resume_text,
                    resume_skills,
                )

            match_score = None
            matched_jd_skills = []
            missing_jd_skills = []

            if job_description.strip():
                with st.spinner("Comparing resume with the job description..."):
                    model = load_model()

                    match_score = calculate_similarity(
                        resume_text,
                        job_description,
                        model,
                    )

                    jd_skills = extract_skills(job_description)
                    matched_jd_skills = sorted(
                        set(resume_skills) & set(jd_skills)
                    )
                    missing_jd_skills = sorted(
                        set(jd_skills) - set(resume_skills)
                    )

            st.session_state["smartats_results"] = {
                "resume_text": resume_text,
                "resume_skills": resume_skills,
                "recommendations": job_recommendations,
                "match_score": match_score,
                "matched_jd_skills": matched_jd_skills,
                "missing_jd_skills": missing_jd_skills,
                "job_description": job_description,
            }

        except Exception as error:
            st.error("The analysis could not be completed.")
            st.exception(error)


# =========================================================
# RESULTS
# =========================================================
results = st.session_state.get("smartats_results")

if results:
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">📊 Your Career Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Insights generated from your resume and the skills detected by SmartATS.</div>',
        unsafe_allow_html=True,
    )

    resume_skills = results["resume_skills"]
    recommendations = results["recommendations"]
    match_score = results["match_score"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Skills Detected", len(resume_skills))
    col2.metric(
        "Job Roles Compared",
        len(recommendations),
    )
    col3.metric(
        "Best Role Match",
        f"{recommendations[0]['score']}%" if recommendations else "N/A",
    )
    col4.metric(
        "JD Similarity",
        f"{match_score}%" if match_score is not None else "Not provided",
    )

    if match_score is not None:
        st.progress(match_score / 100)
        st.caption(
            "JD similarity is a text-similarity estimate, not a probability of getting hired."
        )

    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------
    st.markdown("### 🧩 Your Resume Skills")

    if resume_skills:
        skills_html = "".join(
            f'<span class="skill skill-good">{skill.title()}</span>'
            for skill in resume_skills
        )
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.warning(
            "No skills from the current dictionary were detected. Try using standard skill names in your resume."
        )

    if results["job_description"].strip():
        skill_col1, skill_col2 = st.columns(2, gap="large")

        with skill_col1:
            with st.container(border=True):
                st.markdown("#### ✅ Skills matched to the target job")

                if results["matched_jd_skills"]:
                    html = "".join(
                        f'<span class="skill skill-good">{s.title()}</span>'
                        for s in results["matched_jd_skills"]
                    )
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("No common skills detected in the current skill dictionary.")

        with skill_col2:
            with st.container(border=True):
                st.markdown("#### 🎯 Potential skill gaps")

                if results["missing_jd_skills"]:
                    html = "".join(
                        f'<span class="skill skill-missing">{s.title()}</span>'
                        for s in results["missing_jd_skills"]
                    )
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.success("No skill gaps detected from the current dictionary.")

    # -----------------------------------------------------
    # BEST-FIT JOB RECOMMENDATIONS
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">💼 Best-fit Career Roles</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Ranked by detected skill overlap and a lightweight text-overlap signal. These are suggested roles, not verified vacancies.</div>',
        unsafe_allow_html=True,
    )

    filter_choice = st.selectbox(
        "Filter career areas",
        [
            "All career areas",
            "Artificial Intelligence",
            "Data Analytics",
            "Data Science",
            "Software Development",
            "Web Development",
            "Mobile Development",
            "Data Engineering",
            "Generative AI",
            "Cloud Engineering",
            "Cybersecurity",
            "Software Quality",
        ],
    )

    filtered_recommendations = recommendations

    if filter_choice != "All career areas":
        filtered_recommendations = [
            role for role in recommendations
            if role["category"] == filter_choice
        ]

    if not filtered_recommendations:
        st.info("No roles are listed for this category yet.")
    else:
        for index, role in enumerate(filtered_recommendations, start=1):
            with st.container(border=True):
                title_col, score_col = st.columns([3, 1])

                with title_col:
                    st.markdown(f"### {index}. {role['title']}")
                    st.caption(
                        f"{role['category']} · {role['level']}"
                    )

                with score_col:
                    st.markdown(
                        f'<span class="score-pill">{role["score"]}% skill fit</span>',
                        unsafe_allow_html=True,
                    )

                st.write(role["description"])

                st.markdown("**Skills detected for this role**")

                if role["matched"]:
                    matched_html = "".join(
                        f'<span class="skill skill-good">{s.title()}</span>'
                        for s in role["matched"]
                    )
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.caption(
                        "No direct matches detected from this role's skill list."
                    )

                if role["missing"]:
                    with st.expander("📚 Skills you could learn next"):
                        for skill in role["missing"]:
                            url = (
                                "https://www.youtube.com/results?search_query="
                                + quote_plus(skill + " tutorial for beginners")
                            )
                            st.markdown(f"- [{skill.title()} tutorials]({url})")
                else:
                    st.success(
                        "All skills in this role's current skill list were detected."
                    )

                search_query = quote_plus(
                    role["title"] + " internship India"
                )
                st.markdown(
                    f"[🔎 Search for related internships on Google](https://www.google.com/search?q={search_query})"
                )

    # -----------------------------------------------------
    # MATCH ANALYTICS
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Role Comparison")

    chart_roles = filtered_recommendations[:8]

    if chart_roles:
        chart_data = {
            "Job Role": [r["title"] for r in chart_roles],
            "Skill Fit (%)": [r["score"] for r in chart_roles],
        }

        st.bar_chart(
            chart_data,
            x="Job Role",
            y="Skill Fit (%)",
            use_container_width=True,
        )

    st.info(
        "SmartATS suggests roles based on the skills it recognizes. The database is limited, skill wording affects the result, and a higher score does not guarantee a job or interview."
    )

    if st.button("🗑️ Clear Analysis"):
        del st.session_state["smartats_results"]
        st.rerun()


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        SmartATS · AI-powered resume and career-role analysis
        <br><br>
        Built with Python · Streamlit · Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True,
)
