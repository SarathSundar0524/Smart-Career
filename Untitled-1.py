import os
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.warning("sentence-transformers not installed or failed to import. Install requirements to run embedding-based matching.")
    SentenceTransformer = None

# Optional OpenAI for justifications
USE_OPENAI = False
try:
    import openai
    if os.getenv('OPENAI_API_KEY'):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# Paths
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
COURSES_CSV = DATA_DIR / "courses.csv"

# Sample courses CSV writer
SAMPLE_COURSES = [
    (1,"Python for Everybody","Coursera",8,"None","python;programming","beginner","https://example.com/python",49),
    (2,"Data Analysis with Pandas","Udemy",6,"python","python;pandas;data-analysis","beginner","https://example.com/pandas",20),
    (3,"SQL for Data Science","Coursera",4,"None","sql;databases","beginner","https://example.com/sql",0),
    (4,"Machine Learning by Andrew Ng","Coursera",11,"linear-algebra;python","ml;scikit-learn","intermediate","https://example.com/ml",49),
    (5,"Deep Learning Specialization","Coursera",16,"python;ml","deep-learning","advanced","https://example.com/dl",79),
    (6,"Intro to Embedded Systems","edX",6,"c-programming","embedded-c;microcontrollers","beginner","https://example.com/embedded",0),
    (7,"Embedded Systems with ARM","Coursera",8,"c-programming;basic-electronics","arm;microcontrollers","intermediate","https://example.com/arm",49),
    (8,"Control Systems","edX",10,"calculus;linear-algebra","control-systems","intermediate","https://example.com/control",99),
    (9,"SolidWorks Essentials","Udemy",5,"None","solidworks;cad","beginner","https://example.com/solidworks",15),
    (10,"ROS for Robotics Engineers","Udacity",12,"python;linux","ros;robotics","intermediate","https://example.com/ros",199),
    (11,"Robotics Specialization","Coursera",20,"linear-algebra;programming","robotics;kinematics","advanced","https://example.com/robotics",79),
    (12,"MATLAB Onramp","MathWorks",3,"None","matlab;numerical","beginner","https://example.com/matlab",0),
    (13,"Git & GitHub","Udacity",4,"None","git;version-control","beginner","https://example.com/git",0),
    (14,"Microcontroller Programming (AVR)","Udemy",6,"c-programming;electronics","microcontrollers","intermediate","https://example.com/avr",25),
    (15,"Cloud Foundations (AWS)","Coursera",8,"None","aws;cloud","beginner","https://example.com/aws",49),
    (16,"DevOps Essentials","Udemy",8,"linux;git","ci-cd;docker","intermediate","https://example.com/devops",20),
    (17,"Data Engineering with Python","DataCamp",10,"python;sql","data-engineering","intermediate","https://example.com/dataeng",39),
    (18,"ML Ops Fundamentals","Coursera",8,"ml;python","mlops;deployment","intermediate","https://example.com/mlops",49),
    (19,"Full Stack Web Dev","FreeCodeCamp",12,"html;css;javascript","web;frontend","beginner","https://example.com/fullstack",0),
    (20,"Advanced C++ for Embedded","Udemy",10,"c++;embedded-systems","c++;embedded-systems","advanced","https://example.com/advcpp",30),
    (21,"Introduction to Networks","Cisco",6,"None","networking","beginner","https://example.com/networks",0),
    (22,"Google Data Analytics Certificate","Coursera",24,"None","data-analysis","beginner-intermediate","https://example.com/gda",49),
    (23,"Introduction to IoT","edX",6,"python;basic-electronics","iot;embedded","beginner","https://example.com/iot",0),
    (24,"Applied Reinforcement Learning","Udacity",12,"ml;python","rl;deep-learning","advanced","https://example.com/rl",399),
    (25,"Professional Certificate in Cybersecurity","IBM/Coursera",20,"None","cybersecurity","beginner-intermediate","https://example.com/cyber",99),
]

def write_sample_csv(path: Path):
    cols = ["id","title","provider","duration_weeks","prerequisites","skill_tags","level","link","cost_usd"]
    rows = []
    for r in SAMPLE_COURSES:
        prereq = r[4] if r[4] else ""
        rows.append({
            "id": r[0],
            "title": r[1],
            "provider": r[2],
            "duration_weeks": r[3],
            "prerequisites": prereq,
            "skill_tags": r[5],
            "level": r[6],
            "link": r[7],
            "cost_usd": r[8]
        })
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)
    return df

if not COURSES_CSV.exists():
    write_sample_csv(COURSES_CSV)

@st.cache_data(show_spinner=False)
def load_courses(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

courses_df = load_courses(COURSES_CSV)

# Build textual representations for embeddings
courses_df['doc_text'] = courses_df.apply(lambda r: f"{r.title} | {r.provider} | tags: {r.skill_tags} | prereq: {r.prerequisites} | level: {r.level}", axis=1)

# Initialize embedding model
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    if SentenceTransformer is None:
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = get_embedding_model()

if model is None:
    st.error("Embedding model not available. Install sentence-transformers to use embedding-based recommendations.")


@st.cache_data(show_spinner=False)
def compute_course_embeddings(_model, texts: List[str]):
    if _model is None:
        return None
    embs = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs

course_embs = compute_course_embeddings(model, courses_df['doc_text'].tolist())

# Helper: simple prereq check
def check_prereqs(user_skills: List[str], prereq_text: str) -> Dict[str, Any]:
    if not prereq_text or str(prereq_text).strip().lower() in ['none','nan','']:
        return {"missing": [], "fraction_missing": 0.0}
    tokens = [t.strip().lower() for t in prereq_text.replace(';',',').split(',') if t.strip()]
    user_lower = [s.lower() for s in user_skills]
    missing = [p for p in tokens if not any(p in u or u in p for u in user_lower)]
    frac = len(missing)/len(tokens) if tokens else 0.0
    return {"missing": missing, "fraction_missing": frac}

# Scoring formula
def compute_fit_score(sim: float, course_level: str, user_level: str, frac_missing_prereqs: float) -> float:
    base_score = (sim + 1)/2 * 100
    level_penalty = 0
    levels = {"beginner":0, "beginner-intermediate":1, "intermediate":2, "advanced":3}
    c_lvl = levels.get(course_level, 1)
    u_lvl = levels.get(user_level, 1)
    if c_lvl > u_lvl:
        level_penalty = -12 * (c_lvl - u_lvl)
    prereq_penalty = 20 * frac_missing_prereqs
    score = base_score + level_penalty - prereq_penalty
    return float(max(0, min(100, score)))

# LLM justification
def generate_justification(course: Dict[str,Any], profile: Dict[str,Any], use_openai: bool=False) -> Dict[str,str]:
    title = course.get('title')
    skill_tags = course.get('skill_tags','')
    prereq = course.get('prerequisites','')

    rationale = f"Matches your background by emphasizing {skill_tags.split(';')[0]} and practical {skill_tags.split(';')[-1]} skills relevant to {profile.get('target_domain','your target domain')}.'"
    gap = f"Fills the gap in {prereq if prereq else 'hands-on domain knowledge'}."
    prep = "No prep needed." if not prereq or str(prereq).lower() in ['none','nan',''] else f"Recommend preparing: {prereq}."
    timeline_reason = "short-term" if int(course.get('duration_weeks',12)) <= 12 else "long-term"

    return {"rationale": rationale, "gap": gap, "prep": prep, "timeline_reason": timeline_reason}

# UI
st.set_page_config(page_title="Personalized Learning Path Recommender", layout='wide')
st.title("Personalized Learning Path Recommender — Prototype")

with st.sidebar:
    st.header("User Profile")
    name = st.text_input("Name", value="Learner")
    education = st.selectbox("Education level", ["High School","Bachelors","Masters","B.Tech","MCA","Other"], index=3)
    major = st.text_input("Major / Degree (optional)", value="Mechatronics")
    tech_skills = st.text_area("Technical skills (comma-separated)", value="SolidWorks, MATLAB, Python")
    soft_skills = st.text_area("Soft skills (comma-separated)", value="problem-solving, teamwork")
    target_domain = st.text_input("Target domain (optional)", value="embedded robotics firmware")
    weekly_hours = st.number_input("Preferred weekly study hours", min_value=1, max_value=60, value=10)
    preferred_weeks = st.number_input("Preferred total study duration (weeks)", min_value=4, max_value=52, value=36)
    user_level = st.selectbox("Self-assessed level", ["beginner","beginner-intermediate","intermediate","advanced"], index=2)
    use_openai_checkbox = st.checkbox("Use OpenAI for richer justifications (requires OPENAI_API_KEY)", value=False)

profile = {
    "name": name,
    "education": education,
    "major": major,
    "technical_skills": [s.strip() for s in tech_skills.split(',') if s.strip()],
    "soft_skills": [s.strip() for s in soft_skills.split(',') if s.strip()],
    "target_domain": target_domain,
    "weekly_hours": int(weekly_hours),
    "preferred_weeks": int(preferred_weeks),
    "user_level": user_level
}

st.markdown("---")
col1, col2 = st.columns([2,3])

with col1:
    st.subheader("Courses Catalog")
    st.write(f"Loaded {len(courses_df)} courses from `{COURSES_CSV}`")
    if st.checkbox("Show course table"):
        st.dataframe(courses_df.drop(columns=['doc_text']))

with col2:
    st.subheader("Recommendation Controls")
    top_k = st.slider("Number of recommendations", min_value=3, max_value=15, value=7)
    compute_btn = st.button("Generate Recommendations")

# Recommendation pipeline
if compute_btn:
    if model is None or course_embs is None:
        st.error("Embedding model unavailable. Cannot compute recommendations.")
    else:
        with st.spinner("Computing recommendations..."):
            profile_text = f"Education: {profile['education']}. Major: {profile['major']}. Skills: {', '.join(profile['technical_skills'])}. Soft: {', '.join(profile['soft_skills'])}. Goal: {profile['target_domain']}."
            p_emb = model.encode(profile_text, convert_to_numpy=True)

            sims = cosine_similarity([p_emb], course_embs)[0]

            candidates = []
            for idx, row in courses_df.iterrows():
                sim = float(sims[idx])
                prereq_check = check_prereqs(profile['technical_skills'], row['prerequisites'])
                fit_score = compute_fit_score(sim, str(row['level']), profile['user_level'], prereq_check['fraction_missing'])
                bucket = 'short-term' if (row['duration_weeks'] <= 12 and fit_score >= 50) else 'long-term'
                candidates.append({
                    'id': int(row['id']),
                    'title': row['title'],
                    'provider': row['provider'],
                    'duration_weeks': int(row['duration_weeks']),
                    'prerequisites': row['prerequisites'],
                    'skill_tags': row['skill_tags'],
                    'level': row['level'],
                    'link': row['link'],
                    'cost_usd': float(row['cost_usd']) if not pd.isna(row['cost_usd']) else 0.0,
                    'sim': sim,
                    'fit_score': fit_score,
                    'bucket': bucket,
                    'prereq_missing': prereq_check['missing']
                })

            sorted_cand = sorted(candidates, key=lambda x: x['fit_score'], reverse=True)[:top_k]

            recs = []
            for c in sorted_cand:
                just = generate_justification(c, profile, use_openai_checkbox)
                rec = {**c, **just}
                recs.append(rec)

            short_term = [r for r in recs if r['bucket'] == 'short-term']
            long_term = [r for r in recs if r['bucket'] == 'long-term']

            timeline = {
                'short_term_sequence': ' -> '.join([f"{r['title']} ({r['duration_weeks']}w)" for r in short_term]) if short_term else "None",
                'long_term_sequence': ' -> '.join([f"{r['title']} ({r['duration_weeks']}w)" for r in long_term]) if long_term else "None",
                'explanation': "Short-term picks are <=12 weeks with high fit; long-term are longer or advanced courses."
            }

            output = {
                'profile_id': profile.get('name'),
                'recommendations': recs,
                'timeline': timeline
            }

        st.success("Recommendations ready")
        st.markdown("---")
        st.subheader("Recommendations (cards)")
        for r in recs:
            st.markdown(f"**{r['title']}** — {r['provider']}  ")
            cols = st.columns([1,3,1])
            cols[0].metric("Fit score", f"{int(r['fit_score'])}")
            cols[1].markdown(f"**Level:** {r['level']} • **Duration:** {r['duration_weeks']} weeks  \n**Tags:** {r['skill_tags']}\n\n**Rationale:** {r['rationale']}\n\n**Gap:** {r['gap']}\n\n**Prep:** {r['prep']}")
            cols[2].write(f"[Link]({r['link']})  \nCost: ${r['cost_usd']}")
            st.markdown("---")

        st.subheader("JSON output")
        st.json(output)
        json_bytes = json.dumps(output, indent=2).encode('utf-8')
        st.download_button("Download JSON", data=json_bytes, file_name=f"recommendations_{profile.get('name','user')}.json")

        if st.button("Save recommendations to file (local)"):
            out_path = DATA_DIR / f"recs_{profile.get('name','user')}.json"
            with open(out_path,'w',encoding='utf-8') as f:
                json.dump(output,f,indent=2)
            st.success(f"Saved to {out_path}")

st.markdown("---")
st.caption("Prototype — explanations are templated unless OpenAI API is enabled. Replace templates and tune scoring for production.")
