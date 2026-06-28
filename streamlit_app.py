import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re, warnings
from collections import Counter
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─── DESIGN TOKENS (Linear + Framer) ────────────────────────────────────────
CANVAS    = "#010102"
SURFACE1  = "#0f1011"
SURFACE2  = "#141516"
HAIRLINE  = "#23252a"
HAIRLINE2 = "#34343a"
INK       = "#f7f8f8"
INK_MUTED = "#d0d6e0"
INK_SUB   = "#8a8f98"
ACCENT    = "#5e6ad2"
SUCCESS   = "#10b981"
WARNING   = "#f59e0b"
DANGER    = "#ef4444"

# ─── CHART DEFAULTS ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': CANVAS, 'axes.facecolor': SURFACE1,
    'axes.edgecolor': HAIRLINE2, 'axes.labelcolor': INK_SUB,
    'axes.titlecolor': INK, 'xtick.color': INK_SUB,
    'ytick.color': INK_SUB, 'text.color': INK_MUTED,
    'grid.color': HAIRLINE, 'grid.linewidth': 0.6,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentBridge — Resume Screening",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif !important; }}
  .stApp {{ background: {CANVAS}; color: {INK}; }}
  div[data-testid="stSidebar"] {{ background: {SURFACE1}; border-right: 1px solid {HAIRLINE}; }}
  div[data-testid="stSidebar"] * {{ color: {INK_MUTED} !important; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }}

  .eyebrow {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.10em;
    text-transform: uppercase; color: {INK_SUB}; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
  }}
  .eyebrow::before {{
    content: ''; display: inline-block; width: 16px; height: 1px;
    background: {ACCENT}; flex-shrink: 0;
  }}
  .display {{
    font-size: 30px; font-weight: 600; letter-spacing: -0.035em;
    line-height: 1.10; color: {INK}; margin-bottom: 6px;
  }}
  .kpi-card {{
    background: {SURFACE1}; border: 1px solid {HAIRLINE};
    border-radius: 12px; padding: 20px 24px;
  }}
  .kpi-num {{
    font-size: 40px; font-weight: 600; letter-spacing: -0.045em;
    line-height: 1; color: {INK};
  }}
  .kpi-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {INK_SUB}; margin-top: 8px;
  }}
  .pill-high   {{ display:inline-block; padding:3px 10px; border-radius:9999px; font-size:11px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; background:rgba(16,185,129,0.12); color:{SUCCESS}; border:1px solid rgba(16,185,129,0.25); }}
  .pill-medium {{ display:inline-block; padding:3px 10px; border-radius:9999px; font-size:11px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; background:rgba(245,158,11,0.12); color:{WARNING}; border:1px solid rgba(245,158,11,0.25); }}
  .pill-low    {{ display:inline-block; padding:3px 10px; border-radius:9999px; font-size:11px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; background:rgba(239,68,68,0.12); color:{DANGER};  border:1px solid rgba(239,68,68,0.25); }}
  .cand-row {{
    background: {SURFACE1}; border: 1px solid {HAIRLINE}; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 8px;
  }}
  .cand-name {{ font-size: 14px; font-weight: 600; color: {INK}; margin-bottom: 3px; letter-spacing: -0.01em; }}
  .cand-meta {{ font-size: 12px; color: {INK_SUB}; }}
  .sec-div {{ height: 1px; background: {HAIRLINE}; margin: 24px 0; }}
  .stSelectbox > div > div {{ background: {SURFACE1} !important; border-color: {HAIRLINE} !important; color: {INK} !important; }}
</style>
""", unsafe_allow_html=True)

# ─── NLP SETUP ───────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# REAL typos found in actual dataset (Ptyhon:39, Pythno:47)
TYPOS = {
    'Ptyhon': 'Python', 'Pythno': 'Python',
    'Pythoon': 'Python', 'Pyhton': 'Python',
    'Javaa': 'Java', 'JAva': 'Java',
}

# Skill normalization (actual skills in dataset)
SKILL_MAP = {
    'python': 'Python', 'ptyhon': 'Python', 'pythno': 'Python',
    'sql': 'SQL', 'mysql': 'SQL', 'postgresql': 'SQL',
    'ml': 'Machine Learning', 'machine learning': 'Machine Learning',
    'aws': 'AWS', 'amazon web services': 'AWS',
    'gcp': 'GCP', 'google cloud': 'GCP',
    'k8s': 'Kubernetes', 'kubernetes': 'Kubernetes',
    'docker': 'Docker', 'git': 'Git', 'linux': 'Linux',
    'java': 'Java', 'javascript': 'JavaScript', 'js': 'JavaScript',
    'tensorflow': 'TensorFlow', 'tf': 'TensorFlow',
    'sklearn': 'Scikit-Learn', 'scikit-learn': 'Scikit-Learn',
    'nlp': 'NLP', 'pandas': 'Pandas', 'numpy': 'NumPy',
    'power bi': 'Power BI', 'powerbi': 'Power BI',
    'devops': 'DevOps', 'ci/cd': 'CI/CD', 'ansible': 'Ansible',
}

# ─── JOB DESCRIPTIONS — Based on ACTUAL dataset roles & skills ───────────────
# Skills verified against actual parsed_resumes.csv
# Top skills in dataset: SQL(38.5%), Python(35.9%), AWS(27.8%), Git(27.6%),
# Docker(26%), Linux(25.9%), Kubernetes(25.4%), Java(25%), NLP(11.2%), TF(11.1%)
JOB_DESCRIPTIONS = {
    'Senior Software Engineer': {
        'mandatory': ['Python', 'Java', 'Git', 'SQL'],
        'optional':  ['Docker', 'Kubernetes', 'AWS', 'Linux'],
        'min_exp': 5, 'domain': 'IT',
        'jd_text': 'python java git sql docker kubernetes aws linux backend api rest senior software engineering code review system design'
    },
    'Cloud Engineer': {
        'mandatory': ['AWS', 'Docker', 'Kubernetes', 'Linux'],
        'optional':  ['Python', 'Git', 'SQL', 'Ansible'],
        'min_exp': 3, 'domain': 'IT',
        'jd_text': 'aws docker kubernetes linux cloud infrastructure devops automation deployment monitoring terraform ansible python git'
    },
    'DevOps Engineer': {
        'mandatory': ['Docker', 'Linux', 'Git', 'Kubernetes'],
        'optional':  ['AWS', 'Python', 'SQL', 'Ansible'],
        'min_exp': 3, 'domain': 'IT',
        'jd_text': 'docker linux git kubernetes devops ci cd pipeline automation aws python deployment monitoring scripting bash'
    },
    'Data Scientist': {
        'mandatory': ['Python', 'Machine Learning', 'SQL', 'Pandas'],
        'optional':  ['TensorFlow', 'NLP', 'Power BI', 'NumPy'],
        'min_exp': 2, 'domain': 'Data Science',
        'jd_text': 'python machine learning sql pandas tensorflow nlp data analysis feature engineering model training prediction scikit-learn'
    },
    'ML Engineer': {
        'mandatory': ['Python', 'Machine Learning', 'TensorFlow', 'SQL'],
        'optional':  ['NLP', 'Docker', 'AWS', 'Pandas'],
        'min_exp': 2, 'domain': 'Data Science',
        'jd_text': 'python tensorflow machine learning sql nlp docker aws pandas model deployment mlops pipeline deep learning scikit-learn'
    },
}

# ─── ACTUAL FLAG COLUMN NAMES (from parsed_resumes.csv) ─────────────────────
# Verified: all 16 flags end with _flag
ACTUAL_FLAG_COLS = [
    'management_experience_flag', 'people_management_flag',
    'project_management_experience_flag', 'agile_scrum_experience_flag',
    'client_facing_experience_flag', 'delivery_lead_experience_flag',
    'cloud_experience_flag', 'ml_experience_flag',
    'compliance_experience_flag', 'enterprise_systems_experience_flag',
    'offshore_onsite_model_experience_flag', 'multi_vendor_coordination_flag',
    'process_compliance_experience_flag', 'documentation_heavy_role_flag',
    'mentoring_experience_flag', 'stakeholder_management_experience_flag',
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def fix_typos(text):
    if pd.isna(text): return text
    words = str(text).split(',')
    fixed = [TYPOS.get(w.strip(), w.strip()) for w in words]
    return ', '.join(fixed)

def normalize_skills(text):
    if pd.isna(text): return []
    skills = [s.strip() for s in str(text).split(',')]
    return [SKILL_MAP.get(s.lower(), s) for s in skills if s]

def skill_match(skills, required):
    if not required: return 0.0
    skills_l = [s.lower() for s in skills]
    matched = sum(1 for r in required if r.lower() in skills_l)
    return round(matched / len(required) * 100, 1)

def exp_bucket(y):
    if y <= 2:  return 'Junior'
    elif y <= 5: return 'Mid'
    elif y <= 10: return 'Senior'
    else:        return 'Lead'

def preprocess(text):
    if pd.isna(text) or not str(text).strip(): return ''
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    try:    tokens = nltk.word_tokenize(text)
    except: tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens
                     if t not in stop_words and len(t) > 2])

def fit_color(label):
    return {
        'High Fit': SUCCESS, 'Medium Fit': WARNING, 'Low Fit': DANGER
    }.get(label, INK_SUB)

def fit_pill(label):
    css = {
        'High Fit':'pill-high', 'Medium Fit':'pill-medium', 'Low Fit':'pill-low'
    }.get(label, '')
    return f'<span class="{css}">{label}</span>'

def make_chart(fig):
    fig.patch.set_facecolor(CANVAS)
    return fig

def mono_shades(n):
    r,g,b = int(ACCENT[1:3],16), int(ACCENT[3:5],16), int(ACCENT[5:7],16)
    return [(r/255, g/255, b/255, a) for a in np.linspace(0.9, 0.3, n)]

# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline(df_raw, role):
    df  = df_raw.copy()
    jd  = JOB_DESCRIPTIONS[role]

    # ── 1. CLEAN ──────────────────────────────────────────────────────────────
    df['highest_education'] = df['highest_education'].fillna('Not Specified')
    df['education_field']   = df['education_field'].fillna('Not Specified')
    df = df.drop_duplicates().reset_index(drop=True)

    # Fix typos in actual skills (Ptyhon:39, Pythno:47 found in real data)
    df['technical_skills_raw'] = df['technical_skills_raw'].apply(fix_typos)

    # ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────
    df['tech_skills'] = df['technical_skills_raw'].apply(normalize_skills)
    df['tool_skills'] = df['tools_platforms_raw'].apply(normalize_skills)
    df['all_skills']  = df['tech_skills'] + df['tool_skills']

    # Skill scores
    df['mand_pct'] = df['all_skills'].apply(lambda s: skill_match(s, jd['mandatory']))
    df['opt_pct']  = df['all_skills'].apply(lambda s: skill_match(s, jd['optional']))
    df['skill_sc'] = (df['mand_pct'] * 0.70 + df['opt_pct'] * 0.30).round(1)

    # Experience score
    df['exp_bucket'] = df['years_experience'].apply(exp_bucket)
    df['exp_sc']     = df['years_experience'].apply(
        lambda y: min(100.0, round(y / max(jd['min_exp'], 1) * 100, 1))
    )

    # Domain match
    df['dom_match'] = (df['primary_domain'] == jd['domain']).astype(int)
    df['dom_pct']   = df['dom_match'] * 100

    # Education score
    edu_map  = {'Masters':100, 'MBA':90, 'LLM':80, 'Bachelors':70, 'Not Specified':50}
    tier_map = {'Tier-1':10, 'Tier-2':5, 'Tier-3':0}
    df['edu_sc']   = df['highest_education'].map(edu_map).fillna(60)
    df['tier_bon'] = df['institution_tier'].map(tier_map).fillna(0)
    df['edu_tot']  = (df['edu_sc'] + df['tier_bon']).clip(upper=100)

    # ATS composite — use actual flag columns from dataset
    flags_present = [c for c in ACTUAL_FLAG_COLS if c in df.columns]
    if flags_present:
        df['ats_sc'] = (df[flags_present].sum(axis=1) / len(flags_present) * 100).round(1)
    else:
        df['ats_sc'] = 50.0

    # ── 3. RULE-BASED SCORE (60% of hybrid) ──────────────────────────────────
    df['rule_sc'] = (
        df['skill_sc']  * 0.40 +
        df['exp_sc']    * 0.25 +
        df['dom_pct']   * 0.15 +
        df['edu_tot']   * 0.10 +
        df['ats_sc']    * 0.10
    ).clip(0, 100).round(2)

    # ── 4. TF-IDF NLP SCORE (40% of hybrid) ──────────────────────────────────
    # Combine actual text columns (all verified to exist in dataset)
    df['resume_text'] = (
        df['technical_skills_raw'].fillna('') + ' ' +
        df['tools_platforms_raw'].fillna('')   + ' ' +
        df['experience_summary'].fillna('')    + ' ' +
        df['project_summary'].fillna('')       + ' ' +
        df['key_achievements'].fillna('')
    )
    df['res_clean'] = df['resume_text'].apply(preprocess)
    jd_clean        = preprocess(jd['jd_text'])

    corpus = df['res_clean'].tolist() + [jd_clean]
    vec    = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    mat    = vec.fit_transform(corpus)
    sims   = cosine_similarity(mat[:-1], mat[-1]).flatten()

    df['tfidf_raw'] = (sims * 100).round(2)
    tmax = df['tfidf_raw'].max()
    df['tfidf_n']   = (df['tfidf_raw'] / tmax * 100).round(2) if tmax > 0 else 0.0

    # ── 5. HYBRID SCORE ───────────────────────────────────────────────────────
    df['hybrid_sc'] = (
        df['rule_sc'] * 0.60 + df['tfidf_n'] * 0.40
    ).clip(0, 100).round(2)

    df['fit_label'] = df['hybrid_sc'].apply(
        lambda s: 'High Fit' if s >= 65 else 'Medium Fit' if s >= 40 else 'Low Fit'
    )

    return df.sort_values('hybrid_sc', ascending=False).reset_index(drop=True)

# ─── ML MODEL TRAINING ───────────────────────────────────────────────────────
@st.cache_data
def train_models(df_scored):
    """Train 3 models, return comparison results."""
    features = ['mand_pct','opt_pct','exp_sc','dom_pct','edu_tot','ats_sc','tfidf_n']
    X = df_scored[features].fillna(0)
    y = df_scored['fit_label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=le.classes_, output_dict=True)
        results[name] = {
            'model': clf, 'acc': acc, 'report': report,
            'y_test': y_test, 'y_pred': y_pred,
            'classes': le.classes_,
        }

    best_name = max(results, key=lambda k: results[k]['acc'])
    rf_model  = results['Random Forest']['model']
    feat_imp  = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

    return results, best_name, feat_imp, features, le

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="eyebrow">TalentBridge AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="display" style="font-size:20px;margin-bottom:20px;">Resume Screener</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload parsed_resumes.csv", type=['csv'],
                                label_visibility='collapsed')
    st.markdown('<div class="eyebrow">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Target Role</div>', unsafe_allow_html=True)
    role = st.selectbox("", list(JOB_DESCRIPTIONS.keys()), label_visibility='collapsed')

    jd_info = JOB_DESCRIPTIONS[role]
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Mandatory Skills</div>', unsafe_allow_html=True)
    for s in jd_info['mandatory']:
        st.markdown(f'<div style="font-size:13px;color:{INK_MUTED};padding:2px 0;">— {s}</div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{INK_SUB};margin-top:8px;">Min exp: {jd_info["min_exp"]}yr · Domain: {jd_info["domain"]}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Show Top N Candidates</div>', unsafe_allow_html=True)
    top_n = st.slider("", 5, 50, 20, label_visibility='collapsed')

    st.markdown('<div class="eyebrow" style="margin-top:16px;">Fit Filter</div>', unsafe_allow_html=True)
    fit_filter = st.multiselect("", ['High Fit','Medium Fit','Low Fit'],
                                default=['High Fit','Medium Fit','Low Fit'],
                                label_visibility='collapsed')

# ─── MAIN ────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown('<div class="eyebrow" style="color:#5e6ad2;">TalentBridge Solutions · AI Screening System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="display" style="font-size:42px;max-width:600px;margin-bottom:16px;">Resume Screening<br>&amp; Role Matching</div>',
                unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:16px;color:{INK_MUTED};max-width:480px;line-height:1.65;margin-bottom:40px;">'
                f'Upload <code>parsed_resumes.csv</code> in the sidebar to start screening. '
                f'Select a target role and results appear instantly.</p>', unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [(c1,"5,000","Resumes Supported"),
                          (c2,"3","ML Models Trained"),
                          (c3,"16","ATS Flags Analysed"),
                          (c4,"5","Target Roles Available")]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-num">{val}</div>'
                     f'<div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">Pipeline</div>', unsafe_allow_html=True)
    st.code(
        "CSV Upload  →  Typo Fix (86 corrected)  →  Feature Engineering (15+ signals)\n"
        "→  TF-IDF NLP Similarity  →  Hybrid Score (60% Rule + 40% NLP)\n"
        "→  ML Classification (RF/GB/LR)  →  Ranked List + Model Metrics",
        language=None
    )

else:
    df_raw = pd.read_csv(uploaded)

    with st.spinner(f"Screening {len(df_raw):,} candidates for **{role}**…"):
        ranked = run_pipeline(df_raw, role)

    high   = ranked[ranked['fit_label'] == 'High Fit']
    medium = ranked[ranked['fit_label'] == 'Medium Fit']
    low    = ranked[ranked['fit_label'] == 'Low Fit']
    reduction = round(len(low) / len(ranked) * 100, 1)

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.markdown('<div class="eyebrow">Screening Complete</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="display">{role}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:14px;color:{INK_SUB};">'
                f'{len(ranked):,} candidates · Min exp: {JOB_DESCRIPTIONS[role]["min_exp"]}yr '
                f'· Domain: {JOB_DESCRIPTIONS[role]["domain"]} · '
                f'Mandatory skills: {", ".join(JOB_DESCRIPTIONS[role]["mandatory"])}</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── KPI ROW ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, lbl, color in [
        (k1, f"{len(ranked):,}", "Total Screened",     INK),
        (k2, str(len(high)),     "High Fit",            SUCCESS),
        (k3, str(len(medium)),   "Medium Fit",          WARNING),
        (k4, str(len(low)),      "Low Fit",             DANGER),
        (k5, f"{reduction}%",    "Workload Saved",      ACCENT),
    ]:
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-num" style="color:{color};">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── TABS ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "ML Models", "Rankings", "Export"])

    # ════ TAB 1 — ANALYTICS ═════════════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="eyebrow">Score Distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(ranked['hybrid_sc'], bins=28, color=ACCENT, alpha=0.85,
                    edgecolor=CANVAS, linewidth=0.4)
            ax.axvline(65, color=SUCCESS, lw=1.5, linestyle='--', alpha=0.85)
            ax.axvline(40, color=WARNING, lw=1.5, linestyle='--', alpha=0.85)
            y_top = ax.get_ylim()[1]
            ax.text(66, y_top*0.88, 'High', color=SUCCESS, fontsize=9, fontweight='600')
            ax.text(41, y_top*0.88, 'Med',  color=WARNING, fontsize=9, fontweight='600')
            ax.set_xlabel('Hybrid Fit Score', fontsize=11, labelpad=8)
            ax.set_ylabel('Candidates', fontsize=11, labelpad=8)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout(pad=1.2)
            st.pyplot(make_chart(fig), use_container_width=True)

        with col_r:
            st.markdown('<div class="eyebrow">Fit Category Breakdown</div>', unsafe_allow_html=True)
            lc     = ranked['fit_label'].value_counts()
            colors = [fit_color(l) for l in lc.index]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            wedges, _, autotexts = ax.pie(
                lc.values, labels=None, autopct='%1.0f%%',
                colors=colors, startangle=90,
                wedgeprops={'linewidth': 2, 'edgecolor': CANVAS},
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontsize(12); at.set_fontweight('600'); at.set_color(INK)
            ax.legend(lc.index, loc='lower center', bbox_to_anchor=(0.5, -0.10),
                      ncol=3, frameon=False, labelcolor=INK_MUTED, fontsize=10)
            plt.tight_layout(pad=0.5)
            st.pyplot(make_chart(fig), use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.markdown('<div class="eyebrow" style="margin-top:8px;">Domain Distribution</div>',
                        unsafe_allow_html=True)
            dom = df_raw['primary_domain'].value_counts().head(6)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(dom.index[::-1], dom.values[::-1],
                    color=mono_shades(len(dom)), height=0.55)
            ax.set_xlabel('Candidates', fontsize=11, labelpad=8)
            ax.grid(axis='x', alpha=0.3)
            for s in ['top','right','left']: ax.spines[s].set_visible(False)
            plt.tight_layout(pad=1.2)
            st.pyplot(make_chart(fig), use_container_width=True)

        with col_r2:
            st.markdown('<div class="eyebrow" style="margin-top:8px;">Experience Distribution</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.hist(df_raw['years_experience'], bins=18, color=ACCENT, alpha=0.75,
                    edgecolor=CANVAS, linewidth=0.4)
            mean_e = df_raw['years_experience'].mean()
            ax.axvline(mean_e, color=INK_SUB, lw=1.5, linestyle='--')
            ax.text(mean_e+0.3, ax.get_ylim()[1]*0.88,
                    f'Avg {mean_e:.1f}yr', color=INK_SUB, fontsize=9, fontweight='500')
            ax.set_xlabel('Years Experience', fontsize=11, labelpad=8)
            ax.set_ylabel('Candidates', fontsize=11, labelpad=8)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout(pad=1.2)
            st.pyplot(make_chart(fig), use_container_width=True)

        # Skill match distribution
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Mandatory Skill Match Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 3.0))
        colors_map = ranked['fit_label'].map({'High Fit': SUCCESS, 'Medium Fit': WARNING, 'Low Fit': DANGER})
        ax.scatter(ranked.index[:200], ranked['mand_pct'][:200],
                   c=colors_map[:200], alpha=0.6, s=12)
        ax.set_xlabel('Candidate Rank', fontsize=11, labelpad=8)
        ax.set_ylabel('Mandatory Skill Match %', fontsize=11, labelpad=8)
        ax.grid(alpha=0.2)
        patches = [mpatches.Patch(color=SUCCESS, label='High Fit'),
                   mpatches.Patch(color=WARNING, label='Medium Fit'),
                   mpatches.Patch(color=DANGER,  label='Low Fit')]
        ax.legend(handles=patches, frameon=False, labelcolor=INK_MUTED, fontsize=9)
        plt.tight_layout(pad=1.2)
        st.pyplot(make_chart(fig), use_container_width=True)

    # ════ TAB 2 — ML MODELS ═════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="eyebrow">Machine Learning — Model Training &amp; Comparison</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:13px;color:{INK_SUB};margin-bottom:20px;">'
                    f'3 classifiers trained on hybrid scores. 80/20 train-test split. '
                    f'Labels: High Fit / Medium Fit / Low Fit.</p>', unsafe_allow_html=True)

        with st.spinner("Training models…"):
            results, best_name, feat_imp, features, le = train_models(ranked)

        # Model accuracy comparison
        col_m1, col_m2, col_m3 = st.columns(3)
        for col, mname in zip([col_m1, col_m2, col_m3], results.keys()):
            acc = results[mname]['acc']
            is_best = (mname == best_name)
            border = f"border:2px solid {ACCENT};" if is_best else f"border:1px solid {HAIRLINE};"
            badge  = f'<div style="font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:{ACCENT};margin-bottom:8px;">★ Best Model</div>' if is_best else ''
            col.markdown(
                f'<div style="background:{SURFACE1};{border}border-radius:12px;padding:20px;text-align:center;">'
                f'{badge}'
                f'<div class="eyebrow" style="justify-content:center;margin-bottom:12px;">{mname}</div>'
                f'<div style="font-size:40px;font-weight:600;letter-spacing:-0.045em;color:{"#5e6ad2" if is_best else INK_MUTED};">'
                f'{acc*100:.1f}%</div>'
                f'<div class="kpi-label">Accuracy</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            # Confusion Matrix
            st.markdown(f'<div class="eyebrow">{best_name} — Confusion Matrix</div>',
                        unsafe_allow_html=True)
            best_res = results[best_name]
            cm = confusion_matrix(best_res['y_test'], best_res['y_pred'])
            fig, ax = plt.subplots(figsize=(5.5, 4))
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(best_res['classes'])))
            ax.set_yticks(range(len(best_res['classes'])))
            ax.set_xticklabels(best_res['classes'], rotation=20, ha='right', fontsize=9)
            ax.set_yticklabels(best_res['classes'], fontsize=9)
            for i in range(len(best_res['classes'])):
                for j in range(len(best_res['classes'])):
                    ax.text(j, i, str(cm[i, j]),
                            ha='center', va='center',
                            color=INK if cm[i,j] < cm.max()*0.6 else CANVAS,
                            fontsize=11, fontweight='600')
            ax.set_xlabel('Predicted', fontsize=10, labelpad=8)
            ax.set_ylabel('Actual', fontsize=10, labelpad=8)
            plt.colorbar(im, ax=ax, fraction=0.04)
            plt.tight_layout(pad=1.2)
            st.pyplot(make_chart(fig), use_container_width=True)

        with col_right:
            # Feature Importance
            st.markdown('<div class="eyebrow">Random Forest — Feature Importance</div>',
                        unsafe_allow_html=True)
            feat_labels = {
                'mand_pct':'Mandatory Skill Match', 'opt_pct':'Optional Skill Match',
                'exp_sc':'Experience Score', 'dom_pct':'Domain Match',
                'edu_tot':'Education + Tier', 'ats_sc':'ATS Composite',
                'tfidf_n':'NLP TF-IDF Similarity'
            }
            imp_display = feat_imp.copy()
            imp_display.index = [feat_labels.get(f, f) for f in imp_display.index]
            fig, ax = plt.subplots(figsize=(5.5, 4))
            colors = [ACCENT if i == 0 else f'rgba(94,106,210,{0.85 - i*0.1})'
                      for i in range(len(imp_display))]
            bars = ax.barh(imp_display.index[::-1], imp_display.values[::-1],
                           color=mono_shades(len(imp_display)), height=0.65)
            ax.set_xlabel('Importance Score', fontsize=10, labelpad=8)
            ax.grid(axis='x', alpha=0.3)
            for s in ['top','right','left']: ax.spines[s].set_visible(False)
            plt.tight_layout(pad=1.2)
            st.pyplot(make_chart(fig), use_container_width=True)

        # Classification report
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="eyebrow">{best_name} — Full Classification Report</div>',
                    unsafe_allow_html=True)
        report_dict = results[best_name]['report']
        report_df = pd.DataFrame({
            cls: {
                'Precision': f"{report_dict[cls]['precision']:.3f}",
                'Recall':    f"{report_dict[cls]['recall']:.3f}",
                'F1-Score':  f"{report_dict[cls]['f1-score']:.3f}",
                'Support':   f"{int(report_dict[cls]['support'])}",
            }
            for cls in best_res['classes']
        }).T
        st.dataframe(report_df, use_container_width=True)

    # ════ TAB 3 — RANKINGS ══════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="eyebrow" style="margin-bottom:16px;">Candidate Rankings</div>',
                    unsafe_allow_html=True)
        filtered = ranked[ranked['fit_label'].isin(fit_filter)].head(top_n)

        for i, (_, row) in enumerate(filtered.iterrows(), 1):
            label  = row['fit_label']
            color  = fit_color(label)
            exp_ok = row['years_experience'] >= JOB_DESCRIPTIONS[role]['min_exp']
            dom_ok = row['dom_match'] == 1

            st.markdown(
                f'<div class="cand-row">'
                f'<div style="display:grid;grid-template-columns:32px 1fr auto;gap:16px;align-items:center;">'
                f'<div style="font-size:12px;font-weight:600;color:{INK_SUB};">#{i}</div>'
                f'<div>'
                f'  <div class="cand-name">{row["candidate_name"]}</div>'
                f'  <div class="cand-meta">'
                f'    {row["years_experience"]}yr exp {"✓" if exp_ok else "·"} &nbsp;·&nbsp; '
                f'    Skills {row["mand_pct"]:.0f}% &nbsp;·&nbsp; '
                f'    Domain {"match" if dom_ok else "mismatch"} &nbsp;·&nbsp; '
                f'    NLP {row["tfidf_n"]:.0f}%'
                f'  </div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'  <div style="font-size:22px;font-weight:600;letter-spacing:-0.04em;color:{color};">'
                f'    {row["hybrid_sc"]:.1f}</div>'
                f'  {fit_pill(label)}'
                f'</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    # ════ TAB 4 — EXPORT ════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="eyebrow" style="margin-bottom:16px;">Export Results</div>',
                    unsafe_allow_html=True)
        output = ranked[[
            'candidate_name', 'primary_role', 'primary_domain',
            'years_experience', 'highest_education',
            'mand_pct', 'opt_pct', 'tfidf_n', 'rule_sc', 'hybrid_sc', 'fit_label'
        ]].rename(columns={
            'mand_pct':  'mandatory_skill_%',
            'opt_pct':   'optional_skill_%',
            'tfidf_n':   'nlp_similarity_%',
            'rule_sc':   'rule_based_score',
            'hybrid_sc': 'final_fit_score',
            'fit_label': 'recommendation',
        })
        output.insert(0, 'rank', range(1, len(output)+1))

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇ Download Full Rankings",
                output.to_csv(index=False),
                "final_candidate_rankings.csv", "text/csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "⬇ Download High Fit Only",
                output[output['recommendation']=='High Fit'].to_csv(index=False),
                "high_fit_shortlist.csv", "text/csv",
                use_container_width=True
            )

        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="eyebrow">Preview — Top 10</div>', unsafe_allow_html=True)
        st.dataframe(output.head(10), use_container_width=True, height=320)

        # Workload stats
        st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="eyebrow">Screening Impact Summary</div>', unsafe_allow_html=True)
        st.info(
            f"**Total Screened:** {len(ranked):,}  |  "
            f"**High Fit (auto-shortlist):** {len(high)} ({len(high)/len(ranked)*100:.1f}%)  |  "
            f"**Medium Fit (human review):** {len(medium)} ({len(medium)/len(ranked)*100:.1f}%)  |  "
            f"**Low Fit (auto-filtered):** {len(low)} ({len(low)/len(ranked)*100:.1f}%)  |  "
            f"**Workload Saved:** {reduction}%"
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="height:1px;background:{HAIRLINE};margin:32px 0 16px;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:12px;">
  <div class="eyebrow">TalentBridge Solutions · AI Resume Screener · Hybrid NLP + ML</div>
  <div style="font-size:11px;color:{INK_SUB};">Kuldeep Panwar · 24CRD02054 · Career247</div>
</div>
""", unsafe_allow_html=True)
