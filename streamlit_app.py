import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re, warnings, os
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

# ── TOKENS ────────────────────────────────────────────────────────────────────
CANVAS   = "#010102"
SURF1    = "#0f1011"
SURF2    = "#141516"
HAIR     = "#23252a"
HAIR2    = "#34343a"
INK      = "#f7f8f8"
INK_M    = "#d0d6e0"
INK_S    = "#8a8f98"
ACCENT   = "#5e6ad2"
OK       = "#10b981"
WARN     = "#f59e0b"
DANGER   = "#ef4444"

plt.rcParams.update({
    'figure.facecolor': CANVAS, 'axes.facecolor': SURF1,
    'axes.edgecolor': HAIR2, 'axes.labelcolor': INK_S,
    'xtick.color': INK_S, 'ytick.color': INK_S,
    'text.color': INK_M, 'grid.color': HAIR,
    'grid.linewidth': 0.5, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

st.set_page_config(
    page_title="TalentBridge — Resume Screening",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="collapsed"          # sidebar collapsed by default
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',system-ui,sans-serif!important;}}
.stApp{{background:{CANVAS};color:{INK};}}
div[data-testid="stSidebar"]{{background:{SURF1};border-right:1px solid {HAIR};}}
div[data-testid="stSidebar"] *{{color:{INK_M}!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding:1.5rem 2.5rem 2rem;max-width:1380px;}}

/* Eyebrow */
.ey{{font-size:11px;font-weight:600;letter-spacing:.10em;text-transform:uppercase;
    color:{INK_S};margin-bottom:8px;display:flex;align-items:center;gap:8px;}}
.ey::before{{content:'';display:inline-block;width:14px;height:1px;
    background:{ACCENT};flex-shrink:0;}}
.ey.ac{{color:{ACCENT};}}

/* Headings */
.h1{{font-size:28px;font-weight:600;letter-spacing:-0.03em;color:{INK};margin-bottom:4px;}}
.h2{{font-size:18px;font-weight:600;letter-spacing:-0.02em;color:{INK};}}

/* Domain group header */
.domain-label{{
    font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
    color:{INK_S};padding:6px 12px 4px;border-bottom:1px solid {HAIR};
    display:block;margin-bottom:4px;
}}

/* Role pill button */
.role-btn{{
    display:inline-block;width:100%;text-align:left;
    padding:10px 14px;border-radius:8px;margin-bottom:4px;cursor:pointer;
    font-size:13px;font-weight:500;letter-spacing:-0.01em;
    border:1px solid {HAIR};background:{SURF1};color:{INK_M};
    transition:all 120ms ease;
}}
.role-btn:hover{{border-color:{HAIR2};color:{INK};}}
.role-btn.active{{
    background:rgba(94,106,210,.14);border-color:rgba(94,106,210,.45);
    color:{ACCENT};font-weight:600;
}}
.role-domain{{font-size:10px;color:{INK_S};display:block;margin-top:2px;}}

/* KPI card */
.kpi{{background:{SURF1};border:1px solid {HAIR};border-radius:12px;padding:18px 20px;}}
.kpi-n{{font-size:36px;font-weight:600;letter-spacing:-0.045em;line-height:1;}}
.kpi-l{{font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
    color:{INK_S};margin-top:8px;margin-bottom:8px;}}
.kpi-bar-track{{height:3px;background:{SURF2};border-radius:99px;overflow:hidden;margin-top:6px;}}
.kpi-bar-fill{{height:100%;border-radius:99px;}}

/* Divider */
.div{{width:100%;height:1px;background:{HAIR};margin:20px 0;}}

/* JD pill */
.jd-pill{{display:inline-flex;align-items:center;gap:5px;
    font-size:12px;font-weight:500;padding:4px 12px;border-radius:9999px;
    border:1px solid {HAIR2};color:{INK_M};background:{SURF1};margin:3px;}}
.jd-pill.must{{background:rgba(94,106,210,.10);border-color:rgba(94,106,210,.30);color:{ACCENT};}}
.jd-pill.nice{{background:{SURF1};border-color:{HAIR2};color:{INK_S};}}

/* Fit pills */
.ph{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;
    font-weight:600;letter-spacing:.04em;text-transform:uppercase;
    background:rgba(16,185,129,.12);color:{OK};border:1px solid rgba(16,185,129,.25);}}
.pm{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;
    font-weight:600;letter-spacing:.04em;text-transform:uppercase;
    background:rgba(245,158,11,.12);color:{WARN};border:1px solid rgba(245,158,11,.25);}}
.pl{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;
    font-weight:600;letter-spacing:.04em;text-transform:uppercase;
    background:rgba(239,68,68,.12);color:{DANGER};border:1px solid rgba(239,68,68,.25);}}

/* Candidate row */
.crow{{background:{SURF1};border:1px solid {HAIR};border-radius:10px;
    padding:14px 18px;margin-bottom:7px;}}
.crow:hover{{border-color:{HAIR2};}}
.cn{{font-size:14px;font-weight:600;color:{INK};letter-spacing:-0.01em;}}
.cm{{font-size:12px;color:{INK_S};margin-top:3px;}}

/* Tabs override */
button[data-baseweb="tab"]{{font-size:13px!important;font-weight:600!important;
    letter-spacing:-0.01em!important;}}

/* Info banner */
.info{{background:rgba(94,106,210,.08);border:1px solid rgba(94,106,210,.22);
    border-radius:10px;padding:12px 18px;font-size:13px;color:{INK_M};
    line-height:1.6;margin-bottom:16px;}}
</style>
""", unsafe_allow_html=True)

# ── NLP SETUP ─────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
TYPOS = {'Ptyhon':'Python','Pythno':'Python','Pythoon':'Python','Pyhton':'Python',
         'Javaa':'Java','JAva':'Java'}
SKILL_MAP = {
    'python':'Python','ptyhon':'Python','pythno':'Python',
    'sql':'SQL','mysql':'SQL','postgresql':'SQL',
    'ml':'Machine Learning','machine learning':'Machine Learning',
    'aws':'AWS','amazon web services':'AWS','k8s':'Kubernetes',
    'kubernetes':'Kubernetes','docker':'Docker','git':'Git','linux':'Linux',
    'java':'Java','javascript':'JavaScript','js':'JavaScript',
    'tensorflow':'TensorFlow','tf':'TensorFlow',
    'sklearn':'Scikit-Learn','scikit-learn':'Scikit-Learn',
    'nlp':'NLP','pandas':'Pandas','numpy':'NumPy',
    'power bi':'Power BI','powerbi':'Power BI',
    'devops':'DevOps','ci/cd':'CI/CD','ansible':'Ansible',
}

# ── JOB DESCRIPTIONS — grouped by domain ─────────────────────────────────────
DOMAINS = {
    '💻 IT': {
        'Senior Software Engineer': {
            'mandatory':['Python','Java','Git','SQL'],
            'optional' :['Docker','Kubernetes','AWS','Linux'],
            'min_exp':5,'domain':'IT',
            'jd_text':'python java git sql docker kubernetes aws linux backend api rest senior software engineering'
        },
        'Cloud Engineer': {
            'mandatory':['AWS','Docker','Kubernetes','Linux'],
            'optional' :['Python','Git','SQL','Ansible'],
            'min_exp':3,'domain':'IT',
            'jd_text':'aws docker kubernetes linux cloud infrastructure devops automation deployment terraform ansible'
        },
        'DevOps Engineer': {
            'mandatory':['Docker','Linux','Git','Kubernetes'],
            'optional' :['AWS','Python','SQL','Ansible'],
            'min_exp':3,'domain':'IT',
            'jd_text':'docker linux git kubernetes devops ci cd pipeline automation aws python deployment monitoring bash'
        },
    },
    '🧠 Data Science': {
        'Data Scientist': {
            'mandatory':['Python','Machine Learning','SQL','Pandas'],
            'optional' :['TensorFlow','NLP','Power BI','NumPy'],
            'min_exp':2,'domain':'Data Science',
            'jd_text':'python machine learning sql pandas tensorflow nlp data analysis feature engineering model training scikit-learn'
        },
        'ML Engineer': {
            'mandatory':['Python','Machine Learning','TensorFlow','SQL'],
            'optional' :['NLP','Docker','AWS','Pandas'],
            'min_exp':2,'domain':'Data Science',
            'jd_text':'python tensorflow machine learning sql nlp docker aws pandas model deployment mlops pipeline deep learning'
        },
    },
}

# Flat lookup
JD_FLAT = {role: info for dom in DOMAINS.values() for role, info in dom.items()}
ROLE_DOMAIN_MAP = {role: dom for dom, roles in DOMAINS.items() for role in roles}

ACTUAL_FLAGS = [
    'management_experience_flag','people_management_flag',
    'project_management_experience_flag','agile_scrum_experience_flag',
    'client_facing_experience_flag','delivery_lead_experience_flag',
    'cloud_experience_flag','ml_experience_flag','compliance_experience_flag',
    'enterprise_systems_experience_flag','offshore_onsite_model_experience_flag',
    'multi_vendor_coordination_flag','process_compliance_experience_flag',
    'documentation_heavy_role_flag','mentoring_experience_flag',
    'stakeholder_management_experience_flag',
]
DEFAULT_CSV = 'data/parsed_resumes.csv'

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fix_typos(t):
    if pd.isna(t): return t
    return ', '.join([TYPOS.get(w.strip(), w.strip()) for w in str(t).split(',')])

def norm_skills(t):
    if pd.isna(t): return []
    return [SKILL_MAP.get(s.lower().strip(), s.strip()) for s in str(t).split(',') if s.strip()]

def skill_pct(skills, req):
    if not req: return 0.0
    sl = [s.lower() for s in skills]
    return round(sum(1 for r in req if r.lower() in sl) / len(req) * 100, 1)

def preprocess(text):
    if pd.isna(text) or not str(text).strip(): return ''
    t = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    try: toks = nltk.word_tokenize(t)
    except: toks = t.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in toks
                     if w not in stop_words and len(w) > 2])

def fit_col(l):
    return {' High Fit':OK,'High Fit':OK,'Medium Fit':WARN,'Low Fit':DANGER}.get(l,INK_S)

def fit_pill_html(l):
    c = {'High Fit':'ph','Medium Fit':'pm','Low Fit':'pl'}.get(l,'')
    return f'<span class="{c}">{l}</span>'

def mkchart(f): f.patch.set_facecolor(CANVAS); return f

def mono(n):
    r,g,b = int(ACCENT[1:3],16),int(ACCENT[3:5],16),int(ACCENT[5:7],16)
    return [(r/255,g/255,b/255,a) for a in np.linspace(0.9,0.3,n)]

# ── PIPELINE ──────────────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline(df_raw, role):
    df  = df_raw.copy()
    jd  = JD_FLAT[role]
    df['highest_education'] = df['highest_education'].fillna('Not Specified')
    df['education_field']   = df['education_field'].fillna('Not Specified')
    df = df.drop_duplicates().reset_index(drop=True)
    df['technical_skills_raw'] = df['technical_skills_raw'].apply(fix_typos)
    df['tech_sk']  = df['technical_skills_raw'].apply(norm_skills)
    df['tool_sk']  = df['tools_platforms_raw'].apply(norm_skills)
    df['all_sk']   = df['tech_sk'] + df['tool_sk']
    df['mand_pct'] = df['all_sk'].apply(lambda s: skill_pct(s, jd['mandatory']))
    df['opt_pct']  = df['all_sk'].apply(lambda s: skill_pct(s, jd['optional']))
    df['skill_sc'] = (df['mand_pct']*.70 + df['opt_pct']*.30).round(1)
    df['exp_sc']   = df['years_experience'].apply(lambda y: min(100.0, round(y/max(jd['min_exp'],1)*100,1)))
    df['dom_match']= (df['primary_domain']==jd['domain']).astype(int)
    df['dom_pct']  = df['dom_match']*100
    edu_m  = {'Masters':100,'MBA':90,'LLM':80,'Bachelors':70,'Not Specified':50}
    tier_m = {'Tier-1':10,'Tier-2':5,'Tier-3':0}
    df['edu_sc']   = df['highest_education'].map(edu_m).fillna(60)
    df['tier_bon'] = df['institution_tier'].map(tier_m).fillna(0)
    df['edu_tot']  = (df['edu_sc']+df['tier_bon']).clip(upper=100)
    fls = [c for c in ACTUAL_FLAGS if c in df.columns]
    df['ats_sc']   = (df[fls].sum(axis=1)/len(fls)*100).round(1) if fls else 50.0
    df['rule_sc']  = (df['skill_sc']*.40+df['exp_sc']*.25+
                      df['dom_pct']*.15+df['edu_tot']*.10+
                      df['ats_sc']*.10).clip(0,100).round(2)
    df['res_text'] = (df['technical_skills_raw'].fillna('')+' '+
                      df['tools_platforms_raw'].fillna('')+' '+
                      df['experience_summary'].fillna('')+' '+
                      df['project_summary'].fillna('')+' '+
                      df['key_achievements'].fillna(''))
    df['res_clean']= df['res_text'].apply(preprocess)
    jd_clean       = preprocess(jd['jd_text'])
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2)
    mat = vec.fit_transform(df['res_clean'].tolist()+[jd_clean])
    sims= cosine_similarity(mat[:-1], mat[-1]).flatten()
    df['tfidf_raw']= (sims*100).round(2)
    tmax = df['tfidf_raw'].max()
    df['tfidf_n']  = (df['tfidf_raw']/tmax*100).round(2) if tmax>0 else 0.0
    df['hybrid_sc']= (df['rule_sc']*.60+df['tfidf_n']*.40).clip(0,100).round(2)
    df['fit_label']= df['hybrid_sc'].apply(
        lambda s: 'High Fit' if s>=65 else 'Medium Fit' if s>=40 else 'Low Fit')
    return df.sort_values('hybrid_sc',ascending=False).reset_index(drop=True)

@st.cache_data
def train_models(df_sc):
    feats = ['mand_pct','opt_pct','exp_sc','dom_pct','edu_tot','ats_sc','tfidf_n']
    X = df_sc[feats].fillna(0)
    y = df_sc['fit_label']
    le = LabelEncoder(); ye = le.fit_transform(y)
    Xtr,Xte,ytr,yte = train_test_split(X,ye,test_size=.20,random_state=42,stratify=ye)
    mdls = {'Logistic Regression':LogisticRegression(max_iter=500,random_state=42),
            'Random Forest':RandomForestClassifier(n_estimators=100,random_state=42),
            'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,random_state=42)}
    res = {}
    for nm,clf in mdls.items():
        clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)
        res[nm]={'model':clf,'acc':acc,'y_test':yte,'y_pred':yp,'classes':le.classes_,
                 'report':classification_report(yte,yp,target_names=le.classes_,output_dict=True)}
    best = max(res, key=lambda k: res[k]['acc'])
    rf   = res['Random Forest']['model']
    fimp = pd.Series(rf.feature_importances_,index=feats).sort_values(ascending=False)
    return res, best, fimp, feats, le

# ── STATE ─────────────────────────────────────────────────────────────────────
if 'role' not in st.session_state:
    st.session_state.role = 'Senior Software Engineer'

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
if os.path.exists(DEFAULT_CSV):
    df_raw     = pd.read_csv(DEFAULT_CSV)
    data_ready = True
else:
    data_ready = False

# ── TOP NAV BAR ───────────────────────────────────────────────────────────────
st.markdown('<div class="ey ac">TalentBridge Solutions · AI Screening System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="h1">Resume Screening & Role Matching</div>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:13px;color:{INK_S};margin-bottom:20px;">'
            f'Hybrid NLP + ML pipeline · 5,000 candidates · 3 ML models · Select a role below to screen</p>',
            unsafe_allow_html=True)
st.markdown('<div class="div" style="margin:16px 0;"></div>', unsafe_allow_html=True)

# ── ROLE SELECTOR (MAIN PAGE — DOMAIN GROUPED) ────────────────────────────────
st.markdown('<div class="ey">Select Target Role</div>', unsafe_allow_html=True)
dom_cols = st.columns(len(DOMAINS), gap='medium')

for col, (domain_name, roles) in zip(dom_cols, DOMAINS.items()):
    with col:
        st.markdown(
            f'<div style="background:{SURF1};border:1px solid {HAIR};border-radius:12px;padding:14px 16px;">'
            f'<span class="domain-label">{domain_name}</span>',
            unsafe_allow_html=True
        )
        for role_name in roles:
            is_active = (st.session_state.role == role_name)
            jd_info   = JD_FLAT[role_name]
            btn_style = (f'background:rgba(94,106,210,.14);border:1px solid rgba(94,106,210,.45);'
                         f'color:{ACCENT};' if is_active else
                         f'background:{SURF2};border:1px solid {HAIR};color:{INK_M};')
            if st.button(
                f"{role_name}",
                key=f"role_{role_name}",
                use_container_width=True,
                type='primary' if is_active else 'secondary'
            ):
                st.session_state.role = role_name
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

role    = st.session_state.role
jd_info = JD_FLAT[role]
dom_grp = ROLE_DOMAIN_MAP[role]

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── JD INFO STRIP ────────────────────────────────────────────────────────────
jd_col1, jd_col2 = st.columns([3,1])
with jd_col1:
    st.markdown(f'<div class="h2" style="margin-bottom:10px;">{role}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="margin-bottom:6px;">'
        f'<span style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:{INK_S};">Must-Have Skills</span></div>'
        + ''.join([f'<span class="jd-pill must">✓ {s}</span>' for s in jd_info['mandatory']])
        + '<br><span style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;'
        + f'color:{INK_S};margin-top:8px;display:inline-block;">Good to Have</span><br>'
        + ''.join([f'<span class="jd-pill nice">{s}</span>' for s in jd_info['optional']]),
        unsafe_allow_html=True
    )
with jd_col2:
    st.markdown(
        f'<div class="kpi" style="text-align:center;">'
        f'<div class="kpi-n" style="font-size:28px;color:{ACCENT};">{jd_info["min_exp"]}yr+</div>'
        f'<div class="kpi-l">Min Experience</div>'
        f'<div style="font-size:12px;color:{INK_S};">Domain: {jd_info["domain"]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── NO DATA ───────────────────────────────────────────────────────────────────
if not data_ready:
    uploaded = st.file_uploader("Upload parsed_resumes.csv to begin", type=['csv'])
    if uploaded:
        df_raw     = pd.read_csv(uploaded)
        data_ready = True
    else:
        st.markdown(
            f'<div class="info">⚡ Upload <code>parsed_resumes.csv</code> to start screening candidates.</div>',
            unsafe_allow_html=True
        )
        st.stop()

# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
with st.spinner(f'Screening {len(df_raw):,} candidates for **{role}**…'):
    ranked = run_pipeline(df_raw, role)

high   = ranked[ranked['fit_label']=='High Fit']
medium = ranked[ranked['fit_label']=='Medium Fit']
low    = ranked[ranked['fit_label']=='Low Fit']
total  = len(ranked)
workload_saved = round(len(low)/total*100,1)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5, gap='small')
kpis = [
    (k1, f"{total:,}", "Total Screened",   INK,    total/total*100),
    (k2, str(len(high)),   "High Fit → Auto Shortlist",  OK,   len(high)/total*100),
    (k3, str(len(medium)), "Medium Fit → Human Review",  WARN, len(medium)/total*100),
    (k4, str(len(low)),    "Low Fit → Auto Filtered",    DANGER, len(low)/total*100),
    (k5, f"{workload_saved}%", "Workload Saved",         ACCENT, workload_saved),
]
for col, val, lbl, color, pct in kpis:
    col.markdown(
        f'<div class="kpi">'
        f'  <div class="kpi-n" style="color:{color};">{val}</div>'
        f'  <div class="kpi-l">{lbl}</div>'
        f'  <div class="kpi-bar-track">'
        f'    <div class="kpi-bar-fill" style="width:{min(pct,100):.0f}%;background:{color};opacity:.8;"></div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Analytics",
    "🤖  ML Models & Accuracy",
    "🏆  Candidate Rankings",
    "📥  Export Results"
])

# ════ TAB 1 — ANALYTICS ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="div" style="margin:12px 0 20px;"></div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap='large')

    with a1:
        st.markdown('<div class="ey">Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        n, bins, patches = ax.hist(ranked['hybrid_sc'], bins=28, edgecolor=CANVAS, linewidth=0.4)
        for patch, left in zip(patches, bins[:-1]):
            if left >= 65:   patch.set_facecolor(OK)
            elif left >= 40: patch.set_facecolor(WARN)
            else:            patch.set_facecolor(DANGER)
        ax.axvline(65, color=OK,   lw=1.5, linestyle='--', alpha=0.9, label=f'High Fit ≥65 (n={len(high)})')
        ax.axvline(40, color=WARN, lw=1.5, linestyle='--', alpha=0.9, label=f'Medium Fit ≥40 (n={len(medium)})')
        ax.set_xlabel('Hybrid Fit Score', fontsize=11, labelpad=8)
        ax.set_ylabel('Candidates', fontsize=11, labelpad=8)
        ax.legend(frameon=False, labelcolor=INK_M, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

    with a2:
        st.markdown('<div class="ey">Fit Category — Stacked View</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        cats   = ['Candidates']
        pcts   = [len(high)/total*100, len(medium)/total*100, len(low)/total*100]
        labels = [f'High Fit\n{len(high)} ({pcts[0]:.1f}%)',
                  f'Medium Fit\n{len(medium)} ({pcts[1]:.1f}%)',
                  f'Low Fit\n{len(low)} ({pcts[2]:.1f}%)']
        colors = [OK, WARN, DANGER]
        left   = 0
        for p, lbl, c in zip(pcts, labels, colors):
            bar = ax.barh(cats, [p], left=[left], color=c, height=0.55)
            if p > 5:
                ax.text(left + p/2, 0, f'{p:.0f}%', ha='center', va='center',
                       fontsize=11, fontweight='600', color=CANVAS)
            left += p
        ax.set_xlim(0, 100)
        ax.set_xlabel('Percentage of Candidates (%)', fontsize=11, labelpad=8)
        ax.tick_params(left=False, labelleft=False)
        patches_leg = [mpatches.Patch(color=c,label=l) for c,l in
                       zip(colors,[f'High Fit ({len(high)})',f'Medium Fit ({len(medium)})',f'Low Fit ({len(low)])'])]
        ax.legend(handles=patches_leg, loc='lower center', bbox_to_anchor=(0.5,-0.28),
                  ncol=3, frameon=False, labelcolor=INK_M, fontsize=9)
        ax.grid(axis='x', alpha=0.2)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

    b1, b2 = st.columns(2, gap='large')
    with b1:
        st.markdown('<div class="ey" style="margin-top:16px;">Domain Distribution</div>',
                    unsafe_allow_html=True)
        dom = df_raw['primary_domain'].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        bar_colors = [ACCENT if d==jd_info['domain'] else f'#{int(ACCENT[1:3],16):02x}{int(ACCENT[3:5],16):02x}{int(ACCENT[5:7],16):02x}'
                      for d in dom.index[::-1]]
        bars = ax.barh(dom.index[::-1], dom.values[::-1], color=mono(len(dom)), height=0.55)
        for bar, val in zip(bars, dom.values[::-1]):
            ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                   str(val), va='center', fontsize=9, color=INK_S)
        ax.set_xlabel('Candidates', fontsize=11, labelpad=8)
        ax.grid(axis='x', alpha=0.2)
        for s in ['top','right','left']: ax.spines[s].set_visible(False)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

    with b2:
        st.markdown('<div class="ey" style="margin-top:16px;">Experience Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.hist(df_raw['years_experience'], bins=18, color=ACCENT, alpha=0.75,
                edgecolor=CANVAS, linewidth=0.4)
        me = df_raw['years_experience'].mean()
        ax.axvline(me, color=INK_S, lw=1.5, linestyle='--')
        ax.axvline(jd_info['min_exp'], color=OK, lw=1.5, linestyle=':',
                   label=f'JD Min: {jd_info["min_exp"]}yr')
        ax.text(me+0.2, ax.get_ylim()[1]*.88, f'Avg {me:.1f}yr',
               color=INK_S, fontsize=9, fontweight='500')
        ax.legend(frameon=False, labelcolor=INK_M, fontsize=9)
        ax.set_xlabel('Years Experience', fontsize=11, labelpad=8)
        ax.set_ylabel('Candidates', fontsize=11, labelpad=8)
        ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

# ════ TAB 2 — ML MODELS ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="div" style="margin:12px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="info">3 classifiers trained on hybrid scores · 80/20 train-test split · '
        f'Labels: High Fit / Medium Fit / Low Fit · Test set: {int(total*0.2):,} candidates</div>',
        unsafe_allow_html=True
    )
    with st.spinner('Training 3 ML models…'):
        results, best_name, feat_imp, feats, le = train_models(ranked)

    # Accuracy cards
    mc1,mc2,mc3 = st.columns(3, gap='medium')
    for col, mname in zip([mc1,mc2,mc3], results):
        acc    = results[mname]['acc']
        is_best= (mname==best_name)
        border = f'border:2px solid {ACCENT};' if is_best else f'border:1px solid {HAIR};'
        badge  = f'<div style="font-size:10px;font-weight:700;color:{ACCENT};letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;">★ Best Model</div>' if is_best else ''
        col.markdown(
            f'<div class="kpi" style="{border}text-align:center;">'
            f'{badge}'
            f'<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:{INK_S};margin-bottom:12px;">{mname}</div>'
            f'<div class="kpi-n" style="color:{"#5e6ad2" if is_best else INK_M};font-size:38px;">{acc*100:.1f}%</div>'
            f'<div class="kpi-l">Accuracy</div>'
            f'<div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{acc*100:.0f}%;background:{"#5e6ad2" if is_best else HAIR2};"></div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    ml1, ml2 = st.columns(2, gap='large')

    with ml1:
        st.markdown(f'<div class="ey">{best_name} — Confusion Matrix</div>', unsafe_allow_html=True)
        br = results[best_name]
        cm = confusion_matrix(br['y_test'], br['y_pred'])
        fig, ax = plt.subplots(figsize=(5.5,4))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(br['classes']))); ax.set_yticks(range(len(br['classes'])))
        ax.set_xticklabels(br['classes'], rotation=20, ha='right', fontsize=9)
        ax.set_yticklabels(br['classes'], fontsize=9)
        for i in range(len(br['classes'])):
            for j in range(len(br['classes'])):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                       color=INK if cm[i,j]<cm.max()*.6 else CANVAS,
                       fontsize=12, fontweight='600')
        ax.set_xlabel('Predicted', fontsize=10, labelpad=8)
        ax.set_ylabel('Actual', fontsize=10, labelpad=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

    with ml2:
        st.markdown('<div class="ey">Random Forest — Feature Importance</div>', unsafe_allow_html=True)
        feat_lbl = {
            'mand_pct':'Mandatory Skill Match','opt_pct':'Optional Skill Match',
            'exp_sc':'Experience Score','dom_pct':'Domain Match',
            'edu_tot':'Education + Tier','ats_sc':'ATS Composite (16 flags)',
            'tfidf_n':'NLP TF-IDF Similarity'
        }
        fi2 = feat_imp.copy()
        fi2.index = [feat_lbl.get(f,f) for f in fi2.index]
        fig, ax = plt.subplots(figsize=(5.5,4))
        colors  = [ACCENT if i==0 else HAIR2 for i in range(len(fi2))]
        ax.barh(fi2.index[::-1], fi2.values[::-1], color=mono(len(fi2)), height=0.65)
        for i, (idx, val) in enumerate(zip(fi2.index[::-1], fi2.values[::-1])):
            ax.text(val+0.002, i, f'{val:.3f}', va='center', fontsize=8, color=INK_S)
        ax.set_xlabel('Importance', fontsize=10, labelpad=8)
        ax.grid(axis='x', alpha=0.2)
        for s in ['top','right','left']: ax.spines[s].set_visible(False)
        plt.tight_layout(pad=1.2)
        st.pyplot(mkchart(fig), use_container_width=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ey">{best_name} — Full Classification Report</div>', unsafe_allow_html=True)
    rep = results[best_name]['report']
    rep_df = pd.DataFrame({
        cls: {
            'Precision':f"{rep[cls]['precision']:.3f}",
            'Recall'   :f"{rep[cls]['recall']:.3f}",
            'F1-Score' :f"{rep[cls]['f1-score']:.3f}",
            'Support'  :f"{int(rep[cls]['support'])}",
        } for cls in br['classes']
    }).T
    st.dataframe(rep_df, use_container_width=True)

# ════ TAB 3 — RANKINGS ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="div" style="margin:12px 0 16px;"></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([2,2,1], gap='medium')
    with f1:
        fit_filter = st.multiselect(
            "Filter by Fit Category",
            ['High Fit','Medium Fit','Low Fit'],
            default=['High Fit','Medium Fit'],
            key='fit_filter'
        )
    with f2:
        search = st.text_input("Search candidate name", placeholder="Type name…", key='search')
    with f3:
        top_n = st.slider("Show top", 5, 100, 25, key='topn')

    filtered = ranked[ranked['fit_label'].isin(fit_filter)]
    if search:
        filtered = filtered[filtered['candidate_name'].str.contains(search, case=False, na=False)]
    filtered = filtered.head(top_n)

    st.markdown(
        f'<div style="font-size:12px;color:{INK_S};margin-bottom:12px;">'
        f'Showing {len(filtered)} candidates · Role: {role} · '
        f'Min exp: {jd_info["min_exp"]}yr · Domain: {jd_info["domain"]}</div>',
        unsafe_allow_html=True
    )

    for i, (_, row) in enumerate(filtered.iterrows(), 1):
        lbl    = row['fit_label']
        color  = fit_col(lbl)
        exp_ok = row['years_experience'] >= jd_info['min_exp']
        dom_ok = row['dom_match'] == 1
        st.markdown(
            f'<div class="crow">'
            f'<div style="display:grid;grid-template-columns:36px 1fr auto;gap:16px;align-items:center;">'
            f'  <div style="font-size:13px;font-weight:700;color:{INK_S};text-align:center;">#{i}</div>'
            f'  <div>'
            f'    <div class="cn">{row["candidate_name"]}</div>'
            f'    <div class="cm">'
            f'      {row["years_experience"]}yr exp {"✓" if exp_ok else "✗ (below min)"} &nbsp;·&nbsp; '
            f'      Must-have skills: {row["mand_pct"]:.0f}% &nbsp;·&nbsp; '
            f'      Domain: {"✓ match" if dom_ok else "✗ mismatch"} &nbsp;·&nbsp; '
            f'      NLP score: {row["tfidf_n"]:.0f}%'
            f'    </div>'
            f'  </div>'
            f'  <div style="text-align:right;">'
            f'    <div style="font-size:24px;font-weight:700;letter-spacing:-0.04em;color:{color};">'
            f'      {row["hybrid_sc"]:.1f}</div>'
            f'    {fit_pill_html(lbl)}'
            f'  </div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

# ════ TAB 4 — EXPORT ═════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="div" style="margin:12px 0 20px;"></div>', unsafe_allow_html=True)
    out = ranked[['candidate_name','primary_role','primary_domain','years_experience',
                  'highest_education','mand_pct','opt_pct','tfidf_n','rule_sc','hybrid_sc','fit_label']
               ].rename(columns={
                   'mand_pct':'mandatory_skill_%','opt_pct':'optional_skill_%',
                   'tfidf_n':'nlp_similarity_%','rule_sc':'rule_score',
                   'hybrid_sc':'final_fit_score','fit_label':'recommendation'
               })
    out.insert(0, 'rank', range(1, len(out)+1))

    e1,e2,e3 = st.columns(3, gap='medium')
    with e1:
        st.download_button("⬇ Full Rankings CSV", out.to_csv(index=False),
                          "full_rankings.csv","text/csv",use_container_width=True)
    with e2:
        st.download_button("⬇ High Fit Only",
                          out[out['recommendation']=='High Fit'].to_csv(index=False),
                          "high_fit.csv","text/csv",use_container_width=True)
    with e3:
        st.download_button("⬇ High + Medium Fit",
                          out[out['recommendation'].isin(['High Fit','Medium Fit'])].to_csv(index=False),
                          "shortlist.csv","text/csv",use_container_width=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="info">'
        f'<strong>Screening Impact · {role}</strong><br>'
        f'Total: {total:,} &nbsp;|&nbsp; '
        f'High Fit: {len(high)} ({len(high)/total*100:.1f}%) &nbsp;|&nbsp; '
        f'Medium Fit: {len(medium)} ({len(medium)/total*100:.1f}%) &nbsp;|&nbsp; '
        f'Low Fit: {len(low)} ({len(low)/total*100:.1f}%) &nbsp;|&nbsp; '
        f'Workload Saved: <strong>{workload_saved}%</strong>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="ey" style="margin-bottom:12px;">Preview — Top 10</div>', unsafe_allow_html=True)
    st.dataframe(out.head(10), use_container_width=True, height=320)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="height:1px;background:{HAIR};margin:28px 0 16px;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;">
  <div class="ey">TalentBridge Solutions · Hybrid NLP + ML · 60% Rule + 40% TF-IDF</div>
  <div style="font-size:11px;color:{INK_S};">Kuldeep Panwar · 24CRD02054 · Career247</div>
</div>
""", unsafe_allow_html=True)
