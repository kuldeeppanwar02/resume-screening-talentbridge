# Resume Screening & Role Matching Using NLP
## TalentBridge Solutions Pvt. Ltd.

**Submitted By:** Kuldeep Panwar  
**Registration ID:** 24CRD02054  
**Program:** Data Science with GenAI — Capstone Project  
**Platform:** Career247 (An AdmEdu Company)

---

## 📌 Project Overview

An end-to-end NLP-powered resume screening and candidate–role matching system for TalentBridge Solutions. The system processes 5,000 structured resume records and ranks candidates against job descriptions using a Hybrid Rule-Based + TF-IDF + Machine Learning pipeline.

---

## 🎯 Problem Statement

TalentBridge processes thousands of resumes weekly but faces:
- High volume overwhelming recruiters
- Keyword-based filters missing varied skill expressions
- Inconsistent shortlisting across locations
- Excessive manual review time

---

## 🔧 Solution Approach — Hybrid Pipeline

```text
Rule-Based Scoring (60%) + TF-IDF NLP Similarity (40%) = Hybrid Fit Score
                    +
        ML Classification (Random Forest / Logistic Regression)
```

---

## 📁 File Structure

```text
24CRD02054_KuldeepPanwar_ResumeScreeningAndRoleMatching/
│
├── 📓 Resume_Screening_LOCAL_JUPYTER.ipynb     ← Main Jupyter Notebook
├── 📊 Report_Deck.html                         ← Analytical report (Print to PDF)
├── 🌐 streamlit_app.py                         ← Interactive UI (Live Dashboard)
├── 📄 README.md                                ← This file
├── 📋 requirements.txt                         ← Python dependencies
│
└── data/
    └── parsed_resumes.csv                      ← Input dataset (5000 rows)
```

---

## ⚙️ How to Run

### Option 1: Live Streamlit Dashboard (Recommended)
You can view the fully functioning, deployed web application here:
🔗 **[Live Streamlit App](https://resume-screening-and-role-matching-using-nlp.streamlit.app/)**

### Option 2: Run Streamlit Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Option 3: Run Jupyter Notebook Locally
```bash
pip install -r requirements.txt
jupyter notebook Resume_Screening_LOCAL_JUPYTER.ipynb
```

---

## 📊 Key Results & Metrics

| Metric | Value |
|--------|-------|
| Dataset Size | 5,000 candidates |
| Best ML Model | Logistic Regression (99.0% Accuracy) |
| Random Forest Accuracy | 99.0% |
| High Fit Candidates (IT) | ~6% |
| Recruiter Workload Reduction | **80.1%** |
| Processing Time | Instant ranking |

---

## 🏗️ Pipeline Architecture

1. **Data Validation & Cleaning:** Handling missing values and fixing typos.
2. **Feature Engineering:** Calculating skill match %, experience penalty/bonus, domain matching, education tier scoring, and ATS flags.
3. **NLP (TF-IDF):** Vectorizing resume text against job descriptions and computing Cosine Similarity.
4. **Hybrid Scoring:** Combining Rule-Based metrics (60%) and NLP scores (40%).
5. **ML Classification:** Training Logistic Regression, Random Forest, and Gradient Boosting on the Hybrid Score data.
6. **Dashboard UI:** Interactive filtering and explainable rankings for recruiters via Streamlit.

---

## 💼 Business Impact

- **Faster Screening:** Automated scoring reduces manual review time drastically.
- **80.1% Workload Saved:** Instantly filters out "Low Fit" candidates.
- **Consistency:** Same objective criteria applied to all candidates.
- **Explainability:** Every score provides transparent matching metrics (Skills, Experience, NLP).

---

## 📞 Contact

**Kuldeep Panwar** | Reg: 24CRD02054  
Career247 — Data Science with GenAI Capstone
