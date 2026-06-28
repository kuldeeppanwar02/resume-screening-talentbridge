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

## 🔧 Solution Approach — Option 3 (Hybrid)

```
Rule-Based Scoring (60%) + TF-IDF NLP Similarity (40%) = Hybrid Fit Score
                    +
        ML Classification (Random Forest — Best Model)
```

---

## 📁 File Structure

```
24CRD02054_KuldeepPanwar_ResumeScreeningAndRoleMatching/
│
├── 📓 Project_4_Resume_Screening_NLP.ipynb     ← Main notebook (run this)
├── 📊 Report_Deck.pdf                          ← 10-slide analytical report
├── 🌐 streamlit_app.py                         ← Interactive UI
├── 🤖 best_model.pkl                           ← Saved Random Forest model
├── 🔧 tfidf_vectorizer.pkl                     ← Saved TF-IDF vectorizer
├── 📄 README.md                                ← This file
├── 📋 requirements.txt                         ← Python dependencies
│
├── data/
│   └── parsed_resumes.csv                      ← Input dataset (5000 rows)
│
└── outputs/
    ├── resumes_cleaned_features.csv            ← Cleaned dataset + features
    ├── final_candidate_rankings.csv            ← Ranked candidates output
    └── plots/                                  ← All EDA & model charts
```

---

## ⚙️ How to Run

### Option 1: Google Colab (Recommended)
1. Upload `Project_4_Resume_Screening_NLP.ipynb` to Google Colab
2. Upload `parsed_resumes.csv` when prompted
3. Run All cells (`Runtime → Run All`)
4. Download outputs

### Option 2: Local Jupyter
```bash
pip install -r requirements.txt
jupyter notebook Project_4_Resume_Screening_NLP.ipynb
```

### Option 3: Streamlit Dashboard
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📦 Dependencies

```
pandas==2.3.3
numpy==2.4.3
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
nltk==3.9.3
xgboost==3.2.0
streamlit==1.55.0
wordcloud==1.9.6
missingno==0.5.2
openpyxl==3.1.5
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 5,000 candidates |
| Features Engineered | 15+ |
| Best ML Model | Random Forest |
| High Fit Candidates | ~X% |
| Recruiter Workload Reduction | ~Y% |
| NLP Vocab Size | 1,000 terms |

---

## 🏗️ Pipeline Architecture

```
Raw CSV (5000 resumes)
    ↓
Data Validation & Cleaning
    ↓
Exploratory Data Analysis (6 visualizations)
    ↓
Feature Engineering (skill match, experience, domain, education, ATS)
    ↓
NLP — TF-IDF Vectorization + Cosine Similarity
    ↓
Hybrid Fit Score = 60% Rule + 40% NLP
    ↓
ML Classification (LR vs RF vs GB — Random Forest wins)
    ↓
Ranked Candidate List + Explainable Recommendations
    ↓
Business Report for TalentBridge Recruiters
```

---

## 💼 Business Impact

- **Faster Screening:** Automated scoring reduces manual review time
- **Consistency:** Same objective criteria applied to all candidates
- **Explainability:** Every score comes with a plain-English explanation
- **Scalability:** System handles thousands of resumes instantly

---

## 📞 Contact

**Kuldeep Panwar** | Reg: 24CRD02054  
Career247 — Data Science with GenAI Capstone  
