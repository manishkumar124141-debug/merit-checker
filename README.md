# ⚖️ The Merit-Checker
### HR Bias Audit Platform — Powered by Fairlearn & scikit-learn

> **"Decisions should be based 100% on knowledge. Everything else is irrelevant."**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-Microsoft-0078D4?logo=microsoft&logoColor=white)](https://fairlearn.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 What Is This?

**The Merit-Checker** is an open-source HR bias detection tool built for HR managers, data scientists, and DEI teams. It audits hiring datasets to detect whether **Protected Attributes** (age, gender, race, religion, nationality) are secretly influencing decisions that should be driven purely by **Legitimate Features** (test scores, experience, qualifications).

In data science terms, this tool acts as an **alarm system** — it fires when your hiring data shows that candidates with identical merit scores are receiving different outcomes based on *who they are*, not *what they know*.

---

## 🚀 Features

- **📂 Universal CSV Upload** — Works with any hiring dataset from Kaggle or your company. No fixed column names required.
- **🤖 Smart Auto-Detection** — Automatically identifies outcome columns, legitimate features, and protected attributes from your column names.
- **⚖️ Merit-Only Model** — Trains a Logistic Regression on *only* legitimate features to establish a fairness baseline.
- **🚨 Bias Alarms** — Fires alarms (🚨 High / ⚠️ Caution / ✅ Fair) when hire-rate gaps exceed your configured threshold.
- **📊 Three Fairness Metrics** per protected attribute:
  - **Demographic Parity Difference (DPD)** — hire-rate gap between groups
  - **Demographic Parity Ratio (DPR)** — ratio of lowest to highest hire rate
  - **Equalized Odds Difference (EOD)** — gap in true positive rates
- **📋 Full Audit Report** — Fairness radar chart, scorecard table, and 7+ actionable recommendations.
- **📥 Export** — Download the bias scorecard and annotated dataset as CSV.
- **🔬 Small Dataset Safe** — Gracefully handles datasets from 5 rows to 120,000+ rows.

---

## 📐 The Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│                    HIRING DECISION                          │
│                                                             │
│  ✅ Should matter:        ❌ Should NOT matter:             │
│  Legitimate Features      Protected Attributes              │
│  ─────────────────        ───────────────────               │
│  • Test scores            • Gender / Sex                    │
│  • Years of experience    • Age                             │
│  • Interview score        • Race / Ethnicity                │
│  • Education level        • Religion                        │
│  • Technical skills       • Nationality / Origin            │
│  • Certifications         • Disability status               │
└─────────────────────────────────────────────────────────────┘
```

The Merit-Checker trains a model on **only** Legitimate Features, then compares its predictions against actual hiring decisions. If outcomes differ systematically across protected groups — **the alarm fires.**

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or newer — [download here](https://python.org)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/merit-checker.git
cd merit-checker
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

Your browser opens automatically at **http://localhost:8501** 🎉

---

## 📦 Dependencies (`requirements.txt`)

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
fairlearn>=0.10.0
plotly>=5.18.0
openpyxl>=3.1.0
```

---

## 📂 Project Structure

```
merit-checker/
├── app.py                  ← Main Streamlit application (all logic lives here)
├── generate_data.py        ← Synthetic HR dataset generator (for demo/testing)
├── hr_hiring_data.csv      ← Pre-generated 500-row demo dataset
├── requirements.txt        ← Python dependencies
└── README.md               ← You are here
```

---

## 📋 How to Use

### Option A — Built-in demo data
1. Run `streamlit run app.py`
2. In the sidebar → select **"Use built-in sample data"** → click **▶ Load**
3. All 4 tabs populate instantly with real bias analysis

### Option B — Your own CSV (any Kaggle dataset)
1. Find a dataset on [Kaggle](https://kaggle.com) — good searches:
   - `HR Analytics dataset`
   - `Hiring bias dataset`
   - `Fair recruitment dataset`
   - `Resume screening dataset`
2. Upload via the sidebar uploader
3. Review auto-detected column roles — adjust if anything looks wrong:

| Role | What to put here |
|------|-----------------|
| 🎯 **Outcome column** | The Hired / Selected / Rejected column |
| ✅ **Legitimate features** | Test scores, experience, education — what *should* matter |
| 🔍 **Protected attributes** | Gender, age, race, religion — what should have *zero* influence |

4. All 4 tabs update automatically — no button needed.

### Reading the Results

| Metric | Plain English | Green Zone |
|--------|---------------|------------|
| **DPD (%)** | Max hire-rate gap between groups | < 5% |
| **DPR** | Lowest ÷ highest group hire rate | > 0.80 |
| **EOD (%)** | Gap in true-positive rates across groups | < 5% |

---

## 🧪 Tested Datasets

| Dataset | Rows | Key Finding |
|---------|------|-------------|
| Built-in synthetic | 500 | 🚨 Race: 15.5% · Age: 11.5% · Gender: 6.9% bias |
| Fair Recruitment (Kaggle) | 121,200 | ⚠️ Race: 12.5% · Age: 10.8% caution |
| AI Resume Screening 2026 (Kaggle) | 5 | ℹ️ Too small for statistics — directional only |
| AI Resume Screening (Kaggle) | 30,000 | ✅ Education within threshold |

---

## 🔬 Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Web app | [Streamlit](https://streamlit.io) | Interactive UI — zero frontend code |
| Bias metrics | [Fairlearn](https://fairlearn.org) by Microsoft | DPD, DPR, EOD calculations |
| ML model | [scikit-learn](https://scikit-learn.org) | Logistic Regression merit model |
| Charts | [Plotly](https://plotly.com) | Interactive visualizations |
| Data | pandas + NumPy | Wrangling and computation |

---

## 🗺️ Roadmap

- [ ] What-If Simulator — flip one attribute and see the hire probability change
- [ ] PDF report export for executives
- [ ] Multi-class outcomes (Hired / Shortlisted / Rejected as separate classes)
- [ ] ATS API integration — Greenhouse, Lever, Workday
- [ ] Scheduled audits with email alerts
- [ ] Intersectional bias detection (e.g. older women vs younger men)
- [ ] Counterfactual fairness — "would this candidate have been hired if only their gender changed?"
- [ ] Bias trend tracking across hiring cycles

---

## 🤝 Contributing

Contributions are welcome!

```bash
# 1. Fork this repo and clone your fork
git clone https://github.com/YOUR_USERNAME/merit-checker.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes, then commit
git commit -m "Add: your feature description"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please test with at least one CSV before submitting a PR.

---

## 📄 License

MIT License — free to use, modify, and distribute for personal or commercial purposes. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- **[Microsoft Fairlearn](https://fairlearn.org)** — the open-source fairness library powering this tool
- **[Streamlit](https://streamlit.io)** — for making Python data apps effortless
- **[Kaggle](https://kaggle.com)** — for open HR datasets used in testing
- **[IBM AIF360](https://aif360.mybluemix.net)** — inspiration for fairness metric design

---

<div align="center">
  <br>
  <strong>⚖️ Hire on merit. Audit your bias. Build fairer teams.</strong>
  <br><br>
  If this project helped you, a ⭐ on GitHub means a lot!
</div>
