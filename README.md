# ⚖️ The Merit-Checker
### HR Bias Audit Platform

> "Decisions should be based 100% on knowledge. Everything else is irrelevant."

The Merit-Checker is a data-science tool that audits your company's hiring data
to detect whether **Protected Attributes** (age, gender, race, religion) are
secretly influencing decisions that should be driven only by
**Legitimate Features** (test scores, experience, qualifications).

---

## How to Run (5 minutes)

### 1. Install Python
Download from https://python.org (3.10 or newer)

### 2. Install dependencies
Open your terminal (Command Prompt / Terminal app) and run:

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

Your browser will open automatically at http://localhost:8501

---

## How It Works

| Concept | Data Science Term | Example Columns |
|---|---|---|
| What should matter | Legitimate Features | Test Score, Experience, Education |
| What shouldn't matter | Protected Attributes | Age, Gender, Race |
| The outcome | Target variable | Hired (Yes/No) |

### The Analysis Pipeline

1. **Load Data** — Upload your CSV (or use our synthetic sample)
2. **Train a Merit Model** — We train a Logistic Regression using *only* Legitimate Features
3. **Compare** — We compare what the merit model *predicts* vs what actually *happened*
4. **Alarm** — If the gap between actual outcomes and merit-predictions differs by protected group → bias alarm fires

### Key Metrics Explained

| Metric | Plain English | Green Zone |
|---|---|---|
| Demographic Parity Difference (DPD) | Max hire rate gap between groups | < 5% |
| Demographic Parity Ratio (DPR) | Lowest ÷ highest hire rate | > 0.8 |
| Equalized Odds Difference (EOD) | Gap in "correct" hire rates across groups | < 5% |

---

## CSV Format

Your CSV should have columns like:

```
Years_Experience, Education_Level, Technical_Score, Interview_Score, Age, Gender, Race, Hired
```

- **Hired** column: 1 = hired, 0 = not hired
- Any column names work — you configure them in the sidebar

---

## Tech Stack

- **Streamlit** — Web app framework
- **Fairlearn (Microsoft)** — Bias detection metrics
- **scikit-learn** — Machine learning (Logistic Regression)
- **Plotly** — Interactive charts
- **pandas / numpy** — Data processing

---

## Project Structure

```
merit_checker/
├── app.py              ← Main Streamlit application
├── generate_data.py    ← Synthetic HR dataset generator
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Roadmap (Next Steps)

- [ ] Add more fairness metrics (Calibrated Equalized Odds, Counterfactual Fairness)
- [ ] Support multi-class outcomes (Hired / Shortlisted / Rejected)
- [ ] Add a "What-If" simulator — change one attribute and see how hire probability changes
- [ ] PDF report export
- [ ] Slack / email alerts for scheduled audits
- [ ] Connect to ATS APIs (Greenhouse, Lever, Workday)

---

Built with ❤️ using open-source tools from Microsoft (Fairlearn) and IBM (AIF360 compatible).
