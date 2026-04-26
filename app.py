"""
╔══════════════════════════════════════════════════════════════════╗
║          THE MERIT-CHECKER  —  HR Bias Audit Platform  v3       ║
║  Universal — works with ANY CSV, any size, any column names.    ║
╚══════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="The Merit-Checker", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.big-title{font-family:'DM Serif Display',serif;font-size:2.4rem;color:#0f172a;margin-bottom:.15rem;}
.sub{font-size:.95rem;color:#64748b;margin-bottom:1rem;}
.philosophy-box{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;
  border-radius:16px;padding:1.5rem 2rem;margin:1rem 0 1.5rem;}
.philosophy-box h3{font-family:'DM Serif Display',serif;font-size:1.3rem;margin-bottom:.4rem;}
.philosophy-box p{font-size:.87rem;opacity:.83;line-height:1.65;margin:0;}
.metric-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:1.1rem 1.4rem;text-align:center;height:100%;}
.metric-val{font-size:1.9rem;font-weight:600;}
.metric-label{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;}
.alarm-card{border-radius:12px;padding:.9rem 1.4rem;margin:.4rem 0;border-left:5px solid;}
.alarm-red{background:#FEF2F2;border-color:#EF4444;color:#7F1D1D;}
.alarm-orange{background:#FFF7ED;border-color:#F97316;color:#7C2D12;}
.alarm-green{background:#F0FDF4;border-color:#22C55E;color:#14532D;}
.alarm-blue{background:#EFF6FF;border-color:#3B82F6;color:#1e3a5f;}
.stTabs [data-baseweb="tab-list"]{gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:8px 20px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
TARGET_KW    = ["hired","decision","hiring","selected","shortlisted","outcome",
                "result","screening_result","interview_call","accepted","offer"]
PROTECTED_KW = ["gender","sex","age","race","ethnicity","nationality","religion",
                "disability","marital","caste","origin","color","colour"]
LEGIT_KW     = ["score","experience","years","exp","education","degree","test",
                "interview","skill","resume","gpa","grade","rating","cert",
                "project","github","aptitude","technical","communication",
                "merit","qualification","cgpa","marks","percentage","match"]

def auto_detect(df):
    cols = df.columns.tolist(); low = [c.lower() for c in cols]
    def first(kws):
        for kw in kws:
            for i,c in enumerate(low):
                if kw in c: return cols[i]
        return None
    def many(kws): return [cols[i] for i,c in enumerate(low) if any(kw in c for kw in kws)]

    target    = first(TARGET_KW)
    protected = many(PROTECTED_KW)
    legit     = many(LEGIT_KW)
    # fallback: all numeric cols are legit
    if not legit:
        legit = df.select_dtypes(include=np.number).columns.tolist()
    # fallback: all object cols with few unique values are protected
    if not protected:
        protected = [c for c in df.columns
                     if df[c].dtype == object and 2 <= df[c].nunique() <= 15
                     and c != target]
    legit     = [c for c in legit     if c != target and c not in protected]
    protected = [c for c in protected if c != target]
    return target, legit[:10], protected[:6]

_POS = {"1","yes","true","hired","selected","shortlisted",
        "accepted","offer","approved","pass","y","positive"}
_NEG = {"0","no","false","rejected","not hired","not selected",
        "not shortlisted","n","fail","negative","declined"}

def to_binary(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        uv = set(s.dropna().unique())
        if uv <= {0,1,0.0,1.0}: return s.fillna(0).astype(int), None
        med = s.median()
        return (s >= med).astype(int), f"Binarized at median ({med:.1f})"
    low = s.astype(str).str.strip().str.lower()
    return low.isin(_POS).astype(int), None

def make_sample(n=500, seed=42):
    np.random.seed(seed)
    ye = np.random.randint(0,25,n)
    ts = np.round(np.random.normal(65,15,n).clip(20,100),1)
    iv = np.round(np.random.normal(60,18,n).clip(10,100),1)
    el = np.random.choice(["High School","Bachelor's","Master's","PhD"],n,p=[.15,.50,.28,.07])
    em = {"High School":0,"Bachelor's":1,"Master's":2,"PhD":3}
    me = .35*ts+.30*iv+.20*np.clip(ye*4,0,100)+.15*(np.array([em[e] for e in el])/3*100)
    gn = np.random.choice(["Male","Female","Non-binary"],n,p=[.52,.44,.04])
    rc = np.random.choice(["White","Asian","Black","Hispanic","Other"],n,p=[.45,.20,.15,.15,.05])
    ag = np.random.randint(22,63,n)
    hp = np.clip(1/(1+np.exp(-(me-58)/12))
         +np.where(ag>45,-.18,0)+np.where(gn=="Female",-.10,np.where(gn=="Non-binary",-.14,0))
         +np.where(rc=="Black",-.15,np.where(rc=="Hispanic",-.10,0)),.05,.95)
    return pd.DataFrame({
        "Years_Experience":ye,"Education_Level":el,"Technical_Score":ts,
        "Interview_Score":iv,"Merit_Score":np.round(me,1),
        "Age":ag,"Gender":gn,"Race":rc,"Hired":(np.random.rand(n)<hp).astype(int)})

def alarm_box(css, icon, title, body):
    st.markdown(f'<div class="alarm-card alarm-{css}"><strong>{icon} {title}</strong>'
                f'<br><span style="font-size:.88rem">{body}</span></div>',
                unsafe_allow_html=True)

COLORS = ["#EF4444","#F97316","#FACC15","#22C55E"]

def bar_chart(gs, x, y, title, avg, attr):
    fig = px.bar(gs, x=x, y=y, text=gs[y].apply(lambda v:f"{v:.1%}"),
        color=y, color_continuous_scale=["#EF4444","#F97316","#22C55E"],
        range_color=[0,1], title=title, labels={y:"Rate",x:attr})
    fig.add_hline(y=avg, line_dash="dash", line_color="#64748b",
                  annotation_text=f"avg {avg:.1%}", annotation_position="top right")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=300,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45,b=10,l=10,r=10), font=dict(family="DM Sans"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RUN FULL ANALYSIS  — called once, results stored in session_state
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(df_raw, target_col, legit_feats, protected_attrs, threshold):
    """Train model + compute all bias metrics. Returns dict or raises."""
    y, warn = to_binary(df_raw[target_col])
    n = len(df_raw)

    # ── encode features ──────────────────────────────────────────────────────
    X = df_raw[legit_feats].copy()
    for c in X.select_dtypes("object").columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # ── train merit model ─────────────────────────────────────────────────────
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    # cross-val only if enough data
    if n >= 20:
        cv_folds = min(5, y.value_counts().min())  # never more folds than minority class
        cv_folds = max(2, cv_folds)
        cv_acc = cross_val_score(pipe, X, y, cv=StratifiedKFold(cv_folds),
                                 scoring="accuracy").mean()
    else:
        cv_acc = accuracy_score_simple(y, y_pred)

    # ── bias metrics per protected attribute ──────────────────────────────────
    bias = {}
    for attr in protected_attrs:
        s = df_raw[attr].copy()
        # bin numeric cols into quartiles / age brackets
        if pd.api.types.is_numeric_dtype(s):
            if "age" in attr.lower():
                try: s = pd.cut(s, bins=[0,30,40,50,200], labels=["≤30","31-40","41-50","50+"])
                except: s = pd.qcut(s, 4, duplicates="drop")
            else:
                try: s = pd.qcut(s, 4, duplicates="drop")
                except: s = s.astype(str)
        else:
            s = s.astype(str).str.strip()

        if s.nunique() < 2: continue

        gs_df = pd.DataFrame({"g": s, "a": y.values, "p": y_pred})
        gs = gs_df.groupby("g", observed=True).agg(
            n=("a","count"), ahr=("a","mean"), phr=("p","mean")).reset_index()

        try:    dpd = demographic_parity_difference(y, y_pred, sensitive_features=s)
        except: dpd = np.nan
        try:    dpr = demographic_parity_ratio(y, y_pred, sensitive_features=s)
        except: dpr = np.nan
        try:    eod = equalized_odds_difference(y, y_pred, sensitive_features=s)
        except: eod = np.nan

        pct = abs(dpd)*100 if not np.isnan(dpd) else 0
        sev = "HIGH" if pct >= threshold*1.5 else ("CAUTION" if pct >= threshold else "FAIR")
        bias[attr] = dict(gs=gs, dpd=dpd, dpr=dpr, eod=eod, pct=pct, sev=sev)

    # feature importance
    coefs = pipe.named_steps["clf"].coef_[0]
    n_show = min(len(legit_feats), len(coefs))
    imp = pd.DataFrame({"Feature": legit_feats[:n_show], "Coeff": coefs[:n_show]})

    return dict(y=y, y_pred=y_pred, cv_acc=cv_acc, imp=imp, bias=bias,
                warn=warn, n=n, hire_rate=y.mean())

def accuracy_score_simple(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean()


# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════  SIDEBAR  ═══════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ The Merit-Checker")
    st.caption("HR Bias Audit Platform")
    st.divider()

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    src = st.radio("", ["Upload my own CSV", "Use built-in sample data"],
                   label_visibility="collapsed",
                   key="data_source")

    df_raw = st.session_state.get("df_raw", None)

    if src == "Upload my own CSV":
        up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if up:
            try:
                new_df = pd.read_csv(up)
                # reset if file changed
                if st.session_state.get("last_filename") != up.name:
                    st.session_state["df_raw"]        = new_df
                    st.session_state["last_filename"] = up.name
                    st.session_state.pop("results", None)
                df_raw = st.session_state["df_raw"]
                st.success(f"✅ {up.name}\n{len(df_raw):,} rows × {df_raw.shape[1]} cols")
                if st.button("🗑 Clear and upload a different file"):
                    for k in ["df_raw","last_filename","results"]:
                        st.session_state.pop(k, None)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read: {e}")
    else:
        if st.button("▶ Load sample dataset", type="primary"):
            st.session_state["df_raw"] = make_sample()
            st.session_state.pop("results", None)
            st.session_state.pop("last_filename", None)
        if "df_raw" in st.session_state and st.session_state.get("last_filename") is None:
            df_raw = st.session_state["df_raw"]
            st.success(f"✅ Sample data — {len(df_raw):,} rows")

    if df_raw is None:
        st.info("Upload a CSV or load sample data to begin.")
        st.divider(); st.caption("Built with Fairlearn · scikit-learn · Streamlit")
        # show welcome page without stopping sidebar
    else:
        st.divider()
        # ── Column mapping ────────────────────────────────────────────────────
        st.markdown("### 🗂 Column Mapping")
        st.caption("Auto-detected from your column names — adjust if needed.")

        auto_t, auto_l, auto_p = auto_detect(df_raw)
        all_cols = df_raw.columns.tolist()

        target_col = st.selectbox("🎯 Outcome column (Hired / Selected)",
            all_cols,
            index=all_cols.index(auto_t) if auto_t in all_cols else 0,
            help="Column that says whether the candidate was hired or not")

        legit_feats = st.multiselect("✅ Legitimate features (skills / scores)",
            [c for c in all_cols if c != target_col],
            default=[c for c in auto_l if c != target_col],
            help="Test scores, experience, education — the ONLY things that should matter")

        # If no protected auto-detected, suggest all categorical columns
        cat_cols = [c for c in all_cols if c != target_col and c not in legit_feats]
        protected_attrs = st.multiselect("🔍 Protected attributes to audit",
            cat_cols,
            default=[c for c in auto_p if c in cat_cols],
            help="Gender, age, race, religion, nationality — should have ZERO influence")

        st.divider()
        threshold = st.slider("⚙️ Bias alarm threshold (%)", 3, 25, 10,
            help="Alert fires when hire-rate gap between groups exceeds this %")

        # Auto-detection log
        with st.expander("🔍 Auto-detection log"):
            st.write(f"**Target:** `{auto_t}`")
            st.write(f"**Legit detected:** {auto_l}")
            st.write(f"**Protected detected:** {auto_p}")

        st.divider()
        st.caption("Built with Fairlearn · scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════  MAIN AREA  ═════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="big-title">⚖️ The Merit-Checker</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload any hiring dataset — get an instant, universal bias audit.</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="philosophy-box">
  <h3>The Core Principle</h3>
  <p>A candidate's <strong>Legitimate Features</strong> (scores, experience, qualifications) should be
  the <em>only</em> driver of any hiring decision. <strong>Protected Attributes</strong> (age, gender,
  race, religion) are irrelevant to job performance and must have zero influence. This tool fires an
  alarm when your data shows otherwise — regardless of dataset size, structure, or source.</p>
</div>""", unsafe_allow_html=True)

# ── No data ────────────────────────────────────────────────────────────────────
if df_raw is None:
    c1, c2, c3 = st.columns(3)
    for col, num, title, body in [
        (c1, "1", "Upload CSV", "Any hiring or recruitment CSV from Kaggle or your company — the tool adapts automatically."),
        (c2, "2", "Map Columns", "Review the auto-detected column roles in the sidebar. Adjust any that look wrong."),
        (c3, "3", "Get Report", "Instant bias audit across every protected attribute — with alarms, charts, and recommendations."),
    ]:
        with col:
            st.markdown(f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
            padding:1.5rem;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#0f172a">{num}</div>
            <div style="font-weight:600;margin:.4rem 0">{title}</div>
            <div style="font-size:.85rem;color:#64748b">{body}</div></div>""",
            unsafe_allow_html=True)
    st.markdown("""---
**🔍 Good Kaggle searches:** `HR Analytics dataset` · `Hiring bias dataset` ·
`Fair recruitment dataset` · `Resume screening dataset`""")
    st.stop()

# ── Validate ───────────────────────────────────────────────────────────────────
issues = []
if not legit_feats:    issues.append("Select at least one **legitimate feature** in the sidebar.")
if not protected_attrs: issues.append("Select at least one **protected attribute** to audit.")

if issues:
    for i in issues: st.warning(i)
    st.info("👈 Configure columns in the sidebar, then results appear automatically.")
    st.stop()

# ── Dataset banner ─────────────────────────────────────────────────────────────
y_preview, _ = to_binary(df_raw[target_col])
name = st.session_state.get("last_filename", "sample data")
st.markdown(f"""<div class="alarm-card alarm-blue" style="margin-bottom:1rem">
  📊 <strong>Dataset ready — {len(df_raw):,} rows × {df_raw.shape[1]} columns</strong>&nbsp;&nbsp;|&nbsp;&nbsp;
  Outcome: <code>{target_col}</code>&nbsp;&nbsp;|&nbsp;&nbsp;
  Features ({len(legit_feats)}): {" · ".join(f"<code>{f}</code>" for f in legit_feats[:6])}
  {"..." if len(legit_feats)>6 else ""}&nbsp;&nbsp;|&nbsp;&nbsp;
  Protected: {" · ".join(f"<code>{p}</code>" for p in protected_attrs) or "(none selected)"}
</div>""", unsafe_allow_html=True)

# ── RUN analysis (cache by config signature) ───────────────────────────────────
sig = (id(df_raw), target_col, tuple(sorted(legit_feats)),
       tuple(sorted(protected_attrs)), threshold)

if st.session_state.get("_sig") != sig:
    with st.spinner("⚙️ Running analysis — training merit model and computing bias metrics…"):
        try:
            res = run_analysis(df_raw, target_col, legit_feats, protected_attrs, threshold)
            st.session_state["results"] = res
            st.session_state["_sig"]    = sig
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

R = st.session_state["results"]
y_true = R["y"].values;  y_pred = R["y_pred"]
bias   = R["bias"];       cv_acc = R["cv_acc"]
n      = R["n"]


# ══════════════════════════════════════════════════════════════════════════════
# TABS  — all data pre-computed, so every tab works immediately
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview", "🤖 Merit Model", "🚨 Bias Alarms", "📋 Full Report"
])


# ════════════════ TAB 1 — DATASET OVERVIEW ════════════════════════════════════
with tab1:
    total = n; hired = int(y_true.sum()); rej = total-hired; hr = hired/total*100

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,clr) in zip([c1,c2,c3,c4],[
        ("Total Candidates", f"{total:,}", "#3B82F6"),
        ("Hired / Selected", f"{hired:,}", "#22C55E"),
        ("Rejected",         f"{rej:,}",   "#EF4444"),
        ("Hire Rate",        f"{hr:.1f}%", "#8B5CF6"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{clr}">'
                        f'{val}</div><div class="metric-label">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if n < 20:
        st.info(f"ℹ️ Small dataset ({n} rows). Charts and metrics shown — statistical power is limited. "
                "More data = more reliable conclusions.")

    left, right = st.columns(2)

    with left:
        st.subheader("Hire rate by protected attribute")
        if not protected_attrs:
            st.info("No protected attributes selected in sidebar.")
        else:
            for attr in protected_attrs:
                s = df_raw[attr].copy()
                if pd.api.types.is_numeric_dtype(s):
                    try: s = pd.qcut(s, min(4, s.nunique()), duplicates="drop")
                    except: s = s.astype(str)
                gs = pd.DataFrame({"g":s.astype(str),"h":y_true}).groupby("g",observed=True)
                gd = gs.agg(n=("h","count"),rate=("h","mean")).reset_index()
                gd["pct"] = gd["rate"]*100
                fig = px.bar(gd, x="g", y="pct",
                    text=gd["pct"].apply(lambda v:f"{v:.1f}%"),
                    color="pct", color_continuous_scale=["#EF4444","#F97316","#22C55E"],
                    range_color=[0,100], title=f"Hire Rate by {attr}",
                    labels={"pct":"Hire Rate (%)","g":attr})
                fig.update_traces(textposition="outside")
                fig.update_layout(coloraxis_showscale=False, height=260,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40,b=10,l=10,r=10), font=dict(family="DM Sans"))
                st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Feature distributions")
        num_f = [f for f in legit_feats if pd.api.types.is_numeric_dtype(df_raw[f])]
        if num_f:
            fc = st.selectbox("Select feature", num_f)
            colors = pd.Series(y_true).map({1:"Hired",0:"Rejected"})
            fig2 = px.histogram(df_raw, x=fc, color=colors,
                barmode="overlay", opacity=.75,
                color_discrete_map={"Hired":"#22C55E","Rejected":"#EF4444"},
                title=f"{fc} — Hired vs Rejected")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40,b=10,l=10,r=10), font=dict(family="DM Sans"), height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric legitimate features to plot.")

    with st.expander(f"📋 Raw data — all {len(df_raw):,} rows"):
        st.dataframe(df_raw, use_container_width=True, height=400)


# ════════════════ TAB 2 — MERIT MODEL ════════════════════════════════════════
with tab2:
    st.subheader("Merit-Only Model")
    st.markdown(
        "Trained on **only** legitimate features (no demographics). "
        "Its predictions are our fairness baseline — what *should* have happened "
        "if only merit mattered.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Features used",    len(legit_feats))
    c2.metric("Model accuracy",   f"{cv_acc:.1%}",
              help="Cross-validated if ≥20 rows, else train accuracy")
    c3.metric("Model hire rate",  f"{y_pred.mean()*100:.1f}%")

    acc_msg = (
        f"**{cv_acc:.1%} accuracy** predicting actual decisions from merit alone. "
        + ("High accuracy → decisions were mostly merit-driven. "
           if cv_acc >= 0.70 else
           "Lower accuracy → something *else* was influencing decisions — likely protected attributes. ")
        + (f"*(Note: only {n} rows — limited statistical power.)*" if n < 50 else "")
    )
    st.info(f"**Features:** {', '.join(legit_feats)}\n\n{acc_msg}")

    if R["warn"]:
        st.warning(f"Target column note: {R['warn']}")

    # Feature importance chart
    imp = R["imp"].sort_values("Coeff", key=abs, ascending=False)
    fig_i = px.bar(imp, x="Coeff", y="Feature", orientation="h",
        color="Coeff", color_continuous_scale=["#EF4444","#94A3B8","#22C55E"],
        title="Merit Feature Importance (Logistic Regression Coefficients)",
        labels={"Coeff":"Coefficient","Feature":"Feature"})
    fig_i.update_layout(coloraxis_showscale=False,
        height=max(250, len(legit_feats)*45),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45,b=15,l=15,r=15), font=dict(family="DM Sans"))
    st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("""
    **How to read this:** A large positive coefficient means higher values of that feature
    strongly predict being hired. A near-zero coefficient means the feature barely influenced
    decisions in your data.
    """)


# ════════════════ TAB 3 — BIAS ALARMS ════════════════════════════════════════
with tab3:
    st.subheader("🚨 Bias Alarms")
    st.markdown("For each protected attribute we measure the hire-rate gap **after controlling for "
                "merit**. Key metric: **Demographic Parity Difference (DPD)** — gap between "
                "best-treated and worst-treated group.")

    if not bias:
        alarm_box("blue", "ℹ️", "No Protected Attributes Analysed",
            "Select protected attributes (gender, age, race…) in the sidebar to see bias alarms.")
        st.stop()

    for attr, m in bias.items():
        st.markdown(f"#### {attr}")
        sev, pct = m["sev"], m["pct"]
        css  = {"HIGH":"red","CAUTION":"orange","FAIR":"green"}[sev]
        icon = {"HIGH":"🚨","CAUTION":"⚠️","FAIR":"✅"}[sev]
        msg  = {
            "HIGH":    f"Hire rates differ by <strong>{pct:.1f}%</strong> between groups — "
                       f"{pct/threshold:.1f}× your {threshold}% threshold. "
                       f"<strong>{attr}</strong> is influencing decisions beyond merit.",
            "CAUTION": f"Hire rates differ by <strong>{pct:.1f}%</strong> — above your "
                       f"{threshold}% threshold. Investigate further.",
            "FAIR":    f"Hire rates differ by only <strong>{pct:.1f}%</strong> — within your "
                       f"{threshold}% threshold. No significant bias detected.",
        }[sev]
        alarm_box(css, icon, f"{sev} — {attr}", msg)

        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Demographic Parity Diff", f"{pct:.1f}%",
                   help="Hire-rate gap between best and worst treated group. Lower = fairer.")
        mc2.metric("Demographic Parity Ratio",
                   f"{m['dpr']:.3f}" if not np.isnan(m['dpr']) else "N/A",
                   help="Lowest ÷ highest group hire rate. 1.0 = perfect parity.")
        mc3.metric("Equalized Odds Diff",
                   f"{abs(m['eod'])*100:.1f}%" if not np.isnan(m['eod']) else "N/A",
                   help="Gap in true positive rates across groups.")

        gs = m["gs"]
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(bar_chart(gs,"g","ahr",f"Actual hire rate by {attr}",
                                      y_true.mean(), attr), use_container_width=True)
        with cb:
            st.plotly_chart(bar_chart(gs,"g","phr",f"Merit prediction by {attr}",
                                      y_pred.mean(), attr), use_container_width=True)
        st.caption("Left = what actually happened · Right = what merit alone predicts. "
                   "A big gap between the two = bias signal.")

        if n < 30:
            st.warning(f"⚠️ Only {n} rows — treat this as a directional signal, not a definitive verdict.")
        st.divider()


# ════════════════ TAB 4 — FULL REPORT ════════════════════════════════════════
with tab4:
    st.subheader("📋 Full Audit Report")

    if not bias:
        alarm_box("blue","ℹ️","No Bias Data",
            "Select protected attributes in the sidebar to generate the full report.")
        st.stop()

    sc_rows = []
    for attr, m in bias.items():
        sc_rows.append({
            "Attribute": attr,
            "Severity":  m["sev"],
            "DPD (%)":   round(m["pct"],1),
            "DPR":       round(m["dpr"],3) if not np.isnan(m["dpr"]) else "N/A",
            "EOD (%)":   round(abs(m["eod"])*100,1) if not np.isnan(m["eod"]) else "N/A",
        })
    sc = pd.DataFrame(sc_rows)
    high  = sc[sc["Severity"]=="HIGH"]
    caut  = sc[sc["Severity"]=="CAUTION"]
    flagged = set(high["Attribute"].tolist() + caut["Attribute"].tolist())

    # Overall verdict
    if len(high):
        alarm_box("red","🚨","Bias Detected",
            f"High bias found on: <strong>{', '.join(high['Attribute'])}</strong>. "
            "Candidates with identical merit scores are receiving different outcomes based on "
            "protected characteristics.")
    elif len(caut):
        alarm_box("orange","⚠️","Caution Advised",
            f"Possible bias on: <strong>{', '.join(caut['Attribute'])}</strong>. "
            "Disparities exceed your threshold. Proactive review recommended.")
    else:
        alarm_box("green","✅","No Significant Bias Found",
            "Your hiring data does not show statistically significant bias across the audited "
            "attributes. Continue monitoring regularly.")

    # Scorecard
    st.subheader("Bias scorecard")
    sc["Status"] = sc["Severity"].map({"HIGH":"🚨 High Bias","CAUTION":"⚠️ Caution","FAIR":"✅ Fair"})
    st.dataframe(sc[["Attribute","Status","DPD (%)","DPR","EOD (%)"]],
                 use_container_width=True, hide_index=True)

    # Radar chart (only if ≥3 attributes)
    if len(sc) >= 2:
        ra = sc["Attribute"].tolist()
        rs = [round(100-min(v,100)) for v in sc["DPD (%)"].tolist()]
        ra += [ra[0]]; rs += [rs[0]]
        fig_r = go.Figure(go.Scatterpolar(r=rs, theta=ra, fill="toself",
            fillcolor="rgba(59,130,246,.15)", line=dict(color="#3B82F6",width=2)))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
            showlegend=False, title="Fairness Radar (100 = perfectly fair)",
            height=380, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"))
        st.plotly_chart(fig_r, use_container_width=True)

    # Recommendations
    st.subheader("Recommendations")
    recs = [
        ("🎯 Blind Screening",
         "Remove protected attributes (name, age, gender, race) from initial screening. "
         "Use anonymized candidate IDs until the interview stage."),
        ("📝 Structured Interviews",
         "Standardize interview questions scored on a rubric — same questions and scoring for every candidate."),
        ("📊 Audit Quarterly",
         "Run this audit every quarter. Treat any DPD above your threshold as a formal incident."),
        ("🤖 Calibrate Your ATS",
         "Audit your ATS rejection logic — many systems have built-in biases from their training data."),
        ("🏋️ Bias Training",
         "Run unconscious bias training focused on the attributes flagged in this report — show interviewers the numbers."),
        ("📋 Mandate Score Sheets",
         "Require numeric score breakdowns from all interviewers before any decision is made."),
        ("🔍 Re-review Rejections",
         "Manually audit 20% of rejected candidates from flagged groups and compare merit scores."),
    ]
    for a in flagged:
        al = a.lower()
        if any(k in al for k in ["gender","sex"]):
            recs.insert(1,("⚧ Gender Parity",
                "Audit job descriptions for gendered language. Ensure gender-diverse interview panels."))
        if "age" in al:
            recs.insert(1,("🎂 Age Discrimination",
                "Remove graduation years from CVs. Review notes for 'overqualified' language."))
        if any(k in al for k in ["race","ethnicity","caste","origin"]):
            recs.insert(1,("🌍 Racial Equity",
                "Implement name-blind screening. Audit referral programs — they reinforce demographic homogeneity."))

    for i,(t,b) in enumerate(recs[:8],1):
        with st.expander(f"{i}. {t}"):
            st.write(b)

    # Export
    st.divider()
    st.subheader("Export")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Bias Scorecard (CSV)",
            sc.to_csv(index=False), "merit_checker_bias_report.csv", "text/csv")
    with d2:
        out = df_raw.copy()
        out["Merit_Model_Prediction"] = y_pred
        out["Actual_Outcome_Binary"]  = y_true
        st.download_button("📥 Download Dataset + Predictions (CSV)",
            out.to_csv(index=False), "merit_checker_annotated.csv", "text/csv")