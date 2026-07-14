import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.models import (
    load_data, train_classifier, train_regressor, train_anomaly_detector,
    predict_risk, predict_metric, score_anomaly, FEATURE_COLUMNS,
)
from utils.assistant import chat, generate_report, chunk_text, build_index, retrieve

st.set_page_config(
    page_title="HealthPilot AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg-app: #0E1220;
  --bg-side: #090B13;
  --bg-card: #161A2B;
  --border: #262B41;
  --accent: #8B6BFF;
  --accent-soft: rgba(139,107,255,0.12);
  --text: #E8E9F1;
  --muted: #8A8FA3;
  --danger: #FF6B6B;
  --safe: #4CD9B0;
  --warn: #F5B84C;
}
html, body, [class*="css"]  { font-family: 'Segoe UI', Inter, sans-serif; }
.stApp{ background: var(--bg-app); color: var(--text); }
section[data-testid="stSidebar"]{
  background: var(--bg-side);
  border-right: 1px solid var(--border);
}
#MainMenu, footer, header[data-testid="stHeader"]{ visibility: hidden; height:0; }

.brand-row{ display:flex; align-items:center; gap:10px; margin: 6px 0 2px 0; }
.brand-title{ font-size:22px; font-weight:800; color:var(--accent); }
.brand-tag{ font-size:12.5px; color:var(--muted); font-style:italic; margin-bottom:18px; }

.nav-section-label{
  font-size:11px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin: 18px 0 6px 4px;
}

div.stButton > button{
  width:100%; text-align:left; background:transparent; border:none;
  color:var(--muted); padding:9px 10px; border-radius:8px; font-size:14.5px;
}
div.stButton > button:hover{ background: var(--accent-soft); color:var(--text); }
div.stButton > button:focus{ box-shadow:none !important; color:var(--text); }

.nav-active > button{ background: var(--accent-soft); color: var(--text) !important; font-weight:600; }

h1, h2, h3 { color: var(--text); }
.hero-h1{ color:var(--accent); font-size:40px; font-weight:800; margin-bottom:6px;}
.hero-sub{ color:var(--muted); font-size:15.5px; max-width:800px; line-height:1.55; }

.section-h{ display:flex; align-items:center; gap:10px; margin:34px 0 16px 0; font-size:22px; font-weight:800;}

.op-card{
  background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
  padding:22px; height:100%;
}
.op-card:hover{ border-color: var(--accent); }
.op-icon{ font-size:22px; }
.op-title{ font-size:18px; font-weight:700; margin:10px 0 8px 0; }
.op-desc{ color:var(--muted); font-size:13.5px; line-height:1.55; }

.pill{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12.5px; font-weight:600;}
.pill-low{ background:rgba(76,217,176,0.14); color:var(--safe);}
.pill-mod{ background:rgba(245,184,76,0.16); color:var(--warn);}
.pill-high{ background:rgba(255,107,107,0.14); color:var(--danger);}

.metric-box{
  background:var(--bg-card); border:1px solid var(--border); border-radius:12px;
  padding:16px 18px;
}
.metric-box .k{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.metric-box .v{ font-size:26px; font-weight:800; margin-top:4px; }

.disclaimer{
  border-left:3px solid var(--warn); padding:10px 14px; color:var(--muted);
  font-size:12.5px; background:rgba(245,184,76,0.06); border-radius:6px; margin-top:18px;
}

.chat-bubble-user{
  background: var(--accent-soft); border:1px solid var(--accent);
  padding:10px 14px; border-radius:12px; margin:6px 0; max-width:80%; margin-left:auto;
}
.chat-bubble-bot{
  background: var(--bg-card); border:1px solid var(--border);
  padding:10px 14px; border-radius:12px; margin:6px 0; max-width:80%;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = "llama-3.1-8b-instant"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ----------------------------------------------------------------------------
# CACHED MODEL TRAINING
# ----------------------------------------------------------------------------
@st.cache_resource
def get_data():
    return load_data()

@st.cache_resource
def get_classifier():
    return train_classifier(get_data())

@st.cache_resource
def get_regressor(target):
    return train_regressor(get_data(), target)

@st.cache_resource
def get_anomaly_model():
    return train_anomaly_detector(get_data())


# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
PAGES = [
    ("Home", "🏠"),
    ("Chat Assistant", "💬"),
    ("Risk Classifier", "🩺"),
    ("Metric Regressor", "📈"),
    ("Anomaly Scanner", "🛡️"),
    ("Report Generator", "📝"),
    ("Records Analyzer", "📁"),
    ("Stats Dashboard", "📊"),
    ("Settings & Keys", "⚙️"),
    ("About", "ℹ️"),
]

with st.sidebar:
    st.markdown(
        '<div class="brand-row">🏥 <span class="brand-title">HealthPilot AI</span></div>'
        '<div class="brand-tag">Your AI Health Co-Pilot</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-section-label">Main Menu</div>', unsafe_allow_html=True)

    for name, icon in PAGES:
        active = st.session_state.page == name
        wrapper_class = "nav-active" if active else ""
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        dot = "🔴" if active else "⚪"
        if st.button(f"{dot} {icon}  {name}", key=f"nav_{name}"):
            st.session_state.page = name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="nav-section-label">LLM Provider</div>', unsafe_allow_html=True)
    st.selectbox("LLM Provider", ["Groq"], label_visibility="collapsed")
    st.markdown('<div class="nav-section-label">Model</div>', unsafe_allow_html=True)
    st.session_state.model_name = st.selectbox(
        "Model", MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.model_name),
        label_visibility="collapsed",
    )

# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------

def page_home():
    st.markdown('<div class="hero-h1">Welcome to HealthPilot AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">HealthPilot AI is your clinical co-pilot. It classifies risk from '
        'patient vitals, predicts continuous health metrics, flags anomalous readings, answers '
        'health questions in plain language, and turns a chart into a written summary — all from '
        'one workspace.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">🚀 Core Operations</div>', unsafe_allow_html=True)

    cards = [
        ("🩺", "Risk Classifier", "Classify diabetes risk from eight vitals using a logistic "
         "regression model, with a per-patient breakdown of contributing factors.", "Risk Classifier"),
        ("📈", "Metric Regressor", "Predict a missing or follow-up clinical value — like glucose or "
         "BMI — from the rest of the chart using linear regression.", "Metric Regressor"),
        ("📁", "Records Analyzer", "Upload notes or reports and ask questions. Retrieves the most "
         "relevant passages and answers grounded in your own documents.", "Records Analyzer"),
        ("🛡️", "Anomaly Scanner", "Detect vitals that fall outside the normal population pattern "
         "using an isolation forest, independent of any single threshold.", "Anomaly Scanner"),
        ("💬", "Chat Assistant", "Ask general health questions in plain language and get grounded, "
         "cautious answers — with a nudge to see a clinician when it matters.", "Chat Assistant"),
        ("📝", "Report Generator", "Turn a set of readings and a risk score into a short, readable "
         "summary a patient can actually use.", "Report Generator"),
    ]

    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (icon, title, desc, target) in zip(cols, cards[row_start:row_start + 3]):
            with col:
                st.markdown(
                    f'<div class="op-card"><div class="op-icon">{icon}</div>'
                    f'<div class="op-title">{title}</div>'
                    f'<div class="op-desc">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open →", key=f"open_{target}"):
                    st.session_state.page = target
                    st.rerun()


def page_chat():
    st.markdown('<div class="section-h">💬 Chat Assistant</div>', unsafe_allow_html=True)
    st.caption("General health information, not a diagnosis. For anything urgent, contact a clinician.")

    for turn in st.session_state.chat_history:
        cls = "chat-bubble-user" if turn["role"] == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{cls}">{turn["content"]}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about symptoms, readings, or general health topics…")
    if prompt:
        if not st.session_state.groq_api_key:
            st.warning("Add a Groq API key on the Settings & Keys page to enable the chat assistant.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                try:
                    reply = chat(
                        st.session_state.groq_api_key,
                        st.session_state.model_name,
                        st.session_state.chat_history[:-1],
                        prompt,
                    )
                except Exception as e:
                    reply = f"Something went wrong reaching the model: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()


def vitals_form(prefix, defaults):
    c1, c2 = st.columns(2)
    values = {}
    fields = [
        ("Pregnancies", 0, 17, 1, "Count"),
        ("Glucose", 0, 250, 1, "mg/dL"),
        ("BloodPressure", 0, 140, 1, "mm Hg"),
        ("SkinThickness", 0, 99, 1, "mm"),
        ("Insulin", 0, 900, 1, "µU/mL"),
        ("BMI", 0.0, 70.0, 0.1, "kg/m²"),
        ("DiabetesPedigreeFunction", 0.05, 2.5, 0.01, "score"),
        ("Age", 18, 100, 1, "years"),
    ]
    for i, (name, lo, hi, step, unit) in enumerate(fields):
        col = c1 if i % 2 == 0 else c2
        with col:
            values[name] = col.number_input(
                f"{name} ({unit})", min_value=lo, max_value=hi, step=step,
                value=defaults[name], key=f"{prefix}_{name}",
            )
    return values


DEFAULTS = {
    "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
    "Insulin": 80, "BMI": 28.5, "DiabetesPedigreeFunction": 0.47, "Age": 33,
}


def page_classifier():
    st.markdown('<div class="section-h">🩺 Risk Classifier</div>', unsafe_allow_html=True)
    st.caption("Logistic regression · classification")

    bundle = get_classifier()
    m1, m2 = st.columns(2)
    m1.markdown(f'<div class="metric-box"><div class="k">Test accuracy</div><div class="v">{bundle["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-box"><div class="k">ROC-AUC</div><div class="v">{bundle["auc"]:.3f}</div></div>', unsafe_allow_html=True)

    st.write("")
    values = vitals_form("clf", DEFAULTS)

    if st.button("Run classification →", key="run_clf"):
        result = predict_risk(bundle, values)
        prob = result["probability"]
        band = "pill-low" if prob < 0.3 else "pill-mod" if prob < 0.6 else "pill-high"
        label = "Low risk" if prob < 0.3 else "Moderate risk" if prob < 0.6 else "Elevated risk"

        st.markdown(f'### {prob*100:.1f}% estimated risk &nbsp; <span class="pill {band}">{label}</span>', unsafe_allow_html=True)

        st.markdown("**Top contributing factors**")
        for feat, contrib in result["contributions"][:5]:
            direction = "raises risk" if contrib > 0 else "lowers risk"
            st.write(f"- {feat}: {direction}")

        st.markdown(
            '<div class="disclaimer">Screening estimate only — not a diagnosis. '
            'Consult a licensed clinician for medical guidance.</div>',
            unsafe_allow_html=True,
        )


def page_regressor():
    st.markdown('<div class="section-h">📈 Metric Regressor</div>', unsafe_allow_html=True)
    st.caption("Linear regression · continuous prediction")

    target = st.selectbox("Value to predict", FEATURE_COLUMNS, index=FEATURE_COLUMNS.index("Glucose"))
    bundle = get_regressor(target)

    m1, m2 = st.columns(2)
    m1.markdown(f'<div class="metric-box"><div class="k">R²</div><div class="v">{bundle["r2"]:.3f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-box"><div class="k">Mean abs. error</div><div class="v">{bundle["mae"]:.1f}</div></div>', unsafe_allow_html=True)

    st.write("")
    remaining_defaults = {k: v for k, v in DEFAULTS.items() if k != target}
    c1, c2 = st.columns(2)
    values = {}
    for i, feat in enumerate(bundle["features"]):
        col = c1 if i % 2 == 0 else c2
        # Ensure `step` is the same numeric type as `value` to avoid Streamlit's
        # MixedNumericTypesError (value float vs step int).
        step = 0.1 if feat in ("BMI", "DiabetesPedigreeFunction") else 1.0
        values[feat] = col.number_input(feat, value=float(remaining_defaults[feat]), step=step, key=f"reg_{feat}")

    if st.button("Predict value →", key="run_reg"):
        pred = predict_metric(bundle, values)
        st.markdown(f'<div class="metric-box"><div class="k">Predicted {target}</div><div class="v">{pred:.1f}</div></div>', unsafe_allow_html=True)


def page_anomaly():
    st.markdown('<div class="section-h">🛡️ Anomaly Scanner</div>', unsafe_allow_html=True)
    st.caption("Isolation forest · unsupervised anomaly detection")

    bundle = get_anomaly_model()
    values = vitals_form("anom", DEFAULTS)

    if st.button("Scan reading →", key="run_anom"):
        result = score_anomaly(bundle, values)
        if result["is_anomaly"]:
            st.markdown('<span class="pill pill-high">Anomalous reading</span>', unsafe_allow_html=True)
            st.write("This combination of vitals falls outside the typical pattern seen in the population.")
        else:
            st.markdown('<span class="pill pill-low">Within normal pattern</span>', unsafe_allow_html=True)
            st.write("This combination of vitals is consistent with the typical population pattern.")
        st.caption(f"Anomaly score: {result['score']:.3f} (lower = more unusual)")


def page_report():
    st.markdown('<div class="section-h">📝 Report Generator</div>', unsafe_allow_html=True)
    st.caption("Turns a set of readings into a plain-language summary")

    bundle = get_classifier()
    values = vitals_form("rep", DEFAULTS)

    if st.button("Generate report →", key="run_report"):
        result = predict_risk(bundle, values)
        if not st.session_state.groq_api_key:
            st.warning("Add a Groq API key on the Settings & Keys page to generate a written report.")
        else:
            with st.spinner("Writing report…"):
                try:
                    report = generate_report(
                        st.session_state.groq_api_key, st.session_state.model_name,
                        values, result["probability"],
                    )
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Could not generate report: {e}")


def page_rag():
    st.markdown('<div class="section-h">📁 Records Analyzer</div>', unsafe_allow_html=True)
    st.caption("Upload a text file and ask questions grounded in its content")

    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
        vectorizer, matrix = build_index(chunks)
        st.session_state.rag_index = (chunks, vectorizer, matrix)
        st.success(f"Indexed {len(chunks)} passages.")

    question = st.text_input("Ask a question about the uploaded document")
    if question and st.session_state.rag_index:
        chunks, vectorizer, matrix = st.session_state.rag_index
        top_chunks = retrieve(question, chunks, vectorizer, matrix, top_k=3)

        if not top_chunks:
            st.write("No relevant passages found.")
        elif st.session_state.groq_api_key:
            context = "\n\n".join(top_chunks)
            with st.spinner("Reading document…"):
                try:
                    answer = chat(
                        st.session_state.groq_api_key, st.session_state.model_name, [],
                        f"Using only this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}",
                    )
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Could not reach the model: {e}")
        else:
            st.info("Add a Groq API key in Settings for a synthesized answer. Showing raw matches instead:")
            for c in top_chunks:
                st.markdown(f'<div class="op-card">{c}</div>', unsafe_allow_html=True)


def page_stats():
    st.markdown('<div class="section-h">📊 Stats Dashboard</div>', unsafe_allow_html=True)
    df = get_data()

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-box"><div class="k">Records</div><div class="v">{len(df)}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-box"><div class="k">Positive rate</div><div class="v">{df["Outcome"].mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-box"><div class="k">Median age</div><div class="v">{df["Age"].median():.0f}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-box"><div class="k">Median BMI</div><div class="v">{df["BMI"].median():.1f}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Glucose", color=df["Outcome"].map({0: "No diabetes", 1: "Diabetes"}),
                            barmode="overlay", template="plotly_dark", title="Glucose distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr = df.corr(numeric_only=True)
        fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Purples"))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            title="Feature correlation")
        st.plotly_chart(fig2, use_container_width=True)


def page_settings():
    st.markdown('<div class="section-h">⚙️ Settings & Keys</div>', unsafe_allow_html=True)

    st.session_state.groq_api_key = st.text_input(
        "Groq API key", value=st.session_state.groq_api_key, type="password",
        help="Get a free key at console.groq.com. Used for Chat Assistant, Report Generator, and Records Analyzer.",
    )
    st.session_state.model_name = st.selectbox(
        "Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.model_name),
    )
    st.caption("Keys are kept only in this session's memory — never written to disk.")


def page_about():
    st.markdown('<div class="section-h">ℹ️ About</div>', unsafe_allow_html=True)
    st.write(
        "HealthPilot AI packages three classic machine learning approaches — classification, "
        "regression, and anomaly detection — behind a single clinical screening workspace, "
        "alongside a conversational assistant and a document-grounded question answerer.\n\n"
        "All predictions are screening estimates for informational and educational use. "
        "They are not a substitute for professional medical advice, diagnosis, or treatment."
    )


PAGE_FUNCS = {
    "Home": page_home,
    "Chat Assistant": page_chat,
    "Risk Classifier": page_classifier,
    "Metric Regressor": page_regressor,
    "Anomaly Scanner": page_anomaly,
    "Report Generator": page_report,
    "Records Analyzer": page_rag,
    "Stats Dashboard": page_stats,
    "Settings & Keys": page_settings,
    "About": page_about,
}

PAGE_FUNCS[st.session_state.page]()
