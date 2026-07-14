# HealthPilot AI

A clinical screening workspace that packages three core machine learning approaches —
**classification**, **regression**, and **anomaly detection** — alongside a conversational
health assistant and a document-grounded question answerer.

## Features

| Page | ML topic | What it does |
|---|---|---|
| Risk Classifier | Classification (Logistic Regression) | Estimates diabetes risk from 8 vitals, with per-patient contributing factors |
| Metric Regressor | Regression (Linear Regression) | Predicts a chosen clinical value (e.g. Glucose) from the rest of the chart |
| Anomaly Scanner | Unsupervised anomaly detection (Isolation Forest) | Flags vitals that fall outside the typical population pattern |
| Chat Assistant | LLM agent (Groq) | Conversational Q&A on general health topics |
| Report Generator | LLM agent (Groq) | Turns readings + risk score into a plain-language summary |
| Records Analyzer | Retrieval (TF-IDF) + LLM | Upload a text file, ask questions grounded in its content |
| Stats Dashboard | EDA | Dataset distributions and feature correlations |

This is a Streamlit app — a Python web app framework. It runs its own local server and
opens in your regular browser; it isn't a static HTML file you double-click.

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the app at `http://localhost:8501` in your default browser.

## Enable the AI features (optional but recommended)

The Chat Assistant, Report Generator, and Records Analyzer call the Groq API.

1. Get a free API key at https://console.groq.com
2. In the running app, open **Settings & Keys** in the sidebar
3. Paste your key — it's kept only in memory for that session, never written to disk

Everything else (Risk Classifier, Metric Regressor, Anomaly Scanner, Stats Dashboard)
works fully offline with no key required.

## Deploy so anyone can open it in a browser

**Streamlit Community Cloud (free, easiest)**
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → "New app" → point it at the repo and `app.py`
3. Add `GROQ_API_KEY` under app settings → Secrets if you want the AI features on by default

**Render**
1. Push this folder to a GitHub repo, create a new **Web Service** on Render
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Project structure

```
healthpilot-ai/
├── app.py                 # Streamlit app: layout, navigation, all pages
├── requirements.txt
├── README.md
├── data/
│   └── diabetes.csv       # training data for all three models
└── utils/
    ├── models.py           # classifier / regressor / anomaly detector training + inference
    └── assistant.py        # Groq chat wrapper + lightweight TF-IDF retrieval
```

## Disclaimer

All predictions are screening estimates for informational and educational use only.
They are not a substitute for professional medical advice, diagnosis, or treatment.
