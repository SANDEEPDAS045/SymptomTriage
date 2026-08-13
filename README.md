# Symptom Triage — Disease Classification

A machine-learning system that predicts a likely disease from a short symptom
checklist, then enriches the prediction with a risk score, surgery flag,
typical medications, and precautions. Trained on 47 conditions with a
Bernoulli Naive Bayes classifier over TF-IDF symptom text plus engineered
demographic features, reaching **96.8% accuracy** on a held-out test split.

> ⚠️ **Not a diagnostic tool.** This project estimates the statistically
> closest match from a small labeled dataset. It is a portfolio/education
> project, not a medical device, and should never replace professional
> medical advice.

![Landing page — "Describe what you feel. See what it could be."](assets/landing-page.png)

---

## What's in this repo

```
.
├── README.md
├── flowchart.jpg              # model prediction pipeline
├── system_design.jpg          # end-to-end system architecture
├── frontend/
│   └── index.html             # standalone web UI (no build step)
├── backend/
│   ├── app.py                 # Flask API wrapping the trained model
│   ├── prediction_helper.py   # feature engineering + predict()
│   ├── disease_data.py        # symptom list, risk/drug/precaution tables
│   ├── model.joblib           # trained BernoulliNB pipeline
│   ├── tfidf.joblib           # ⚠️ NOT included — see note below
│   └── requirements.txt
├── main.py                    # original Streamlit reference app
└── health_care_5.ipynb        # training notebook (EDA → model → export)
```

### ⚠️ Missing artifact: `tfidf.joblib`

`prediction_helper.py` loads two files at import time — `model.joblib`
**and** `tfidf.joblib`. Only `model.joblib` was provided for this build, so
the API will start but `/api/predict` will return a `503` until you export
the fitted `TfidfVectorizer` from the notebook (`health_care_5.ipynb`,
around the TF-IDF cell) and drop it in `backend/` as `tfidf.joblib`:

```python
import joblib
joblib.dump(tfidf, "tfidf.joblib")
```

The frontend is fully usable without it — it falls back to a small
client-side preview so you can review the UI/UX before wiring up the real
model (see **Demo mode** below).

---

## Quickstart

```bash
cd backend
pip install -r requirements.txt
# copy your fitted tfidf.joblib into this folder (see note above)
python app.py
```

Then open **http://localhost:5000** — Flask serves both the frontend and
the API from the same origin.

To run the original Streamlit reference app instead:

```bash
pip install -r requirements.txt   # project root requirements.txt
streamlit run main.py
```

---

## Frontend

`frontend/index.html` is a single self-contained file (HTML/CSS/vanilla JS,
no build tooling) styled as a clinical "monitor console":

- **Symptom search & chips** — type-ahead search over the 109-token symptom
  vocabulary instead of 17 stacked dropdowns; excludes already-picked
  symptoms and enforces the 3–17 range the model was trained on.
- **Live vitals panel** — mirrors `prediction_helper.process()` client-side
  (age group, symptom count, pediatric/elderly flags) so users see the same
  engineered features the model will consume, before submitting.
- **Result card** — predicted disease, a risk-score gauge (color-coded
  low/moderate/high), surgery flag, medication chips, and a precautions
  checklist, pulled from `disease_data.py` via the API response.
- **Demo mode** — if `/api/predict` is unreachable (e.g. `tfidf.joblib` is
  missing, or you're just previewing the UI), the page shows a banner and
  renders a locally-computed sample result instead of failing silently. A
  **"Load sample case"** button fills in a verified example from the
  training notebook (`age 22, female → migraine`).

No frameworks, no bundler — open `frontend/index.html` directly, or serve it
through Flask for same-origin API calls.

### Screens

**Diagnose form** — type-ahead symptom search with a live vitals read-out of
the same engineered features (age group, symptom count, pediatric/elderly
flags) the model will consume:

![Diagnose form with symptom chips and live vitals panel](assets/diagnose-form.png)

**Result card** — predicted condition, risk-score gauge, surgery flag,
medication chips, and a precautions checklist:

![Result card showing predicted condition, risk gauge, medications, and precautions]

---

## API

| Method | Route            | Body                                   | Response                                                              |
|--------|------------------|-----------------------------------------|-------------------------------------------------------------------------|
| GET    | `/api/symptoms`  | —                                       | `{ "symptoms": [...] }`                                                 |
| POST   | `/api/predict`   | `{ "age": 22, "sex": "Female", "symptoms": ["headache", "..."] }` | `{ "disease", "risk_score", "requires_surgery", "drugs", "precautions" }` |

`/api/predict` validates that at least 3 symptoms are provided (matching the
Streamlit app's requirement), pads the rest to the model's expected 17
symptom slots, and returns a `503` with a clear message if the model
artifacts can't be loaded.

---

## Model summary

| | |
|---|---|
| Algorithm | Bernoulli Naive Bayes (hyperparameters tuned with Optuna) |
| Text features | `TfidfVectorizer(max_features=15000)` over joined symptom tokens |
| Numeric features | age group, sex one-hot, symptom count, per-age-group symptom counts, pediatric-fever flag, elderly-chronic flag, male-elderly flag, symptom-count-vs-age-mean ratio |
| Feature fusion | `scipy.sparse.hstack([tfidf_vector, numerical_features])` |
| Classes | 47 diseases |
| Held-out accuracy | 96.8% (`test_size=0.25`, stratified split) |
| Training source | `health_care_5.ipynb` |

### Prediction pipeline

How a raw symptom checklist becomes a ranked diagnosis — categorical and
engineered features feed a TF-IDF vector, get fused into one sparse matrix,
and are scored by the tuned BernoulliNB classifier before being enriched
with the static risk/drug/precaution lookup tables:

![Prediction pipeline flowchart: patient input → feature engineering → TF-IDF → feature fusion → BernoulliNB → enriched report]

### System architecture

How the client, API server, model artifacts, offline training notebook, and
reporting layer fit together:

![System architecture diagram: browser/Streamlit clients, Flask API, prediction_helper, model artifacts, offline training, and reporting]

---

## Reference implementation (`main.py`)

The root-level `main.py` is the original Streamlit app the API/frontend
were derived from. It's useful as a self-contained reference for the full
symptom vocabulary and the per-disease lookup data:

- **109-symptom vocabulary** — a flat, sorted list used to populate 17
  sequential `st.selectbox` dropdowns (symptoms 1–3 required, 4–17 optional,
  each excluding symptoms already chosen).
- **`disease_advice`** — a dict of 5 precaution bullet points per disease
  (fungal infection, allergic rhinitis, GERD, diabetes, migraine, etc.).
- **`disease_map`** — a dict of `(risk_score 1–10, requires_surgery bool,
  [up to 4 common drugs])` per disease, e.g. lung cancer scores 10 and
  requires surgery, while the common cold scores 1 and doesn't.
- **`generate_pdf()`** — builds a one-page FPDF report (age, sex, symptoms,
  prediction, risk score, surgery flag, drugs, precautions) and offers it as
  a base64 download link via `st.markdown`.
- On clicking **"Diagnose Disease"**, it calls `predict()` from
  `prediction_helper.py`, renders the enriched result as a table, and
  attaches the downloadable PDF.

This is the same `disease_advice` / `disease_map` data that backs
`disease_data.py` in the Flask backend — the API and frontend are a
re-platformed version of this same logic, minus the manual PDF-download
flow.

---

## Disclaimer

Predictions are for demonstration purposes only. The training set covers a
fixed set of 47 conditions and a limited symptom vocabulary — it cannot
account for medical history, lab results, or symptoms outside that
vocabulary. Always consult a licensed medical professional for real
diagnosis and treatment decisions.
