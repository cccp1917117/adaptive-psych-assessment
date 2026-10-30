# Adaptive Psych Assessment

A rule-based psychological-assessment application prototype, with an experimental VAE-based questionnaire-shortening scaffold.

## Project status

- **Existing application:** FastAPI backend and Streamlit interface; rule-based question selection and dimension-level scoring.
- **New research prototype:** [VAE-based questionnaire shortening](experiments/vae_short_form/README.md), tested on synthetic responses only. It is separate from the application and does not establish real-scale or clinical validity.
- Personalized model-based adaptive testing, psychometric validation, and production use remain future work.

## Features
- Rule-based follow-up question selection
- Dynamic dimension-level scoring
- FastAPI backend
- Streamlit frontend
- Extensible question bank

## Roadmap
- [x] Backend prototype
- [x] Streamlit frontend
- [x] Synthetic-data VAE short-form scaffold
- [ ] Approved real-data experiments and psychometric validation
- [ ] Database
- [ ] Model-based personalized item selection
- [ ] IRT/CAT comparison

## Launch the existing application

Install dependencies:

```bash
python -m pip install fastapi uvicorn streamlit requests
```

In one terminal:

```bash
cd backend
uvicorn main:app --reload
```

In another terminal:

```bash
cd frontend
streamlit run app.py
```

Use Ctrl+C to stop either process. The existing application is a prototype, not a diagnostic tool.

## Run the VAE experiment

```bash
cd experiments/vae_short_form
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py --epochs 20
```

See the experiment README for the method, limitations, synthetic-data provenance, AI-assistance disclosure, and validation roadmap. Do not commit private participant data or claim that simulated reconstruction results validate a shortened psychological scale.
