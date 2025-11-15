# Bank Churn Predictor — Streamlit App

This repository contains a trained Random Forest model (`bank_churn_model.joblib`) and a Streamlit app to make single or batch churn predictions.

Getting started (Windows PowerShell):

1. Create a virtual environment and activate it (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

UI features:
- Single prediction: fill a form with customer attributes (auto-fill from sample rows available).
- Batch CSV: upload a CSV with original dataset columns to get predicted classes and probabilities.

Notes:
- The app mirrors preprocessing used in `churn_prediction.py` (one-hot encoding with `drop_first=True`).
- Ensure `bank_churn_model.joblib` and `Bank_Churners_Credit_Cards.csv` are in the same folder as `app.py`.
