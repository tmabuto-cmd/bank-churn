import joblib
import pandas as pd
import os


MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'bank_churn_model.joblib')
DATA_FILENAME = os.path.join(os.path.dirname(__file__), 'Bank_Churners_Credit_Cards.csv')


def load_model(path: str = None):
    """Load the trained model from a joblib file."""
    p = path or MODEL_FILENAME
    model = joblib.load(p)
    return model


def load_raw_data(path: str = None):
    p = path or DATA_FILENAME
    df = pd.read_csv(p)
    return df


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Drop training-only / ID columns as in the training script
    cols_to_drop = ['CLIENTNUM',
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols)


def build_training_features(df: pd.DataFrame):
    """Return the feature matrix X and target y created the same way as training.

    This mirrors the preprocessing done in `churn_prediction.py` so the Streamlit
    app can construct inputs with the exact same encoded columns.
    """
    df = _clean_raw(df.copy())
    if 'Attrition_Flag' in df.columns:
        df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
        df = df.drop(columns=['Attrition_Flag'])

    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    if 'Churn' in df_encoded.columns:
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
    else:
        X = df_encoded
        y = None
    return X, y


def get_feature_columns():
    df = load_raw_data()
    X, _ = build_training_features(df)
    return list(X.columns)


def prepare_input(raw_input: dict):
    """Convert a raw (original-format) input dict into a single-row DataFrame
    with the same columns and ordering as the training features.

    raw_input keys should be the original dataset column names (before one-hot).
    """
    df_orig = load_raw_data()
    df_clean = _clean_raw(df_orig.copy())
    # If Attrition_Flag exists, drop it for building a one-row df
    if 'Attrition_Flag' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Attrition_Flag'])

    # Build a template row by taking the first row and replacing values with provided ones
    template = df_clean.iloc[[0]].copy()
    for k, v in raw_input.items():
        if k in template.columns:
            template.at[template.index[0], k] = v

    # One-hot encode same way as training
    categorical_cols = template.select_dtypes(include=['object']).columns
    encoded = pd.get_dummies(template, columns=categorical_cols, drop_first=True, dtype=int)

    # Reindex to training feature columns, filling missing columns with 0
    X_train_cols = get_feature_columns()
    encoded = encoded.reindex(columns=X_train_cols, fill_value=0)
    return encoded


def predict_from_raw(model, raw_input: dict):
    X = prepare_input(raw_input)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    return pred, proba


def get_top_feature_importances(model, n=20):
    cols = get_feature_columns()
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        s = pd.Series(fi, index=cols).sort_values(ascending=False)
        return s.head(n)
    return None
