import streamlit as st
import pandas as pd
import numpy as np
import model_utils


st.set_page_config(page_title='Bank Churn Predictor', layout='wide')

@st.cache_data
def load_model():
    return model_utils.load_model()


@st.cache_data
def load_raw():
    return model_utils.load_raw_data()


model = load_model()
raw_df = load_raw()

st.title('Bank Customer Churn — Predict & Explore')
st.markdown('''A small, intuitive web UI to predict customer churn using a trained Random Forest model.
Use the left sidebar to pick input mode and provide values.''')

with st.sidebar:
    st.header('Input Mode')
    mode = st.radio('Mode', ['Single prediction', 'Batch CSV'])
    st.markdown('---')
    st.write('Model file:`bank_churn_model.joblib`')

if mode == 'Single prediction':
    st.subheader('Single customer prediction')

    # Build form from original columns (before one-hot)
    df_clean = model_utils._clean_raw(raw_df.copy())
    if 'Attrition_Flag' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Attrition_Flag'])

    sample_choice = st.selectbox('Auto-fill from sample row (optional)',
                                 options=['(none)'] + [f'Row {i}' for i in range(len(df_clean))])
    if sample_choice != '(none)':
        idx = int(sample_choice.split()[1])
        sample_row = df_clean.iloc[idx].to_dict()
    else:
        sample_row = {}

    with st.form('input_form'):
        cols = st.columns(2)
        inputs = {}
        for i, col in enumerate(df_clean.columns):
            widget_col = cols[i % 2]
            val = sample_row.get(col, None)
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                mn = float(np.nanmin(df_clean[col]))
                mx = float(np.nanmax(df_clean[col]))
                mean = float(np.nanmean(df_clean[col]))
                step = (mx - mn) / 100 if mx != mn else 1.0
                inputs[col] = widget_col.number_input(col, value=float(val) if val is not None else mean, min_value=mn, max_value=mx, step=step)
            else:
                opts = list(df_clean[col].dropna().unique())
                if val is None:
                    inputs[col] = widget_col.selectbox(col, options=opts)
                else:
                    inputs[col] = widget_col.selectbox(col, options=opts, index=opts.index(val) if val in opts else 0)

        submitted = st.form_submit_button('Predict')

    if submitted:
        with st.spinner('Preparing input and predicting...'):
            pred, proba = model_utils.predict_from_raw(model, inputs)

        st.metric('Predicted class', 'Churn' if pred == 1 else 'No churn')
        if proba is not None:
            st.write(f'Probability (No churn / Churn): {proba[0]:.3f} / {proba[1]:.3f}')

        st.subheader('Top model feature importances (global)')
        fi = model_utils.get_top_feature_importances(model, n=15)
        if fi is not None:
            st.bar_chart(fi)
        else:
            st.write('Feature importances not available for this model.')

elif mode == 'Batch CSV':
    st.subheader('Batch predictions via CSV upload')
    uploaded = st.file_uploader('Upload CSV file with the original dataset columns', type=['csv'])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write('Preview of uploaded data:')
        st.dataframe(df_up.head())

        if st.button('Run batch predictions'):
            with st.spinner('Processing...'):
                X_train_cols = model_utils.get_feature_columns()
                # Build encoded dataframe matching training columns
                # Clean uploaded data similarly
                df_proc = model_utils._clean_raw(df_up.copy())
                if 'Attrition_Flag' in df_proc.columns:
                    df_proc = df_proc.drop(columns=['Attrition_Flag'])
                df_enc = pd.get_dummies(df_proc, drop_first=True, dtype=int)
                df_enc = df_enc.reindex(columns=X_train_cols, fill_value=0)
                preds = model.predict(df_enc)
                proba = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df_enc)

                results = df_up.copy()
                results['pred_churn'] = preds
                if proba is not None:
                    results['prob_no_churn'] = proba[:, 0]
                    results['prob_churn'] = proba[:, 1]

                st.write('Predictions:')
                st.dataframe(results.head())

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button('Download results CSV', data=csv, file_name='predictions.csv', mime='text/csv')

st.markdown('---')
st.caption('App built with Streamlit — fills inputs using the original training schema.')
