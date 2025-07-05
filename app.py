import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and config
model_cb = joblib.load('model_cb.pkl')
model_hgb = joblib.load('model_hgb.pkl')
feature_columns = joblib.load('model_columns.pkl')
pollutants = joblib.load('pollutants_list.pkl')

st.set_page_config(page_title="Water Pollutants Predictor", page_icon="💧")

st.title("Water Pollutants Predictor")
st.markdown("Predict water pollutant levels based on month, year and station ID")

# User inputs
station_ids = [str(i) for i in range(1, 23)]
station_id = st.selectbox("Select Station ID", options=station_ids)
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2024)
month_input = st.slider("Select Month", min_value=1, max_value=12, value=3)

# Predict button
if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'year': [year_input],
        'month': [month_input],
        'id': [station_id]
    })

    # Feature engineering
    input_data['year_squared'] = input_data['year'] ** 2
    input_data['month_sin'] = np.sin(2 * np.pi * input_data['month'] / 12)
    input_data['month_cos'] = np.cos(2 * np.pi * input_data['month'] / 12)

    # One-hot encode station_id
    input_encoded = pd.get_dummies(input_data, columns=['id']).astype(int)

    # Align with training columns
    missing_cols = set(feature_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    # Predict using both models
    pred_cb = model_cb.predict(input_encoded)[0]
    pred_hgb = model_hgb.predict(input_encoded)[0]
    ensemble_pred = (pred_cb + pred_hgb) / 2
    ensemble_pred = np.maximum(0, ensemble_pred)

    # Show results
    st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_input}-{month_input:02d}:")
    results_df = pd.DataFrame({
        'Pollutant': pollutants,
        'Predicted Level': [f"{val:.2f}" for val in ensemble_pred]
    })
    st.dataframe(results_df, hide_index=True)
