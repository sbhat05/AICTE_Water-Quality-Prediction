# AICTE_Water-Quality-Prediction
This project aims to predict multiple water quality parameters across 22 monitoring stations using various machine learning techniques.

## Overview
As access to clean water is necessary, prediction of various water quality metrics will help in early detection of water pollution and ensure timely intervention.
In this project, we:
- Collected and preprocessed real-world water quality datasets
- Filled missing values with station-wise medians
- Engineered new features (e.g., cyclical month encoding)
- Explored multiple advanced machine learning models for multi-target regression
- Compared model performance using test set R² scores and MSE
- Selected the best model for deployment

## Dataset Used
Southern Bug River water quality dataset (measurements from 22 monitoring stations over time)

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn
- CatBoost, LightGBM, XGBoost
- Jupyter Notebook
- Streamlit (for deployment)

## Predicted water quality parameters
- NH4
- BOD5 (BSK5)
- Colloids (Suspended)
- O2, NO3, NO2, SO4, PO4 and Cl

## Models Explored
MultiOutputRegressor wrapped around:
- RandomForestRegressor
- CatBoostRegressor
- HistGradientBoostingRegressor
- LightGBMRegressor
- XGBoostRegressor
ExtraTreesRegressor
Ensemble model: average of:
MultiOutputRegresor wrapped around CatBoostRegressor + MultiOutputRegressor wrapped around HistGradientBoostingRegressor

## Model Performance
Model was evaluated using
- Mean Squared Error (MSE)
- R2 Score

## Deployment
The final model was deployed using Streamlit, providing an interactive interface:
- Select station ID
- Enter year & month
- Instantly view predicted pollutant levels

## Repository Contents
- WaterQualityPrediction.ipynb : Data preprocessing, feature engineering, model training & evaluation
- app.py : Streamlit app for deployment
- Saved model files: model_cb.pkl, model_hgb.pkl
- Saved features list: model_columns.pkl
- Pollutants list: pollutants_list.pkl

