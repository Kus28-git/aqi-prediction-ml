# AQI Prediction Using Machine Learning

## Project Overview

This project predicts Air Quality Index (AQI) using Machine Learning techniques.

The system takes air pollutant values as input and predicts the AQI along with the air quality category.

---

## Models Used

1. Linear Regression
2. Logistic Regression
3. Random Forest Regressor

After evaluation, Random Forest was selected as the final model because it provided the most reliable AQI predictions.

---

## Dataset

The project uses AQI data for the year 2024.

Features used:

* PM2.5
* PM10
* NO2
* SO2
* CO

Target:

* AQI

Category Labels:

* Good
* Satisfactory
* Moderate
* Poor
* Very Poor
* Severe

---

## Data Preprocessing

* Missing value handling
* Date and timestamp processing
* Feature selection
* AQI conversion to numeric format

---

## Model Evaluation

Evaluation Metrics:

* Mean Squared Error (MSE)
* R² Score
* Accuracy (for Logistic Regression)

The models were compared and Random Forest achieved the best overall performance.

---

## Application Features

* Model Comparison Table
* Model Evaluation Graph
* AQI Prediction using Random Forest
* Actual vs Predicted AQI Graph
* Monthly Average AQI Graph (2024)

---

## Deployment

The application is deployed using Streamlit Community Cloud.

Hosted Application:

https://aqi-prediction-ml-kushi.streamlit.app

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Author

Kushi Prasad
AIML Mini Project
