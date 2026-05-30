# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score
)

# LOAD DATASET(2024)
df = pd.read_csv("AQI_2024.csv")

df = df[
    [
        'timestamp',
        'pm2.5',
        'pm10',
        'no2',
        'so2',
        'co',
        'AQI',
        'Category'
    ]
]

df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='%d-%m-%Y %H:%M',
    errors='coerce'
)

df['year'] = df['timestamp'].dt.year

df = df.fillna(df.mean(numeric_only=True))

df = df.dropna()

df['month'] = df['timestamp'].dt.month

# DEFINE INPUT FEATURES
X = df[['pm2.5', 'pm10', 'no2', 'so2', 'co']]

y_reg = df['AQI']

y_class = df['Category']

# TRAIN TEST SPLIT
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

X_train2, X_test2, y_train_class, y_test_class = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

# LINEAR REGRESSION
linear_model = LinearRegression()

linear_model.fit(X_train, y_train_reg)

linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(
    y_test_reg,
    linear_pred
)

linear_r2 = r2_score(
    y_test_reg,
    linear_pred
)

sample_input = [[350, 500, 150, 120, 8]]

linear_sample_pred = linear_model.predict(
    sample_input
)[0]

# LOGISTIC REGRESSION
logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train2,
    y_train_class
)

logistic_pred = logistic_model.predict(
    X_test2
)

logistic_acc = accuracy_score(
    y_test_class,
    logistic_pred
)

logistic_sample_pred = logistic_model.predict(
    sample_input
)[0]

# RANDOM FOREST REGRESSOR
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train_reg
)

rf_pred = rf_model.predict(
    X_test
)

rf_mse = mean_squared_error(
    y_test_reg,
    rf_pred
)

rf_r2 = r2_score(
    y_test_reg,
    rf_pred
)

rf_sample_pred = rf_model.predict(
    sample_input
)[0]

# STREAMLIT TITLE
st.title("AQI Prediction Using Machine Learning")

st.write(
    "Comparison of Linear Regression, Logistic Regression and Random Forest"
)

# MODEL COMPARISON TABLE
st.subheader("Model Performance Table")

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Logistic Regression",
        "Random Forest"
    ],

    "Predicted Value": [
        round(linear_sample_pred, 2),
        logistic_sample_pred,
        round(rf_sample_pred, 2)
    ],

    "MSE": [
        round(linear_mse, 2),
        "Classification Model",
        round(rf_mse, 2)
    ],

    "R2 Score": [
        round(linear_r2, 2),
        "Accuracy Used",
        round(rf_r2, 2)
    ],

    "Accuracy": [
        "-",
        round(logistic_acc, 2),
        "-"
    ],

    "Performance (%)": [
        round(linear_r2 * 100, 2),
        round(logistic_acc * 100, 2),
        round(rf_r2 * 100, 2)
    ]

})

st.dataframe(comparison)

# GRAPH 1 : MODEL Evalutaion COMPARISON
st.subheader("Model Evaluation Comparison")

models = [
    "Linear Regression",
    "Logistic Regression",
    "Random Forest"
]

performance_scores = [
    linear_r2 * 100,
    logistic_acc * 100,
    rf_r2 * 100
]

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.bar(
    models,
    performance_scores
)

ax1.set_title(
    "Model Performance Comparison"
)

ax1.set_ylabel(
    "Performance (%)"
)

st.pyplot(fig1)

# CONCLUSION
st.subheader("Conclusion")

st.success("""
After comparing Linear Regression,
Logistic Regression and Random Forest:

• Random Forest achieved the best performance.
• It produced lower prediction error.
• It handled complex AQI patterns effectively.
• It provided more reliable AQI predictions.

Therefore, Random Forest was selected
for final AQI prediction and deployment.
""")

# AQI PREDICTION USING RANDOM FOREST
st.subheader(
    "AQI Prediction Using Random Forest"
)

pm25 = st.number_input(
    "Enter PM2.5"
)

pm10 = st.number_input(
    "Enter PM10"
)

no2 = st.number_input(
    "Enter NO2"
)

so2 = st.number_input(
    "Enter SO2"
)

co = st.number_input(
    "Enter CO"
)

if st.button("Predict AQI"):

    user_input = [
        [
            pm25,
            pm10,
            no2,
            so2,
            co
        ]
    ]

    predicted_aqi = rf_model.predict(
        user_input
    )[0]

    st.success(
        f"Predicted AQI = {predicted_aqi:.2f}"
    )

    if predicted_aqi <= 50:
        category = "Good"

    elif predicted_aqi <= 100:
        category = "Satisfactory"

    elif predicted_aqi <= 200:
        category = "Moderate"

    elif predicted_aqi <= 300:
        category = "Poor"

    elif predicted_aqi <= 400:
        category = "Very Poor"

    else:
        category = "Severe"

    st.info(
        f"Air Quality Category = {category}"
    )

# GRAPH 2 : ACTUAL VS PREDICTED AQI
st.subheader(
    "Actual vs Predicted AQI"
)

fig2, ax2 = plt.subplots(
    figsize=(10, 6)
)

actual_values = y_test_reg.values[:50]
predicted_values = rf_pred[:50]

ax2.plot(
    actual_values,
    label="Actual AQI",
    marker='o'
)

ax2.plot(
    predicted_values,
    label="Predicted AQI",
    marker='x'
)

ax2.set_xlabel(
    "Test Data Records"
)

ax2.set_ylabel(
    "AQI"
)

ax2.set_title(
    "Actual vs Predicted AQI"
)

ax2.legend()

ax2.grid(True)

st.pyplot(fig2)

# GRAPH 3 : MONTHLY AVERAGE AQI - 2024
st.subheader(
    "Monthly Average AQI - 2024"
)

monthly_avg_aqi = (
    df.groupby('month')['AQI']
    .mean()
)

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)

ax3.plot(
    month_names[:len(monthly_avg_aqi)],
    monthly_avg_aqi.values,
    marker='o'
)

ax3.set_xlabel(
    "Month"
)

ax3.set_ylabel(
    "Average AQI"
)

ax3.set_title(
    "Monthly Average AQI - 2024"
)

ax3.grid(True)

st.pyplot(fig3)