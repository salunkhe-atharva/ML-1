import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Diabetes Prediction - MSE Model", layout="wide")

st.title("📊 Diabetes Progression Prediction using Linear Regression")


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Squared Error", f"{mse:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")

st.divider()


st.subheader("🔵 True vs Predicted Values")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.5)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
)
ax1.set_xlabel("True Values")
ax1.set_ylabel("Predicted Values")
ax1.set_title("True vs Predicted")
ax1.grid(True)

st.pyplot(fig1)


st.subheader("🟢 BMI vs Predicted Progression")

fig2, ax2 = plt.subplots()
ax2.scatter(X_test[:, 2], y_pred, alpha=0.7)
ax2.set_xlabel("BMI Feature")
ax2.set_ylabel("Predicted Diabetes Progression")
ax2.set_title("BMI vs Prediction")
ax2.grid(True)

st.pyplot(fig2)
