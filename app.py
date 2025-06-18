import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set Streamlit page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("House Price Prediction in INR 🇮🇳")

# Sidebar inputs
st.sidebar.header("Enter House Details")
sqft = st.sidebar.number_input("Square Footage", min_value=200, max_value=10000, step=100)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025)

# Add Apply button
if st.sidebar.button("Apply"):

    # Sample dataset (price in USD)
    data = {
        "sqft": [1500, 2000, 2500, 1800, 2200, 3000, 2800],
        "bedrooms": [3, 4, 3, 3, 4, 5, 4],
        "bathrooms": [2, 3, 2, 2, 3, 4, 3],
        "year_built": [1995, 2000, 2005, 2010, 2015, 2020, 2018],
        "price": [400000, 500000, 600000, 450000, 520000, 750000, 650000]
    }
    df = pd.DataFrame(data)

    # Features and target
    X = df[["sqft", "bedrooms", "bathrooms", "year_built"]]
    y = df["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict price
    input_data = np.array([[sqft, bedrooms, bathrooms, year_built]])
    predicted_usd = model.predict(input_data)[0]

    # Convert to INR
    usd_to_inr = 83.0
    predicted_inr = predicted_usd * usd_to_inr

    # Display result
    st.subheader("Estimated House Price")
    st.success(f"₹{predicted_inr:,.2f} (approx)")

    # Show MSE for reference
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.caption(f"Model Mean Squared Error (test set): {mse:,.2f} USD²")

#python -m streamlit run app.py

