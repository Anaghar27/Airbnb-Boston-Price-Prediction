# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import shap
from textblob import TextBlob
from prophet.plot import plot_plotly
from prophet import Prophet
import os

# ---------------------------
# Load Models & Data
# ---------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    try:
        with open("models/price_prediciton_model.pkl", "rb") as f:
            models['base_price'] = pickle.load(f)
        with open("models/final_price_prediction_model.pkl", "rb") as f:
            models['final_price'] = pickle.load(f)
        with open("models/booking_prediction_model.pkl", "rb") as f:
            models['booking'] = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
    return models

@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    listings = pd.read_csv("data/listings.csv")
    calendar = pd.read_csv("data/calendar.csv")
    reviews = pd.read_csv("data/reviews.csv")
    final_df = pd.read_csv("outputs/final_price_prediction_data.csv")
    forecast_df = pd.read_csv("outputs/forecasted_prices.csv")
    return listings, calendar, reviews, final_df, forecast_df

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Airbnb Boston Dashboard", layout="wide")
st.title("Airbnb Boston Analytics Dashboard")

models = load_models()
listings, calendar, reviews, final_df, forecast_df = load_data()

page = st.sidebar.radio("Select Feature", [
    "Price Prediction",
    "Booking Classifier",
    "Review Sentiment Impact",
    "Time Series Forecast",
    "Visual Insights"
])

# ---------------------------
# PRICE PREDICTION
# ---------------------------
if page == "Price Prediction":
    st.header("📈 Price Prediction")
    col1, col2 = st.columns(2)

    guests = col1.slider("Number of Guests", 1, 10, 2, help="Total number of guests expected")
    wifi = col2.checkbox("Has Wireless Internet", True)
    kitchen = col1.checkbox("Has Kitchen", True)
    tv = col2.checkbox("Has TV", True)

    heating = col1.checkbox("Has Heating", True)
    essentials = col2.checkbox("Has Essentials", True)
    smoke_detector = col1.checkbox("Has Smoke Detector", True)
    air_conditioning = col2.checkbox("Has Air Conditioning", True)
    internet = col1.checkbox("Has Internet", True)
    washer = col2.checkbox("Has Washer", True)
    dryer = col1.checkbox("Has Dryer", True)

    input_data = pd.DataFrame({
        "price_per_person": [150 / guests],
        "has_wireless_internet": [int(wifi)],
        "has_heating": [int(heating)],
        "has_kitchen": [int(kitchen)],
        "has_essentials": [int(essentials)],
        "has_smoke_detector": [int(smoke_detector)],
        "has_air_conditioning": [int(air_conditioning)],
        "has_tv": [int(tv)],
        "has_internet": [int(internet)],
        "has_washer": [int(washer)],
        "has_dryer": [int(dryer)],
        "host_total_listings_count": [1],
        "host_listings_count": [1],
        "calculated_host_listings_count": [1],
        "guests_included": [guests],
        "availability_30": [60],
        "availability_60": [45],
        "availability_90": [30],
        "price_per_guest": [150 / guests],
        "avg_short_term_availability": [45],
        "is_power_host": [0]
    })


    if st.button("Predict Price"):
        pred_log = models['base_price'].predict(input_data)[0]
        pred_price = np.expm1(pred_log)
        st.success(f"Predicted Price: ${pred_price:.2f}")

        background_data = final_df.drop(columns=["id", "price", "log_price", "listing_id"], errors='ignore')
        background_sample = background_data.sample(n=100, random_state=42)

        explainer = shap.Explainer(models['base_price'], background_sample)
        shap_vals = explainer.shap_values(input_data, check_additivity=False)
        
        st.subheader("Feature Importance (SHAP)")
        shap.summary_plot(shap_vals, input_data, plot_type="bar", show=False)
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

# ---------------------------
# BOOKING CLASSIFIER
# ---------------------------
elif page == "Booking Classifier":
    st.header("📅 Booking Availability Classifier")

    day = st.slider("Day of Month", 1, 31, 15)
    weekday = st.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))
    month = st.selectbox("Month", list(range(1, 13)))
    price = st.number_input("Listing Price", 50.0, 1000.0, 150.0)
    avg_freq = calendar["listing_id"].value_counts().mean()

    from math import pi
    data = pd.DataFrame({
        "price_imputed": [price],
        "weekday": [weekday],
        "weekend": [int(weekday in [5, 6])],
        "listing_freq": [avg_freq],
        "day_sin": [np.sin(2 * pi * day / 31)],
        "day_cos": [np.cos(2 * pi * day / 31)],
        "month_sin": [np.sin(2 * pi * month / 12)],
        "month_cos": [np.cos(2 * pi * month / 12)],
    })

    if st.button("Predict Booking Status"):
        pred = models['booking'].predict(data)[0]
        label = "✅ Likely Booked" if pred else "❌ Likely Not Booked"
        st.success(label)

# ---------------------------
# REVIEW NLP IMPACT
# ---------------------------
elif page == "Review Sentiment Impact":
    st.header("✍️ Review Sentiment Impact on Price")
    review = st.text_area("Enter a Review:", "The place was clean and cozy with amazing service.")
    if st.button("Analyze Sentiment"):
        polarity = TextBlob(review).sentiment.polarity
        st.info(f"Sentiment Polarity Score: {polarity:.2f}")
        test_data = final_df.drop(columns=['id', 'price', 'log_price', 'listing_id'], errors='ignore').iloc[[0]].copy()
        test_data["avg_sentiment"] = polarity
        pred_log = models['final_price'].predict(test_data)[0]
        st.success(f"Updated Price Prediction: ${np.expm1(pred_log):.2f}")

# ---------------------------
# TIME SERIES FORECAST
# ---------------------------
elif page == "Time Series Forecast":
    st.header("📉 30-Day Price Forecast per Listing")
    ids = forecast_df["listing_id"].unique().tolist()
    selected_id = st.selectbox("Select Listing ID", ids)
    fcast = forecast_df[forecast_df["listing_id"] == selected_id]
    st.line_chart(fcast.set_index("ds")["yhat"])
    st.dataframe(fcast.tail(10))

    csv = fcast.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name=f"forecast_{selected_id}.csv")

# ---------------------------
# VISUAL INSIGHTS
# ---------------------------
elif page == "Visual Insights":
    st.header("📊 Dataset Visualizations")
    
    st.subheader("Distribution of Airbnb Prices in Boston")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(listings["price"].dropna(), bins=50, color="skyblue", ax=ax)
    ax.set_title("Distribution of Airbnb Prices in Boston", fontsize=14)
    ax.set_xlabel("Price ($)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.tick_params(axis='x', labelrotation=45)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Limit number of ticks
    st.pyplot(fig)

    st.subheader("Correlation Between Numerical Features")
    num_cols = listings.select_dtypes(include=np.number).dropna(axis=1).columns
    corr_matrix = listings[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                cbar_kws={"shrink": 0.7}, linewidths=0.5, square=True)
    ax.set_title("Correlation Between Numerical Features", fontsize=14)
    st.pyplot(fig)

