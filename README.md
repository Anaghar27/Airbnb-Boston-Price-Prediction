# Airbnb Boston Analytics 🏠📊

This project presents a comprehensive data science pipeline that analyzes Airbnb Boston listings to understand pricing behavior, booking trends, and customer sentiment. It integrates machine learning models, sentiment analysis, time-series forecasting into a unified solution using **Streamlit**.

---

## 🔍 Project Objectives

- Predict optimal listing **prices** based on features and amenities
- Classify whether a listing is **likely to be booked**
- Measure the **impact of review sentiment** on listing prices
- Forecast future **listing prices** using time-series models
- Provide an **interactive dashboard** for real-time insights

---

## ✨ Key Features

### 1. 📈 **Price Prediction Model**
- Trained a Random Forest Regressor on cleaned listing features
- Included engineered features like `price_per_guest`, `is_power_host`, `amenities`, etc.
- SHAP analysis reveals which features impact pricing the most

### 2. 📅 **Booking Classifier**
- Classified availability (`is_booked`) using calendar behavior
- Features include weekday/weekend flags, previous booking trend, price signals
- Evaluated across Logistic Regression, Random Forest, XGBoost with accuracy > 72%

### 3. ✍️ **Review Sentiment Integration**
- Applied `TextBlob` to extract sentiment polarity from guest reviews
- Aggregated sentiment per listing and joined with final dataset
- Model retrained to reflect the **impact of review tone on pricing**

### 4. 📉 **Time-Series Forecasting**
- Used Facebook Prophet to forecast listing prices for top listings
- 30-day forecast per listing with upper/lower prediction bounds
- Visualized using Streamlit line charts and downloadable CSVs

### 5. 🧩 **Streamlit Dashboard**
- Unified frontend to interact with all models:
  - Predict price from input sliders
  - Check booking likelihood
  - Analyze review sentiment effect
  - View 30-day forecasts
  - Explore visual insights

## 🗂️ Project Structure

```
.
├── app.py                       # Main Streamlit App
├── models/                      # Pickle files for trained models
├── data/                        # Original CSVs from Kaggle
├── outputs/                     # Processed and predicted data
├── notebooks/                   # Jupyter Notebooks (EDA, training)
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repo  
   `git clone https://github.com/anaghar27/airbnb-boston-dashboard.git`

2. Navigate into project  
   `cd airbnb-boston-dashboard`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Launch the dashboard  
   `streamlit run app.py`

## 📊 Data

- Airbnb Boston listings dataset from Kaggle  
- Merged: `listings.csv`, `calendar.csv`, `reviews.csv`
- Other datsets created for the model training can be saved through the code and further visualized using Tableau

## 📌 Tools Used

- Python, Pandas, Scikit-Learn, XGBoost, SHAP
- Streamlit for interactive dashboard
- Prophet for time-series

## ✨ Authors

- Anagha Raveendra
