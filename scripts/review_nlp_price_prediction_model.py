# review_nlp_price_prediction_model.py
# ---------------------------------------------------------------
# Airbnb Boston - Price Prediction Model
# ---------------------------------------------------------------
# This module performs sentiment-based enhancement of Airbnb price prediction.
# It uses TextBlob to extract sentiment scores from user reviews, aggregates them per listing,
# merges them with existing features, and evaluates their impact on Random Forest regression.

# ---------------------------------------------------------------

import os
import pickle
import numpy as np
import pandas as pd
import shap
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Constants for reusability
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
MODEL_DIR = "../models"

def load_and_process_reviews(review_path: str) -> pd.DataFrame:
    # Load reviews and compute sentiment polarity.
    reviews = pd.read_csv(review_path)
    reviews = reviews.dropna(subset=["comments"])
    reviews = reviews[reviews["comments"].str.strip() != ""]
    reviews["sentiment"] = reviews["comments"].apply(lambda x: TextBlob(x).sentiment.polarity)
    return reviews

def aggregate_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    # Aggregate sentiment statistics per listing_id.
    agg = reviews.groupby("listing_id").agg({"sentiment": ["mean", "count"]})
    agg.columns = ["avg_sentiment", "review_count"]
    agg.reset_index(inplace=True)
    return agg

def merge_with_listings(enhanced_path: str, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    # Merge aggregated sentiment with enhanced listing features.
    enhanced_df = pd.read_csv(enhanced_path)
    if 'id' not in enhanced_df.columns:
        listings_df = pd.read_csv(os.path.join(DATA_DIR, "listings.csv"))
        enhanced_df = listings_df[['id']].merge(enhanced_df, left_index=True, right_index=True)
    df_merged = enhanced_df.merge(sentiment_df, left_on="id", right_on="listing_id", how="left")
    df_merged["avg_sentiment"] = df_merged["avg_sentiment"].fillna(0)
    df_merged["review_count"] = df_merged["review_count"].fillna(0)
    return df_merged

def train_price_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    # Train Random Forest on enhanced features including sentiment.
    X = df.drop(columns=["id", "price", "log_price", "listing_id"], errors='ignore')
    y = np.log1p(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred_price = np.expm1(y_pred_log)
    y_true_price = np.expm1(y_test)

    rmse = mean_squared_error(y_true_price, y_pred_price, squared=False)
    print(f"RMSE after including review sentiment: ${rmse:.2f}")

    return model, X_test

def run_shap(model: RandomForestRegressor, X_test: pd.DataFrame, save: bool = False):
    # Visualize feature importance using SHAP.
    X_sample = X_test.sample(n=200, random_state=42)
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)

    shap.plots.bar(shap_values, show=not save)
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"))

    shap.plots.beeswarm(shap_values, show=not save)
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"))

def save_outputs(df: pd.DataFrame, model: RandomForestRegressor):
    # Save the enhanced dataset and trained model.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "final_price_prediction_data.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "final_price_prediction_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("Saved enhanced dataset and model.")

def load_review_sentiment_model(model_path: str = os.path.join(MODEL_DIR, "final_price_prediction_model.pkl")) -> RandomForestRegressor:
    # Load trained sentiment-aware price prediction model.
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_price_with_review_sentiment(model: RandomForestRegressor, features_df: pd.DataFrame) -> np.ndarray:
    # Make predictions from input features and return de-logged prices.
    return np.expm1(model.predict(features_df))

def main():
    reviews = load_and_process_reviews(os.path.join(DATA_DIR, "reviews.csv"))
    sentiment_df = aggregate_sentiment(reviews)
    df_nlp = merge_with_listings(os.path.join(OUTPUT_DIR, "price_prediction.csv"), sentiment_df)
    model, X_test = train_price_model(df_nlp)
    run_shap(model, X_test)
    save_outputs(df_nlp, model)

if __name__ == "__main__":
    main()