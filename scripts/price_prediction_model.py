# booking_classifier_model.py
# ---------------------------------------------------------------
# Airbnb Boston - Booking Classifier
# ---------------------------------------------------------------
# This script trains a model to classify booking availability
# using Airbnb calendar data. Includes preprocessing, modeling,
# SHAP explanation, and model export.
# ---------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def get_models():
    # Returns a dictionary of baseline classification models to compare.
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }


def load_and_preprocess(filepath):
    # Loads calendar.csv and performs initial preprocessing on availability and price.
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["available"])
    df = df[df["available"].isin(["t", "f"])]
    df["is_booked"] = df["available"].apply(lambda x: 0 if x == "t" else 1)

    df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    df["median_price"] = df.groupby("listing_id")["price"].transform("median")
    df["price_imputed"] = df["price"].fillna(df["median_price"])
    df["price_imputed"] = df["price_imputed"].fillna(df["price"].median())

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["weekend"] = df["weekday"].isin([5, 6]).astype(int)

    return df


def engineer_features(df):
    # Adds time-based and frequency-based engineered features to the calendar data.
    df.sort_values(by=["listing_id", "date"], inplace=True)
    df["listing_freq"] = df["listing_id"].map(df["listing_id"].value_counts())
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Trains and compares multiple classifiers; returns the best model and scores.
    results = {}
    best_model = None
    best_score = 0
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"{name}:\nAccuracy = {acc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

        if acc > best_score:
            best_score = acc
            best_model = model

    return best_model, results


def explain_model_with_shap(model, X_sample):
    # Generates SHAP summary plots to interpret the model predictions.
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)


def save_model_and_data(df, model, data_path, model_path):
    # Saves the trained model and processed dataset to the disk.
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df.to_csv(data_path, index=False)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Data saved to: {data_path}")
    print(f"Model saved to: {model_path}")


def main():
    print("Loading and preprocessing calendar data...")
    df = load_and_preprocess("../data/calendar.csv")
    df = engineer_features(df)

    features = [
        "price_imputed", "day", "weekday", "weekend",
        "listing_freq", "day_sin", "day_cos", "month_sin", "month_cos"
    ]
    X = df[features]
    y = df["is_booked"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining and comparing models...")
    best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Performance Summary")
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 40)
    for name, acc in results.items():
        print(f"{name:<25} {acc:>10.4f}")

    plt.figure(figsize=(8, 5))
    plt.barh(list(results.keys()), list(results.values()), color='skyblue')
    plt.xlabel("Accuracy")
    plt.title("Model Comparison: Booking Classifier")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("\nRunning SHAP for explainability...")
    explain_model_with_shap(best_model, X_test.sample(200, random_state=42))

    save_model_and_data(
        df, best_model,
        "../outputs/booking_classifier_data.csv",
        "../models/booking_classifier_model.pkl"
    )


if __name__ == "__main__":
    main()
