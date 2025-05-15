# booking_classifier_model.py
# ---------------------------------------------------------------
# Airbnb Boston - Price Prediction Model
# ---------------------------------------------------------------
# This script trains a machine learning model to classify whether a listing is booked or not
# based on Airbnb calendar data. It includes preprocessing, feature engineering, model training,
# performance evaluation, SHAP explainability, and model persistence.
# ---------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_and_preprocess(filepath: str):
    # Load calendar data, clean missing values, encode booking status, and extract temporal features.
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


def engineer_features(df: pd.DataFrame):
    # Generate engineered features including listing frequency and cyclical transformations of date.
    df.sort_values(by=["listing_id", "date"], inplace=True)

    df["listing_freq"] = df["listing_id"].map(df["listing_id"].value_counts())

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

def get_models():
    # Returns a dictionary of baseline classification models to compare.
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }


def train_models(X_train, X_test, y_train, y_test):
    # Train multiple classifiers and select the best performing model based on accuracy.
    models = get_models()
    results = {}
    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"{name}:\nAccuracy = {acc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model, results


def explain_model(model, X_sample):
    # Visualize feature importance using SHAP values.
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)

    print("SHAP Summary Bar Plot:")
    shap.plots.bar(shap_values)

    print("SHAP Summary Beeswarm Plot:")
    shap.plots.beeswarm(shap_values)


def save_outputs(df: pd.DataFrame, model, output_path: str, model_path: str):
    # Save processed dataset and trained model to disk.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Data saved to: {output_path}")
    print(f"Model saved to: {model_path}")


def main():
    print("Loading and preprocessing data...")
    calendar = load_and_preprocess("../data/calendar.csv")
    calendar = engineer_features(calendar)

    feature_cols = [
        "price_imputed", "day", "weekday", "weekend",
        "listing_freq", "day_sin", "day_cos", "month_sin", "month_cos"
    ]

    X = calendar[feature_cols]
    y = calendar["is_booked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining models...")
    best_model, results = train_models(X_train, X_test, y_train, y_test)

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
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\nExplaining best model using SHAP...")
    X_sample = X_test.sample(n=200, random_state=42)
    explain_model(best_model, X_sample)

    save_outputs(
        calendar,
        best_model,
        "../outputs/booking_classifier_data.csv",
        "../models/booking_classifier_model.pkl"
    )


if __name__ == "__main__":
    main()
