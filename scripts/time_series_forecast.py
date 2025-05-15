# time_series_forecast.py
# ---------------------------------------------------------------
# Airbnb Boston - Time Series Price Forecasting
# ---------------------------------------------------------------
# Forecasts future Airbnb listing prices using Facebook Prophet.
# Processes calendar data, selects top listings, fits models,
# saves individual forecasts and visualizations.
# ---------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def load_and_clean_calendar(path):
    # Loads and preprocesses calendar.csv (price cleaning, date parsing).
    df = pd.read_csv(path)
    df = df.dropna(subset=["price", "date", "listing_id"])
    df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    df = df[df["price"] > 0]
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_top_listings(df, top_n=5):
    # Selects top N listings with the most price records.
    return df["listing_id"].value_counts().head(top_n).index.tolist()


def forecast_listing(df_listing, listing_id, output_dir):
    # Fits Prophet model for one listing and saves CSV and forecast plot.
    df_prophet = df_listing.rename(columns={"date": "ds", "price": "y"})

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast["listing_id"] = listing_id

    # Save forecast and plot
    forecast_path = os.path.join(output_dir, f"forecast_{listing_id}.csv")
    plot_path = os.path.join(output_dir, f"forecast_plot_{listing_id}.png")
    forecast.to_csv(forecast_path, index=False)

    fig = model.plot(forecast)
    plt.title(f"30-Day Price Forecast - Listing ID {listing_id}")
    fig.savefig(plot_path)
    plt.close()

    return forecast[["listing_id", "ds", "yhat", "yhat_lower", "yhat_upper"]]


def save_combined_forecast(all_forecasts, output_dir):
    # Combines all forecasts into one file and saves it.
    combined = pd.concat(all_forecasts, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "forecasted_prices.csv"), index=False)
    print(f"Saved combined forecast to {output_dir}/forecasted_prices.csv")


def main():
    calendar_path = "../data/calendar.csv"
    output_dir = "../outputs/forecast_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and cleaning calendar data...")
    calendar = load_and_clean_calendar(calendar_path)

    print("Selecting top listings...")
    top_listings = get_top_listings(calendar, top_n=5)

    print("Generating forecasts...")
    all_forecasts = []
    for listing_id in top_listings:
        df_listing = calendar[calendar["listing_id"] == listing_id][["date", "price"]]
        forecast = forecast_listing(df_listing, listing_id, output_dir)
        all_forecasts.append(forecast)

    print("Saving final combined forecast...")
    save_combined_forecast(all_forecasts, output_dir)
    print("Forecasting complete.")


if __name__ == "__main__":
    main()
