import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.stats import pearsonr
import argparse
import os

# Function to compute sentiment polarity
def sentiment_score(text):
    """Return sentiment polarity using TextBlob."""
    return TextBlob(str(text)).sentiment.polarity

def main(stock_path, news_path, out_dir):
    # Load datasets
    stock = pd.read_csv(stock_path)
    news = pd.read_csv(news_path)

    # Ensure date formatting
    stock['Date'] = pd.to_datetime(stock['Date'])
    news['Date'] = pd.to_datetime(news['Date'])

    # Compute daily returns
    stock['Daily_Return'] = stock['Close'].pct_change()

    # Sentiment analysis
    news['Sentiment'] = news['Headline'].apply(sentiment_score)

    # Aggregate sentiment by day
    daily_sent = news.groupby('Date')['Sentiment'].mean().reset_index()

    # Merge with stock daily returns
    merged = pd.merge(stock, daily_sent, on='Date', how='inner')

    # Drop NaN values before correlation
    merged_clean = merged.dropna(subset=['Daily_Return', 'Sentiment'])

    # Calculate correlation
    corr, p_value = pearsonr(merged_clean['Sentiment'], merged_clean['Daily_Return'])
    print(f"Pearson Correlation: {corr}")
    print(f"P-value: {p_value}")

    # Plot correlation
    plt.figure(figsize=(8,6))
    plt.scatter(merged_clean['Sentiment'], merged_clean['Daily_Return'])
    plt.xlabel("Daily Sentiment")
    plt.ylabel("Daily Stock Return")
    plt.title("Correlation Between News Sentiment and Stock Movement")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/sentiment_vs_return.png")
    print(f"Saved plot to {out_dir}/sentiment_vs_return.png")

    # Save merged data
    merged_clean.to_csv(f"{out_dir}/merged_output.csv", index=False)
    print(f"Saved merged file to {out_dir}/merged_output.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, help="Path to stock CSV")
    parser.add_argument("--news", required=True, help="Path to news CSV")
    parser.add_argument("--out", default="task3_output", help="Output directory")
    args = parser.parse_args()
    main(args.stock, args.news, args.out)
