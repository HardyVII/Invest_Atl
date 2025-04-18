import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Config
input_paths = {
    'UBER': 'data/sentiment/UBER_Sentiment.csv',
    'MARTA': 'data/sentiment/Marta_Sentiment.csv',
    'Beltline': 'data/sentiment/AtlBeltline_Sentiment.csv'
}

output_dir = "outputs/trends"
os.makedirs(output_dir, exist_ok=True)

# Helper function to process and visualize sentiment trends
def plot_sentiment_trend(company, path):
    df = pd.read_csv(path)

    # Convert timestamp
    if 'published_at' not in df.columns:
        print(f"⏳ Skipped {company}: No 'published_at' column found")
        return

    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at'])
    df['date'] = df['published_at'].dt.to_period('M').dt.to_timestamp()

    # Group and count sentiments per month
    sentiment_trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)

    # Plot
    sentiment_trend.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title(f"Sentiment Trend Over Time - {company}")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="Sentiment")
    output_file = f"{output_dir}/{company}_sentiment_trend.png"
    plt.savefig(output_file)
    plt.show()
    print(f"📈 Saved: {output_file}")

# Run for each organization
for org, path in input_paths.items():
    plot_sentiment_trend(org, path)
