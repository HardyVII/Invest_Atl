# Sentiment Scoring Using VADER

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define function to apply sentiment scoring
def apply_sentiment(df):
    df['compound_score'] = df['post_body_text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

    # Assign sentiment label based on compound score
    def label_sentiment(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df['compound_score'].apply(label_sentiment)
    return df

# Load safety-related posts
uber_df = pd.read_csv('data/processed/UBER_Safety_Post.csv')
marta_df = pd.read_csv('data/processed/Marta_Safety_Post.csv')
beltline_df = pd.read_csv('data/processed/AtlBeltline_Safety_Post.csv')

# Apply sentiment analysis
uber_df = apply_sentiment(uber_df)
marta_df = apply_sentiment(marta_df)
beltline_df = apply_sentiment(beltline_df)

# Save outputs
uber_df.to_csv('data/sentiment/UBER_Sentiment.csv', index=False)
marta_df.to_csv('data/sentiment/Marta_Sentiment.csv', index=False)
beltline_df.to_csv('data/sentiment/AtlBeltline_Sentiment.csv', index=False)

print("✅ Sentiment analysis complete. Files saved in data/sentiment/")
