# Post Cleaning & Preprocessing

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Load raw data
Uber_df = pd.read_csv('data/raw/UBER_posts_1740093505.csv')
print(f"Loaded {len(Uber_df)} UBER Data")

Marta_df = pd.read_csv('data/raw/Marta_posts_1740093357.csv')
print(f"Loaded {len(Marta_df)} Marta Data")

AtlantaBeltline_df = pd.read_csv('data/raw/ATL_Beltline_posts_1740699373.csv')
print(f"Loaded {len(AtlantaBeltline_df)} AtlBeltline Data")


# Define a post cleaning function
def clean_tweet(df):
    df = df.drop(
        columns=["PostId", "PostUrl", "PostEngagement", "ChannelID", "Platform", "ChannelUid", "ChannelUrl",
                 "ChannelEngagement", "GoogleAudioText", "post_data", "post_media_urls", "post_media_file",
                 "embedded_post_text", "search_data", "EmbeddedContentText", "VoskAudioText", "EmbeddedContentText"])

    return df

def is_safety_related(post):
    safety_keywords = [
        "accident", "fire", "explosion", "shooting", "evacuate", "emergency",
        "danger", "hazard", "injury", "ambulance", "violence", "earthquake", "crime"
    ]

    post = str(post).lower()
    return any(keyword in post for keyword in safety_keywords)


# Apply the cleaning function
Uber_df = clean_tweet(Uber_df)
Marta_df = clean_tweet(Marta_df)
AtlantaBeltline_df = clean_tweet(AtlantaBeltline_df)

# Narrow to safety-issue post
Uber_df['is_safety_related'] = Uber_df['post_body_text'].apply(is_safety_related)
Uber_df = Uber_df[Uber_df['is_safety_related'] == True]

Marta_df['is_safety_related'] = Marta_df['post_body_text'].apply(is_safety_related)
Marta_df = Marta_df[Marta_df['is_safety_related'] == True]

AtlantaBeltline_df['is_safety_related'] = AtlantaBeltline_df['post_body_text'].apply(is_safety_related)
AtlantaBeltline_df = AtlantaBeltline_df[AtlantaBeltline_df['is_safety_related'] == True]

# Save the cleaned dataset
Uber_df.to_csv('data/processed/UBER_Safety_Post', index=False)
Marta_df.to_csv('data/processed/Marta_Safety_Post', index=False)
AtlantaBeltline_df.to_csv('data/processed/AtlBeltline_Safety_Post', index=False)
print("✅ Cleaned post saved to data/processed/")
