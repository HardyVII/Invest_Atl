# Post Cleaning & Preprocessing

import pandas as pd
import re
from rapidfuzz import fuzz
import nltk

# Load raw data
Uber_df = pd.read_csv('data/raw/UBER_posts_1740093505.csv')
print(f"Loaded {len(Uber_df)} UBER Data")

Marta_df = pd.read_csv('data/raw/Marta_posts_1740093357.csv')
print(f"Loaded {len(Marta_df)} Marta Data")

AtlantaBeltline_df = pd.read_csv('data/raw/ATL_Beltline_posts_1740699373.csv')
print(f"Loaded {len(AtlantaBeltline_df)} AtlBeltline Data")

# Step 1: Clean columns
def clean_tweet(df):
    return df.drop(
        columns=[
            "PostId", "PostUrl", "PostEngagement", "ChannelID", "Platform", "ChannelUid", "ChannelUrl",
            "ChannelEngagement", "GoogleAudioText", "post_data", "post_media_urls", "post_media_file",
            "embedded_post_text", "search_data", "EmbeddedContentText", "VoskAudioText", "EmbeddedContentText"
        ],
        errors="ignore"
    )

# Step 2: Normalize exaggerated characters
def normalize_exaggeration(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

# Step 3: Fuzzy safety classifier
def is_safety_related_fuzzy(post, threshold=80):
    safety_keywords = [
        "accident", "fire", "explosion", "shoot", "shooting", "evacuate", "emergency",
        "danger", "hazard", "injury", "ambulance", "violence", "earthquake", "crime",
        "gun", "guns", "weapon", "unsafe", "threat", "robbery", "murder", "assault",
        "attack", "abuse", "active shooter", "riot", "panic", "fear"
    ]

    post = normalize_exaggeration(str(post).lower())
    words = re.findall(r'\w+', post)

    for keyword in safety_keywords:
        for word in words:
            if fuzz.partial_ratio(word, keyword) >= threshold:
                return True
    return False

# Step 4: Apply cleaning and classification
Uber_df = clean_tweet(Uber_df)
Marta_df = clean_tweet(Marta_df)
AtlantaBeltline_df = clean_tweet(AtlantaBeltline_df)

Uber_df['is_safety_related'] = Uber_df['post_body_text'].apply(is_safety_related_fuzzy)
Marta_df['is_safety_related'] = Marta_df['post_body_text'].apply(is_safety_related_fuzzy)
AtlantaBeltline_df['is_safety_related'] = AtlantaBeltline_df['post_body_text'].apply(is_safety_related_fuzzy)

Uber_df = Uber_df[Uber_df['is_safety_related']]
Marta_df = Marta_df[Marta_df['is_safety_related']]
AtlantaBeltline_df = AtlantaBeltline_df[AtlantaBeltline_df['is_safety_related']]

# Step 5: Save the filtered datasets
Uber_df.to_csv('data/processed/UBER_Safety_Post.csv', index=False)
Marta_df.to_csv('data/processed/Marta_Safety_Post.csv', index=False)
AtlantaBeltline_df.to_csv('data/processed/AtlBeltline_Safety_Post.csv', index=False)
print("✅ Cleaned safety-related posts saved to data/processed/")
