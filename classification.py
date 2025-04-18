import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# API set up
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")

candidate_labels = [
    "crime", "shooting", "robbery", "assault", "kidnapping", "theft", "gun violence", "stabbing", "active shooter",
    "sexual harassment", "verbal harassment", "physical harassment", "driver behavior", "reckless driving",
    "hit and run", "drunk driving", "abandoned vehicle", "unsafe pedestrian crossing", "traffic accident",
    "car crash", "infrastructure damage", "unsafe sidewalk", "road hazard", "potholes", "collapsed bridge",
    "unsafe lighting", "street lights out", "dark alley", "surveillance concern", "police presence",
    "police brutality", "emergency response delay", "slow ambulance", "public transportation delay",
    "bus breakdown", "MARTA malfunction", "Beltline crime", "public intoxication", "homeless safety concern",
    "mental health crisis", "fire hazard", "explosion", "evacuation", "natural disaster", "earthquake", "flooding",
    "hazardous material", "unsafe construction site", "unsafe gathering", "crowd control issue",
    "noise disturbance", "loitering", "racial profiling", "discrimination", "public panic", "terrorism threat",
    "public health risk", "contamination", "disease outbreak", "unsanitary condition", "inadequate sanitation"
]

system_prompt = (
    "You are a social media analyst. A user will give you a set of tweets from a cluster. "
    "Your task is to classify the theme using one of the candidate labels provided. "
    "Respond ONLY with one label from this list: " + ", ".join(candidate_labels) + "."
)

def summarize_cluster(posts):
    joined = "\n\n".join(f"- {p}" for p in posts)
    user_prompt = f"Here are some posts from the same cluster:\n\n{joined}\n\nPick the best matching topic from the list."

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error summarizing cluster: {e}")
        return "unknown"

def process_cluster_file(filepath, output_path):
    df = pd.read_csv(filepath)
    cluster_labels = []

    for cluster_id in sorted(df['cluster'].unique()):
        samples = df[df['cluster'] == cluster_id].sample(n=5, random_state=42)['post_body_text'].tolist()
        label = summarize_cluster(samples)
        print(f"Cluster {cluster_id + 1}: {label}")
        cluster_labels.append({"cluster": cluster_id, "label": label})

    # Save result
    label_df = pd.DataFrame(cluster_labels)
    label_df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}\n")

# run the classification
os.makedirs("outputs/analysis", exist_ok=True)

print("📌UBER Major Safety Issues")
process_cluster_file(
    filepath="data/processed/UBER_Clustered.csv",
    output_path="outputs/analysis/UBER_Cluster_Labels.csv"
)

print("📌Marta Major Safety Issues")
process_cluster_file(
    filepath="data/processed/Marta_Clustered.csv",
    output_path="outputs/analysis/Marta_Cluster_Labels.csv"
)

print("📌AtlantaBeltline Major Safety Issues")
process_cluster_file(
    filepath="data/processed/AtlantaBeltline_Clustered.csv",
    output_path="outputs/analysis/AtlantaBeltline_Cluster_Labels.csv"
)
