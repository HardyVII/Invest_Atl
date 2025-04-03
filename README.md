# 🚇 Public Safety Sentiment Analysis in Atlanta

## Overview

This project analyzes **public perception of safety issues** across three key transportation-related sectors in Atlanta:

- 🏙️ **Atlanta Beltline** *(Non-Governmental Organization - NGO)*
- 🏛️ **MARTA** *(Government Agency)*
- 🚗 **Uber** *(Private Company)*

We use X, IG, etc. data to perform sentiment and topic analysis to uncover trends, concerns, and public opinion about safety for each organization.

---

## 🧠 Objective

To understand how the public perceives **safety** in:
- Non-public organizations (NGO: Atlanta Beltline),
- Government-operated services (MARTA), and
- Private sector services (Uber).

This insight can inform each organization on how to better address safety concerns and improve public trust.

---

## 📊 Methodology

1. **Data Collection**
   - Twitter data related to the three organizations was collected using Junkipedia and custom filters.
   - Exported as CSV and processed using Python.

2. **Data Preprocessing**
   - Cleaned tweets using NLP techniques (removal of URLs, mentions, stopwords, etc.).
   - Filtered tweets based on safety-related keywords.

3. **Sentiment Analysis**
   - Used VADER and transformer-based models to classify each tweet as positive, neutral, or negative.
   - Calculated sentiment distributions per organization.

4. **Topic Discovery via Clustering**
   - Converted tweets into embeddings using `sentence-transformers`.
   - Applied **k-means clustering** to discover subtopics related to safety (e.g., crime, infrastructure, harassment).

5. **Temporal & Comparative Analysis**
   - Tracked changes in public sentiment over time.
   - Compared public perception across Uber, MARTA, and the Atlanta Beltline.

6. **Visualization**
   - Created insightful charts to present sentiment distribution, cluster themes, and time trends.
   - Used t-SNE and bar plots for visual clarity.
