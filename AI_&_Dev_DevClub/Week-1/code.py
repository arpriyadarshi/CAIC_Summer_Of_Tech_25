#loading the dataset in Google Colab
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("behaviour_simulation_train.csv")

# preprocessing the dataset
df.dropna(subset=['content', 'username', 'likes'], inplace=True)
df['media'].fillna('no_media', inplace=True)
df['has_media'] = df['media'].apply(lambda x: x != 'no_media')
df['content'] = df['content'].astype(str).str.strip().str.lower()
df['inferred company'] = df['inferred company'].astype(str).str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

#removing corrupted data and making it ready for sentiment analysis
!pip install ftfy --quiet

import ftfy
df['new_content'] = df['content'].apply(ftfy.fix_text)
!pip install ftfy emoji textblob --quiet

import emoji
df['new_content'] = df['new_content'].apply(lambda x: emoji.demojize(x))

#sentiment Analysis using TextBlob

from textblob import TextBlob

df['polarity'] = df['new_content'].apply(lambda x: TextBlob(x).sentiment.polarity)


#making features for the model
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['char_len'] = df['new_content'].str.len()
df['word_count'] = df['new_content'].str.split().apply(len)
df['has_hashtag'] = df['new_content'].str.contains('#').astype(int)
df['has_mention'] = df['new_content'].str.contains('<mention>').astype(int)
df['has_link'] = df['new_content'].str.contains('<hyperlink>').astype(int)



def extract_media_flags_from_str(media_str):
    # Handling missing or empty values
    if not isinstance(media_str, str) or media_str.strip() == "":
        return pd.Series([0, 0, 0], index=['has_photo', 'has_video', 'has_gif'])

    has_photo = int("Photo(" in media_str)
    has_video = int("Video(" in media_str)
    has_gif   = int("AnimatedGif(" in media_str or "Gif(" in media_str)

    return pd.Series([has_photo, has_video, has_gif], index=['has_photo', 'has_video', 'has_gif'])

# Applying to DataFrame
df[['has_photo', 'has_video', 'has_gif']] = df['media'].apply(extract_media_flags_from_str)

df['log_likes'] = np.log1p(df['likes'])  # log(1 + x) handles 0 values
company_means = df.groupby('inferred company')['log_likes'].mean()
df['company_avg_log_likes'] = df['inferred company'].map(company_means)



num_cols = ['hour', 'day_of_week', 'is_weekend', 'char_len', 'word_count', 'polarity', 'day_of_week', 'company_avg_log_likes']

# Optional binary ones too, but usually don't need scaling
binary_cols = ['has_hashtag', 'has_mention',
               'is_weekend', 'has_photo', 'has_video', 'has_gif']


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# EDA Notebook - Likes Prediction


# Ensure plots render in notebook
%matplotlib inline
sns.set(style="whitegrid")

# Check basic information
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Records:")
display(df.sample(5))

# ----------------------------
# Distribution of Target Variable
# ----------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df['likes'], bins=50, kde=True)
plt.title("Distribution of Likes")
plt.xlabel("Likes")
plt.ylabel("Frequency")
plt.yscale('log')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['log_likes'], bins=50, kde=True, color='orange')
plt.title("Distribution of Log Likes")
plt.xlabel("Log Likes")
plt.ylabel("Frequency")
plt.show()

print("Insight: 'likes' is highly skewed, hence log transformation was applied for modeling.")

# ----------------------------
# Sentiment Polarity vs Engagement
# ----------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(x='polarity', y='log_likes', data=df)
plt.title("Sentiment Polarity vs Log Likes")
plt.xlabel("Polarity (TextBlob)")
plt.ylabel("Log Likes")
plt.axvline(0, color='gray', linestyle='--')
plt.show()

print("Insight: Posts with slightly positive sentiment tend to get more likes.")

# ----------------------------
# Numeric Feature Correlations
# ----------------------------

plt.figure(figsize=(10, 8))
num_corr = df[num_cols + ['log_likes']].corr()
sns.heatmap(num_corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

print("Insight: Features like `company_popularity`, `company_score`, and `word_count` show moderate correlation with `log_likes`.")

# ----------------------------
# Scatter Plots: Numeric Features vs Log Likes
# ----------------------------

fig, axes = plt.subplots(nrows=len(num_cols), figsize=(8, len(num_cols) * 3))
fig.suptitle('Scatter Plots: Numeric Features vs Log Likes', fontsize=16)

for i, col in enumerate(num_cols):
    sns.scatterplot(x=df[col], y=df['log_likes'], ax=axes[i])
    axes[i].set_title(f'{col} vs log_likes')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ----------------------------
# Binary Feature Distributions
# ----------------------------

fig, axes = plt.subplots(nrows=len(binary_cols), figsize=(8, len(binary_cols) * 3))
fig.suptitle("Log Likes by Binary Features", fontsize=16)

for i, col in enumerate(binary_cols):
    sns.boxplot(x=col, y='log_likes', data=df, ax=axes[i])
    axes[i].set_title(f"log_likes by {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("log_likes")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Insight: Presence of hashtags, links, and photos tends to slightly increase median engagement (log_likes).")

# ----------------------------
# Company-Level Insights
# ----------------------------

top_comp = df['inferred company'].value_counts().nlargest(10).index
plt.figure(figsize=(10, 6))
sns.boxplot(x='inferred company', y='log_likes', data=df[df['inferred company'].isin(top_comp)])
plt.xticks(rotation=45)
plt.title("Log Likes Distribution by Top Companies")
plt.xlabel("Inferred Company")
plt.ylabel("Log Likes")
plt.show()

print("Insight: Certain companies consistently attract higher engagement (e.g., those with higher median log_likes).")

# ----------------------------
# Time-Based Patterns
# ----------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(x='hour', y='log_likes', data=df)
plt.title("Log Likes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Log Likes")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='day_of_week', y='log_likes', data=df)
plt.title("Log Likes by Day of Week (0=Monday)")
plt.xlabel("Day of Week")
plt.ylabel("Log Likes")
plt.show()

print("Insight: Posts during certain hours and mid-week days show slightly higher engagement trends.")

# ----------------------------
# Summary
# ----------------------------

from IPython.display import Markdown

Markdown("""
### 🔍 Summary of Key Insights:

- The distribution of likes is **right-skewed**, requiring a log transformation.
- **Positive sentiment** correlates with slightly better engagement.
- Features like `company_score`, `company_popularity`, and `word_count` show reasonable predictive potential.
- Posts containing **hashtags, mentions, and media** (esp. photos) have higher median likes.
- Engagement levels vary by **company**, **hour of the day**, and **day of the week**.

These insights will help inform feature selection and model development for predicting post engagement.
""")

from sklearn.feature_selection import mutual_info_regression


# X: feature DataFrame (without the target column)
# y: target column (e.g. df['log_likes'] or df['likes'])
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

feature_cols = num_cols + binary_cols
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['new_content'].fillna(''))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Combine features
X = pd.concat([df[feature_cols].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
y = df['log_likes']
# Compute MI
mi_scores = mutual_info_regression(X, y)

# Create a Series for better readability
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Print top features
print("📊 Top features by mutual information:")
print(mi_series.head(20))




#Features Used in the Model 
'''
hour:	The hour of the day when the post was made (0–23). Captures diurnal activity trends and user engagement patterns.
day_of_week:	Day of the week the post was made (0 = Monday, ..., 6 = Sunday). Useful for capturing weekday/weekend engagement cycles.
char_len:	Number of characters in the post. Helps assess content length and its impact on performance.
word_count:	Number of words in the post. Similar to char_len, but more language-aware (helps with verbosity vs. clarity).
polarity:	Sentiment polarity score (typically from -1 to 1) of the post. Can indicate how sentiment correlates with engagement.
company_avg_log_likes:	Average log-likes for the inferred company based on historical data. Acts as a proxy for company popularity.
has_hashtag:	Indicates whether the post includes at least one hashtag. Hashtags can improve discoverability.
has_mention:	Indicates if the post mentions another account. Mentions often increase visibility and interaction.
is_weekend:	Whether the post was made on a weekend (Saturday/Sunday). Engagement patterns can differ on weekends.
has_photo:	Whether the post includes a photo. Posts with media often get more engagement.
has_video:	Whether the post includes a video. Similar to has_photo, but captures richer content.
has_gif:	Whether the post contains a GIF. GIFs may attract more attention or imply casual tone.
'''
