import pandas as pd

# Load dataset
train_data = pd.read_parquet(
    r"C:\Users\ASUS\OneDrive\Desktop\Emotion_Detection\dataset\train-00000-of-00001.parquet"
)

# Emotion labels
emotion_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Convert label numbers into emotion names
train_data["emotion"] = train_data["label"].map(emotion_names)

# Show sample data
print("Sample data:")
print(train_data[["text", "emotion"]].head(10))

# Show emotion distribution
print("\nEmotion distribution:")
print(train_data["emotion"].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

# Separate text and emotion label
X = train_data["text"]
y = train_data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF completed!")
print("Original data shape:", X.shape)
print("TF-IDF shape:", X_tfidf.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create the ML model
model = LogisticRegression(max_iter=1000)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Model training completed!")

# Predict emotions for test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Detailed performance
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Sadness",
        "Joy",
        "Love",
        "Anger",
        "Fear",
        "Surprise"
    ]
))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

emotion_names_list = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

# Plot confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=emotion_names_list,
    yticklabels=emotion_names_list
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Emotion Detection - Confusion Matrix")
plt.tight_layout()
plt.show()

# Test the model with your own sentence

user_text = input("\nEnter a sentence: ")

# Convert user text into TF-IDF
user_text_tfidf = vectorizer.transform([user_text])

# Predict emotion
prediction = model.predict(user_text_tfidf)[0]

# Convert label to emotion name
predicted_emotion = emotion_names[prediction]

print("\nPredicted Emotion:", predicted_emotion)

import pickle

# Save the trained model
with open("emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")

import pickle

# Save trained model
with open("emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")