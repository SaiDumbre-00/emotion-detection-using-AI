import tkinter as tk
from tkinter import messagebox
import pickle

# Load trained model
with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Emotion labels
emotion_names = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}


def predict_emotion():
    text = text_box.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning("Warning", "Please enter some text.")
        return

    # Convert text to TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Predict emotion
    prediction = model.predict(text_tfidf)[0]

    # Get emotion name
    emotion = emotion_names[prediction]

    result_label.config(text="Predicted Emotion: " + emotion)


# Create window
window = tk.Tk()
window.title("Emotion Detection")
window.geometry("600x450")


# Title
title_label = tk.Label(
    window,
    text="Emotion Detection System",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=25)


# Instruction
instruction = tk.Label(
    window,
    text="Enter a sentence to detect its emotion:",
    font=("Arial", 12)
)
instruction.pack()


# Text box
text_box = tk.Text(
    window,
    height=6,
    width=60,
    font=("Arial", 12)
)
text_box.pack(pady=15)


# Predict button
predict_button = tk.Button(
    window,
    text="Predict Emotion",
    font=("Arial", 12, "bold"),
    command=predict_emotion
)
predict_button.pack(pady=10)


# Result
result_label = tk.Label(
    window,
    text="Predicted Emotion: ",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=20)


# Start application
window.mainloop()