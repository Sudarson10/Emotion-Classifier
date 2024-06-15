import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import warnings
import joblib
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load models and setup NLTK
rf = joblib.load(r'C:\Users\sudar\OneDrive\Desktop\miniprojects\emotionclassifier\randomforest.joblib')
tfidf_vectorizer = pickle.load(open(r'C:\Users\sudar\OneDrive\Desktop\miniprojects\emotionclassifier\tfidfvectorizer.pkl', 'rb'))
lb = pickle.load(open(r'C:\Users\sudar\OneDrive\Desktop\miniprojects\emotionclassifier\label_encoder.pkl', 'rb'))
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to clean text
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Function to predict emotion
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = rf.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion

# Function to handle button click
def classify_text():
    input_text = entry.get()
    if input_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter some text.")
    else:
        predicted_emotion = predict_emotion(input_text)
        result_label.config(text=f"Predicted Emotion: {predicted_emotion}")

# Create tkinter window
window = tk.Tk()
window.title("Emotion Classification")
window.geometry("500x300")


label = ttk.Label(window, text="Enter text to classify:")
label.pack(pady=10)


entry = ttk.Entry(window, width=50)
entry.pack(pady=10)

classify_button = ttk.Button(window, text="Classify", command=classify_text)
classify_button.pack(pady=10)

result_label = ttk.Label(window, text="")
result_label.pack(pady=10)

window.mainloop()
