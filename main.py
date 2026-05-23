import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASETS
# =========================

fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

# =========================
# ADD LABELS
# =========================

fake_data["label"] = 0
true_data["label"] = 1

# =========================
# COMBINE DATASETS
# =========================

data = pd.concat([fake_data, true_data], axis=0)

# Shuffle data
data = data.sample(frac=1)

# =========================
# SELECT FEATURES
# =========================

x = data["text"]
y = data["label"]

# =========================
# CONVERT TEXT TO NUMBERS
# =========================

vectorization = TfidfVectorizer()

xv = vectorization.fit_transform(x)

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    xv,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression()

model.fit(x_train, y_train)

# =========================
# TEST MODEL
# =========================

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

# =========================
# USER INPUT
# =========================

news = input("Enter News Text: ")

news_vector = vectorization.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")