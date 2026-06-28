import re
import joblib

def clean(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

model = joblib.load("model.pkl")

vectorizer = joblib.load("vectorizer.pkl")

news = input("Digite a notícia:\n")
news = clean(news)
news = vectorizer.transform([news])

prediction = model.predict(news)

if prediction[0] == 0:
    print("\nFake News")

else:
    print("\nTrue News")