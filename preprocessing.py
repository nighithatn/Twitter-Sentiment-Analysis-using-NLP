#Text cleaning 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stopWords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def cleanText(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    cleanedWords = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopWords
    ]

    return " ".join(cleanedWords)
