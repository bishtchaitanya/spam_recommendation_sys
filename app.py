import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import scipy
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def remove_hyperlink(word):
    word = str(word)
    return re.sub(r"http\S+", "", word)


def to_lower(word):
    result = word.lower()
    return result


def remove_number(word):
    word = str(word)
    result = re.sub(r'\d+', '', word)
    return result


def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(word):
    result = word.strip()
    return result


def replace_newline(word):
    return word.replace('\n', '')


def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation, remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def remove_stop_words(words):
    result = [i for i in words if i not in stopwords.words()]
    return result


def word_stemmer(words):
    return [stemmer.stem(o) for o in words]


def word_lemmatizer(words):
    return [lemmatizer.lemmatize(o) for o in words]


def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words, word_lemmatizer]
    for o in cleaning_utils:
        words = o(words)
    return words


def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    return tfidf.transform(raw_sentences)


def data_prep(x_test):
    if isinstance(x_test, str):
        x_test = [x_test]

    # Tokenize each document in the list
    x_test = [word_tokenize(doc) for doc in x_test]
    # Clean each document in the list
    x_test = [clean_token_pipeline(doc) for doc in x_test]
    # Join the tokens back into strings
    x_test = [" ".join(doc) for doc in x_test]

    x_test_features = tfidf.transform(x_test)

    return x_test_features


tfidf = joblib.load('vectorizer.pkl')
model = joblib.load('spam_classifier.pkl')

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    vector_input = data_prep(input_sms)
    # 3. predict
    result = model.predict(vector_input)
    # 4. Display
    if result[0] == 'spam':
        st.header("Spam")
    else:
        st.header("Not Spam")
