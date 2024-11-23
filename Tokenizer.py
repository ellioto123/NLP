#Tokenization

import nltk
import string
import re

nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

def tokenize(text):
    for x in range(len(text)):
        text[x] = text[x].lower()
        text[x] = re.sub(r'[^\w\s]', '', text[x])  # removes punctuation
        text[x] = text[x].split(" ")


def remove_stopwords_and_lemmatize(text):
    lemmatizer = nltk.WordNetLemmatizer()
    stopwordlist = set(stopwords.words('english'))
    stopwordlist.discard('not')
    wordlist = [lemmatizer.lemmatize(word) for word in text
                 if word not in stopwordlist]
    return wordlist


def remove_stopwords_and_stem(text):
    stopwordlist = set(stopwords.words('english'))
    stopwordlist.discard('not')
    stemmer = LancasterStemmer()
    word_list = [stemmer.stem(word) for word in text
                 if word not in stopwordlist]
    
    return word_list


    





