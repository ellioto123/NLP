#Tokenization

import nltk
import string

nltk.download('stopwords')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

def remove_stopwords_and_stem(text):
    stopwordlist = set(stopwords.words('english'))
    stemmer = LancasterStemmer()
    word_list = [stemmer.stem(word) for word in text
                 if word not in stopwordlist]
    
    return word_list





