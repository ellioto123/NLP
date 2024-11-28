import nltk
import pandas as pd
import numpy as np
from nltk import RegexpParser
from sklearn.feature_extraction import DictVectorizer

def posTag(reviewset):
    posTaggedTokens = []
    for review in reviewset:
        posTaggedTokens.append(nltk.pos_tag(review))

    return posTaggedTokens

grammar = "NP: {<DT>?<JJ>*<NN>+}"

def constParse(taggedReviews, grammar):
    parser = RegexpParser(grammar)
    parsedReviews = []
    for review in taggedReviews:
        parsedReviews.append(parser.parse(review))
    print(taggedReviews[0])
    return parsedReviews

def extract_noun_phrases(parsedReviews):
    noun_phrases_list = []
    for tree in parsedReviews:
        nounphrases = []
        for subtree in tree.subtrees(filter=lambda x: x.label() == 'NP'):
            noun_phrase = ' '.join(word for word, pos in subtree.leaves())
            nounphrases.append(noun_phrase)
        noun_phrases_list.append(nounphrases)
    return noun_phrases_list

def getcountNP(parsedReviews):
    countNP = []
    for review in parsedReviews:
        count = 0
        for subtree in review.subtrees():
            if subtree.label() == 'NP':
                count += 1
        countNP.append(count)
    print(parsedReviews[0])
    return countNP




def combine_matrices(tfidfmatrix, countNP_matrix):
    countNP_matrix = np.array(countNP_matrix)[:, np.newaxis] #make sure npmatrix is a column vector
    countNP_matrix = pd.DataFrame(countNP_matrix,columns=['CountNP'])
    combined_df = pd.concat([tfidfmatrix, countNP_matrix], axis=1)
    return combined_df

def run(reviewset):
    taggedReviews = posTag(reviewset)
    parsedReviews = constParse(taggedReviews, grammar)
    # countNPs = getcountNP(parsedReviews)
    nounPhrases = extract_noun_phrases(parsedReviews)
    return nounPhrases





