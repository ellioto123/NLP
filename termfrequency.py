import numpy as np
def get_terms(docs):
    terms = set()
    for doc in docs:
        terms.update(doc)
    return terms

def TF(terms, doc):
    array = []
    for term in terms:
        array.append( doc.count(term))
    return array
def DF(term, docs):
     return sum([1 for doc in docs if term in doc])    

def IDF(terms, docs):
    array = []
    for term in terms:
        array.append( np.log10(len(docs) / (DF(term, docs)+1)))
    return array

def TFIDF(terms, docs):
    tfidf_matrix = []
    idf_values = IDF(terms, docs)
    for doc in docs:
        tf_values = TF(terms, doc)
        tfidf_values = [tf * idf for tf, idf in zip(tf_values, idf_values)]
        tfidf_matrix.append(tfidf_values)
    return tfidf_matrix
    # array = []
    # array.append( [TF(terms, doc) * IDF(terms, docs) for doc in docs])
    # return array

def get_term_frequency(docs):
    terms = get_terms(docs)
    return TFIDF(terms, docs)