import numpy as np
import math

def get_terms(text):
    terms = {}
    for word in text:
        terms[word] = terms.get(word, 0) + 1
    return terms

def collect_vocabulary(reviews):
    all_terms = []
    for review in range(len(reviews)):
        for term in reviews[review]:
            all_terms.append(term)  
        
    return sorted(set(all_terms))

def vectorize(input_terms, shared_vocabulary):
    output = {}
    for item_id in input_terms.keys(): # e.g., a review in input_terms
        terms = input_terms.get(item_id) 
        output_vector = []
        for word in shared_vocabulary:
            if word in terms.keys():
                # add the raw count of the word from the shared vocabulary in doc to the doc vector
                output_vector.append(int(terms.get(word,0)))
            else:
                # if the word from the shared vocabulary is not in doc, add 0 to the doc vector in this position
                output_vector.append(0)
        output[item_id] = output_vector
    return output

def calculate_idfs(shared_vocabulary, d_terms):
    doc_idfs = {}
    for term in shared_vocabulary:
        doc_count = 0 # the number of documents containing this term
        for doc_id in d_terms.keys():
            terms = d_terms.get(doc_id)
            if term in terms.keys():
                doc_count += 1
        doc_idfs[term] = math.log(float(len(d_terms.keys()))/float(1 + doc_count), 10)
    return doc_idfs

def vectorize_idf(input_terms, input_idfs, shared_vocabulary):
    output = {}
    for item_id in input_terms.keys():
        terms = input_terms.get(item_id)# collect terms from the document
        output_vector = []
        for term in shared_vocabulary:
            if term in terms.keys():
                output_vector.append(input_idfs.get(term)*float(terms.get(term)))
            else:
                output_vector.append(float(0))
        output[item_id] = output_vector
    return output





# def get_terms(docs):
#     terms = set()
#     for doc in docs:
#         terms.update(doc)
#     return terms

# def TF(terms, doc):
#     array = []
#     for term in terms:
#         array.append( doc.count(term))
#     return array
# def DF(term, docs):
#      return sum([1 for doc in docs if term in doc])    

# def IDF(terms, docs):
#     array = []
#     for term in terms:
#         array.append( np.log10(len(docs) / (DF(term, docs)+1)))
#     return array

# def TFIDF(terms, docs):
#     tfidf_matrix = []
#     idf_values = IDF(terms, docs)
#     for doc in docs:
#         tf_values = TF(terms, doc)
#         tfidf_values = [tf * idf for tf, idf in zip(tf_values, idf_values)]
#         tfidf_matrix.append(tfidf_values)
#     return tfidf_matrix
#     # array = []
#     # array.append( [TF(terms, doc) * IDF(terms, docs) for doc in docs])
#     # return array

# def get_term_frequency(docs):
#     terms = get_terms(docs)
#     return TFIDF(terms, docs)