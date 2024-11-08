from nltk.util import ngrams

def generate_ngrams(token_arrays, n):
    
    ngram_arrays = []
    for tokens in token_arrays:
        ngram_list = list(ngrams(tokens, n))
        ngram_arrays.append(ngram_list)
    return ngram_arrays