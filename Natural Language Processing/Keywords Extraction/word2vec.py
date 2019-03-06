#   Please run this file in python2.
#   It would take too much time to train the embedding model with python3.

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors

def word_list(text):
    """
    remove common words and non-alphabetic characters
    """
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
           
    stop_words = set(stopwords.words('english'))
    word_list = [w for w in words if w.isalpha() and not w in stop_words]

    return word_list



def get_vectors(words, model):
    """
    get vectors of words
    """
    
    #set of names of the words in embedding model
    word_idx = set(model.index2word) 

    text = []    
    for word in words:
        if word in word_idx:
            text.append(model[word])

    return text


def pagerank(A, x0, m, iter):
    """
    compute pagerank powermethod
    """
    
    n = A.shape[1]
    delta = m * (np.array([1] * n, dtype='float64') / n)

    for i in range(iter):
        x0 = np.dot((1 - m), np.dot(A, x0)) + delta

    return x0


if __name__ == '__main__':

    file_path = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    embedding_model = KeyedVectors.load_word2vec_format(file_path, binary=True)

    #another embedding model
    #wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)

    with open('sample_jet.txt') as f:
            words = f.read()

    #bag of words list
    bag_of_words = (word_list(words))

    #remove words that occur more than one
    train = []
    for words in bag_of_words:
        if words in train:
            words = +1
        else:
            train.append(words)

    trained_vecs = get_vectors(train, embedding_model)
    train_data = np.asarray(trained_vecs)

    #calculate cosine similarity matrix to use in pagerank algorithm for dense matrix
    similarity = np.dot(train_data, train_data.T)

    #inverse of squared magnitude of number of occurrences
    inv_sq_magnitude = 1 / np.diag(similarity)

    #set inverse magnitude to zero if it doesn't occur
    inv_sq_magnitude[np.isinf(inv_sq_magnitude)] = 0

    #inverse of the magnitude
    inv_mag = np.sqrt(inv_sq_magnitude)

    #cosine similarity
    cosine = similarity * inv_mag.T * inv_mag

    x0 = [1] * cosine.shape[0]
    pagerank_ = pagerank(cosine, x0, 0.15, 130)

    #select 10 most frequent words
    srt = np.argsort(pagerank_)
    list_of_keywords = srt[:10]

    most_frequent_keywords = []

    for words in list_of_keywords:
        most_frequent_keywords.append(bag_of_words[words])

    
    print'Extracted Keywords:'    
    print most_frequent_keywords
