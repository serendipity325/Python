# Keywords Extraction

Select 3-7 keywords that are somewhat representative of what the content of the document is. 

1. TF-IDF 

    The TF-IDF method applies Term-Frequency-Inverse-Document-Frequency (TF-IDF) normalization to a sparse matrix of occurrence counts. 
    In other words, it is a method to extract words with low frequency in training data but high frequency in test data. 
    Text files that are not mostly associated with the test data are used as training data. 
    Stop-words and special characters are removed. Seven words with the highest frequency were extracted.
    
    
2. word2vec

    The model takes word2vec representations of words in a vector space. 
    I used Google News corpora which provided by Google which consist of 3 million word vectors. 
    Due to the large size of the data, it takes about 30 minutes to train the embedding model.
    The second step is to find the PageRank value of each word. In the model PageRank algorithm takes word2vec representations of words. 
    The cosine distance and similarity is used to calculate edge weights between nodes. 
    After PageRank values of words are found, words which have the highest PageRank values will be selected as keywords.
    
    
3. non-overlapping-unigram-bigram model
    
    The objective of this model is to use words that are treated as a single word, such as San Francisco.  
    The algorithm is as follows:

    1. Using CountVectorize from Scikit-Learn, erase special characters and stop-words, 
       then extract both unigrams and bigrams and put them in a table.
    2. Find the smallest frequency in the table. Then make a hash-table (dictionary) that represents 
       the frequency of each word from all unigrams and bigrams which have the smallest frequency.   
    3. Remove the used unigrams and bigrams from the table, and use the remaining table as a data. 
    4. Extract 10 words with the highest frequency from the hash-table.
    5. After extracting bigrams again from the remaining data, leave only bigrams containing 
       the selected 10 most frequent words and erase the remaining bigrams from the data. 
       The 10 most frequent words are also deleted from the remaining data.
    6. Extract unigrams from the remaining data. 
       If an extracted unigram is included in any bigrams which are still in the data, the unigram is removed.
    7. By sorting the frequency of the remaining words, words with the highest frequency are extracted as keywords.





