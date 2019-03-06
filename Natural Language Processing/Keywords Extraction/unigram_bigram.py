import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def get_TF(file, stop_words = None, ngram_min = 1, ngram_max = 1):
    """
    return count table
    
    file:  article used for analysis
    stop_words: words commonly used (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore
    """
    
    if len(file) != 0:
        
        #drop common terms and special charactera
        #create unigram
        cvec = CountVectorizer(ngram_range=(ngram_min , ngram_max), stop_words=stop_words)
        cvec.fit_transform(file)
        words = cvec.get_feature_names()
        counts = np.asarray(cvec.transform(file).sum(axis=0)).ravel().tolist()

        table = pd.DataFrame({'Word': words, 'Count': counts})
        table = table[['Word', 'Count']]
        table.sort_values('Count', ascending=False, inplace = True)
        table.reset_index(drop=True, inplace = True)   
               
        return table
        
        
    else:
        print('This file is empty.')

def count_(df, k):
    
    d = defaultdict(int)

    for words in df['Word']:
        w = tuple(words.split())
        if len(w) == 1:
            d[w] += 1
        else:
            for i in range(2):
                d[w[i]] += 1 

    #sort dictionary by values in descending order
    d_sorted = sorted(d.items(), key=lambda t: t[1], reverse = True)
    selected_words = []
    for i in range(10):
        selected_words.append(d_sorted[i][0])

    return selected_words


def remove_idx(idx, selected_words):

    for i in idx:
        if any(j in i for j in selected_words):
            idx.remove(i)
        
    return idx


if __name__ == '__main__':

    with open('sample_jet.txt') as f:
                words = [f.read()]
            
    df = get_TF(words, stop_words='english', ngram_min = 1, ngram_max = 2)

    min_freq = df['Count'].min()

    a = 0
    for i, j in enumerate(df['Count']):
        if j == min_freq:
            a = i
            break
        
    upper_df = df.iloc[:a]
    lower_df = df.iloc[a:]

    selected_words = count_(lower_df, 10)
    upper_bigram = [j for j in upper_df['Word'] if len(j.split()) == 2] 


    for i in selected_words:
        upper_df = upper_df[upper_df.Word != i]

    idx = remove_idx(upper_bigram, selected_words)
    
    for i in idx:
        upper_df = upper_df[upper_df.Word != i]
        
    idx1 = [j for j in upper_df['Word'] if len(j.split()) == 2] 
    one_word = [j for j in upper_df['Word'] if len(j.split()) == 1]

    k = len(one_word)
    while k > 0:   

        for i in one_word:
            if any(i in j for j in idx1):
                one_word.remove(i)

        k -= 1
        
    keywords = idx1 + one_word
    
    df_new = pd.DataFrame()
    for key in keywords:
        df1 = upper_df[upper_df.Word == key]
        df_new = df_new.append(df1)
        
    df_new = df_new.sort_values(by='Count', ascending=False)
    df_new = df_new.reset_index(drop=True)
    
    print('Extracted Keywords:')
    print(*df_new.Word[:9], sep=', ')
    print()
    print()
    print('Generated Frequency Table:')
    print()
    print(df_new)
