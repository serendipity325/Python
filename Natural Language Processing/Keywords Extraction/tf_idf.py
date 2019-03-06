import numpy as np
import pandas as pd

# need to download brown and reuters corpora first if you don't have them
"""
import nltk
nltk.download('brown')
nltk.download('reuters')
"""

from nltk.corpus import stopwords
from nltk.corpus import brown, reuters
from nltk.tokenize import RegexpTokenizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

   
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
        


def get_TFIDF(train_data, test_data, stop_words = None, ngram_min = 1, ngram_max = 1):
    """
    return frequency table using TF-IDF mothod
    
    train_data: articles for training TFIDF included in sections other than test_data
    test_data:  article used for analysis
    stop_words: words commonly used (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore
    """
    
    if len(train_data) != 0 or len(test_data) != 0 :
        
        #drop common terms and special charactera
        #create unigram
        tvec = TfidfVectorizer(ngram_range=(ngram_min , ngram_max), stop_words=stop_words)
        tvec_weights = tvec.fit_transform(train_data)
        words = tvec.get_feature_names()
        weights = np.asarray(tvec.transform(test_data).mean(axis=0)).ravel().tolist()

        table = pd.DataFrame({'Word': words, 'Weight': weights})
        table = table[['Word', 'Weight']]
        table.sort_values('Weight', ascending=False, inplace = True)
        table.reset_index(drop=True, inplace = True)
        table = table[(table[['Word', 'Weight']] != 0).all(axis=1)] 
        
        return table
        
        
    else:
        print('Train or Test file is empty.')


def _merge(df1, df2):
    """
    merge local frequency table with universal frequency table
    
    df1: local frequency table
    df2: universal frequency table
    """

    df = df1.merge(df2, on = 'Word', how='left')
    df.columns = ['Word', 'Local_Frequency', 'Universal_Frequency']
    df.sort_values(['Local_Frequency', 'Universal_Frequency'], ascending=[False, False], inplace=True)
    df = df.reset_index(drop=True)

    return df


def performTermExtraction(df, a, b):
    """
    remove words sequentially by most and least elimination ratios
    return a frequency table with 3 - 7 words or error message  
    
    df: frequency table
    a:  elimination ratio of most frequent words
    b:  elimination ratio of least frequent words
    """
    
    N = len(df)
    c = a + b
        
    while N > 7 + c:

        df =  df.iloc[a : N - b]
        N -= c

    if N < 3:
        print('Error: The input file is too small to use or need to adjust elimination ratios.')    

    elif 3 <= N <= 7:
         return df
            
    elif 7 < N <= b + 7:        
        if N < a:
            print('Error: Need to adjust elimination ratios.')        
        else: 
            df =  df.iloc[a : N]
            if 3 <= len(df) <= 7:
                return df 

            elif len(df) > 7:
                if N - b < 3 or N - c < 3:
                    print('Error: Need to adjust elimination ratios.')                                    
                else:
                    return df.iloc[: len(df) - b]                     
                    
            elif len(df) < 3:
                print('Error: Need to adjust elimination ratios.')     
        
    elif b + 7 < N <= c + 7:
        if (N - a < 3) or (N - c < 3):
            print('Error: Need to adjust elimination ratios.')
            
        elif 3 <= N - a <= 7 :    
            return df.iloc[a : N + a]               
        
        elif N - c >= 3:
            return df.iloc[a : N - b]


def return_result(var):
    """
    return extracted keywords or error message
    
    var: returned dataframe or error message
    """    
    if isinstance(var, pd.DataFrame):
        print(*var['Word'], sep=', ')

    else:
        var            

def least_ratios(df, a):
    """
    generate possible elimination ratios of least frequent words
    
    df: frequency table
    a:  elimination ratio of most frequent words
    """
    N = len(df)
    remainders = list(range(0, 7 + a + 1))
    ratios = []
    for i in range(N):
        _, remainder = divmod(N, i + a)
        if remainder in remainders:
            ratios.append(i)
        
    return ratios

def keywords_by_diff_ratios(df, a, least_ratios):
    """
    return the most frequent elimination ratio, the least frequent elimination ratio,
    and the corresponding keywords or error message, respectively, used in the extraction 

    df: frequency table
    a:  most frequent elimination ratio
    least_ratios: least frequent elimination ratios that may be possible to extract keywords
    """
    for ratio in least_ratios:
        print('most freq. ratio: %d' % a, ', ', 'least freq. ratio: %d' % ratio)
        print('------' * 7)
        return_result(performTermExtraction(df, a, ratio))
        print()
        print()


def universal_unigram():
    """
    create universal unigram using brown and reuters corpus from nltk
    and fetch_20newsgroups corpus from scikit-learn
    """
    
    brown_ = [brown.words(categories=c) for c in brown.categories()]
    flattened_brown = [val for sublist in brown_ for val in sublist]

    reuters_ = [reuters.words(categories=c) for c in reuters.categories()]
    flattened_reuters = [val for sublist in reuters_ for val in sublist]

    newsgroups_ = fetch_20newsgroups(subset='all')
    flattened_newsgroups = [val for sublist in newsgroups_.data for val in sublist.split()]

    universal_ = flattened_brown + flattened_reuters + flattened_newsgroups

    return get_TF(universal_)
    
    
    
if __name__ == '__main__':
    
    with open('sample_jet.txt') as f:
            words = [f.read()]

    local_unigram = get_TF(words)    
    universal_unigram = universal_unigram()
    
    frequency_table = _merge(local_unigram, universal_unigram)

    print('Look at the keywords extracted by changing ratios')
    print()
    
    #The following shows the keywords that can be extracted using various elimination ratios.
    # a's are most frequent elimination ratios 
        
    a = 1
    print('Selected Keywords with %d as most frequent elimination ratio and various least frequent elimination ratios' %a)
    print()
    keywords_by_diff_ratios(frequency_table, a, least_ratios(frequency_table, a))
    print()
    print()
    
    a = 2
    print('Selected Keywords with %d as most frequent elimination ratio and various least frequent elimination ratios' %a)
    print()
    keywords_by_diff_ratios(frequency_table, a, least_ratios(frequency_table, a))
    print()
    print()
    
    
    a = 5
    print('Selected Keywords with %d as most frequent elimination ratio and various least frequent elimination ratios' %a)
    print()
    keywords_by_diff_ratios(frequency_table, a, least_ratios(frequency_table, a))
    print()
    print()    
    
    a = 10
    print('Selected Keywords with %d as most frequent elimination ratio and various least frequent elimination ratios' %a)
    print()
    keywords_by_diff_ratios(frequency_table, a, least_ratios(frequency_table, a))
    print()
    print()

    ##########################################       #4 with TF-IDF      ##########################################    
    print()
    print()
    print('Result of #4 with TF-IDF')
    
    #train data for TF-IDF
    train_data = brown.words(categories='adventure') + brown.words(categories='lore') + brown.words(categories='fiction')
    test_data = words

    #remove stop-words and employ only unigram 
    tfidf = get_TFIDF(train_data, test_data, stop_words='english', ngram_min = 1, ngram_max = 1)
    frequency_table2 = _merge(tfidf, universal_unigram)
    print('Extracted Keywords:')
    return_result(frequency_table2[:7])
    print()
    print()
    print('*Frequency table*')      
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(frequency_table2)
    print()
    print()
