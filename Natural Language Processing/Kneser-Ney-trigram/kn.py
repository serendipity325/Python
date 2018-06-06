import numpy as np

#read only words and add <s> and </s> to front and end of each line
def get_words(x):

    words = []
    with open(x) as fout:
        for line in fout:
            line = '<s> '+line.strip()+' </s>'
            word = line.split(' ')[0::2]
            words.append(word)

    return words


def get_ngrams(words):
    
    unigram = {}
    bigram = {}
    trigram = {}
    N_1 = {}
    N1_ = {}
    N_2 = {}
    N2_ = {}
    N12 = {}

    for word in words:
        for i, w in enumerate(word):

    #generate unigram
            if w in unigram:
                unigram[w]+=1
            else:
                unigram[w]=1

    #generate bigram
            if i < len(word) - 1:
                if (word[i],word[i+1]) in bigram:
                    bigram[word[i],word[i+1]]+=1
                else:
                    bigram[word[i],word[i+1]]=1

    #generate trigram
            if i < len(word) - 2:
                if (word[i], word[i+1], word[i+2]) in trigram:
                    trigram[word[i], word[i+1], word[i+2]] += 1
                else:
                    trigram[word[i], word[i+1], word[i+2]] = 1



    #count values for each dictionary using bigram and trigram
    for key in bigram.keys():

        #N1+(Wi-1, *)
        if key[0] in N1_:
            N1_[key[0]]+=1
        else:
            N1_[key[0]]=1

        #N1+(*, Wi)
        if key[1] in N_1:
            N_1[key[1]]+=1
        else:
            N_1[key[1]]=1

    for key in trigram.keys():

        #N1+(Wi-2, Wi-1, *)
        if (key[0], key[1]) in N2_:
            N2_[(key[0], key[1])] += 1
        else:
            N2_[(key[0], key[1])] = 1

        #N1+(*, Wi-1, Wi)
        if (key[1], key[2]) in N_2:
            N_2[(key[1], key[2])] += 1
        else:
            N_2[(key[1], key[2])] = 1

        #N1+(*, Wi-1, *)
        if key[1] in N12:
            N12[key[1]] += 1
        else:
            N12[key[1]] = 1
    
    return unigram, bigram, trigram, N_1, N1_, N2_, N_2, N12

#original version of Kneser-Ney smoother
def Kneser_Ney1(C210, C21_, N2_, C1_ , N1_, N_1, d,  N000, C0, N):

    #C(Wi-2, Wi-1, Wi) > 0 and N1+(Wi-2, Wi-1, *) > 0
    if C210 > 0 and C21_ > 0:
        return (max(0.0, C210 - d) / C21_) + (d * N2_ / C21_ *  (max(0.0, C21_ - d) / C1_ + (d * N1_ * N_1/ C1_ /N000)))

    #N1+(Wi-2, Wi-1, *) > 0 and C(Wi-2, Wi-1) > 0
    elif N2_ > 0 and C21_ > 0:
        return (d * N2_ / C21_ *  (max(0.0, C21_ - d) / C1_ + (d * N1_ * N_1/ C1_ /N000)))

    #C(Wi-2, Wi-1) = 0
    #elif C21_ == 0:
     #   return (max(0.0, C21_ - d) / C1_ + (d * N1_ * N_1/ C1_ /N000))
    
    else:
        return C0 / float(N)


#compute probability of original version of Kneser-Ney smoother
def get_KN1(testfile, unigram, bigram, trigram, N_1, N1_, N2_, d): 

    N00 = sum(N_1.values())
    N000 = sum(unigram.values()) - unigram['<s>']
    Nt = sum(trigram.values()) - unigram['<s>']
    
    pro_KN = 0.0
    num_process = 0
    KNS = []

    for sentence in testfile:

        for i, _ in enumerate(sentence, start = -2):

            if i == -2:
                continue
            if i == -1:
                continue

            count = trigram.get((sentence[i], sentence[i+1], sentence[i+2]), 0) 

            KN = 0

            #if C(Wi-2, Wi-1, Wi) = 0    
            if count == 0:

                #skip if there is no Wi
                if sentence[i+2] not in unigram:
                    continue

                #Wi is in unigram but Wi-1 is not in unigram
                elif (sentence[i+1] not in unigram) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #Wi-1 is in unigram but (Wi-2, Wi-1) is not in bigram
                elif ((sentence[i], sentence[i+1]) not in bigram) and (sentence[i+1] in unigram):
                    KN = unigram[sentence[i+1]] / float(Nt)

                #Wi is in unigram but (Wi-1, Wi) is not in bigram
                elif ((sentence[i+1], sentence[i+2]) not in bigram) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #Wi-1 is in unigram but (Wi-2, Wi-1) is not in N1+((Wi-2, Wi-1, *)
                elif ((sentence[i], sentence[i+1]) not in N2_) and (sentence[i+1] in unigram):
                    KN = unigram[sentence[i+1]] / float(Nt)

                #Wi is in unigram but (Wi-1, Wi) is not in N1+((Wi-1, Wi, *)
                elif ((sentence[i+1], sentence[i+2]) not in N2_) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #if Wi-2, Wi-1, Wi exist in unigram
                elif (sentence[i] in unigram) and (sentence[i+1] in unigram) and (sentence[i+2] in unigram):
                    KN = Kneser_Ney1(count, bigram[(sentence[i], sentence[i+1])], N2_[(sentence[i], sentence[i+1])], \
                                           unigram[sentence[i+1]], N1_[sentence[i+1]], N_1[sentence[i+2]], d, N00, \
                                           unigram[sentence[i+2]], Nt)                                    
                          
            else:
                #when trigram has count > 0
                KN = Kneser_Ney1(count, bigram[(sentence[i], sentence[i+1])], N2_[(sentence[i], sentence[i+1])], \
                                           unigram[sentence[i+1]], N1_[sentence[i+1]], N_1[sentence[i+2]], d, N00, \
                                           unigram[sentence[i+2]], Nt)
                             
                                   
                    
            pro_KN = pro_KN + np.log(KN)
            num_process = num_process + 1
            KNS.append(KN)

    print 
    print 'total log(p_kn) =', pro_KN
    print 'Perplexity      =', np.exp((-1 * pro_KN) / float(num_process))
        
    return KNS

#revised version of Kneser-Ney smoother: suggested by professor
def Kneser_Ney2(C210, C21_, N2_, N_2, N12, N11, N00, N_1, d, N1_, N_11, N000, C0, N):

    #C(Wi-2, Wi-1, Wi) > 0 and N1+(Wi-2, Wi-1, *) > 0
    if C210 > 0 and C21_ > 0:
        return (max(0.0, C210 - d) / C21_) + (d * N2_ / C21_ * (max(0.0,  N_2 - d) / N12 + (d * N11 * N_1 / N12 / N00)))

    #N1+(Wi-2, Wi-1, *) > 0 and C(Wi-2, Wi-1) > 0
    elif N2_ > 0 and C21_ > 0:
        return (d * N2_ / C21_ * (max(0.0,  N_2 - d) / N12 + (d * N11 * N_1 / N12 / N00)))

    #C(Wi-2, Wi-1) = 0
    #elif C21_ == 0:
     #   return (max(0.0, C21_ - d) / N1_) + (d * N_11 * N_1/ N1_ /N000)

    else:
        return C0 / float(N)


#compute probability of revised version of Kneser-Ney smoother: 
def get_KN2(testfile, unigram, bigram, trigram, N_1, N1_, N2_, N_2, N12, d): 

    N00 = sum(N_1.values())
    N000 = sum(unigram.values()) - unigram['<s>']
    Nb = sum(bigram.values()) - unigram['<s>']
    Nt = sum(trigram.values()) - unigram['<s>']
    
    pro_KN = 0.0
    num_process = 0
    KNS = []

    for sentence in testfile:

        for i, _ in enumerate(sentence, start = -2):

            if i == -2:
                continue
            if i == -1:
                continue

            count = trigram.get((sentence[i], sentence[i+1], sentence[i+2]), 0) 

            KN = 0

            #if C(Wi-2, Wi-1, Wi) = 0   
            if count == 0:

                #skip if there is no Wi
                if sentence[i+2] not in unigram:
                    continue

                #Wi is in unigram but Wi-1 is not in unigram
                elif (sentence[i+1] not in unigram) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #Wi-1 is in unigram but (Wi-2, Wi-1) is not in bigram
                elif ((sentence[i], sentence[i+1]) not in bigram) and (sentence[i+1] in unigram):
                    KN = unigram[sentence[i+1]] / float(Nt)

                #Wi is in unigram but (Wi-1, Wi) is not in bigram
                elif ((sentence[i+1], sentence[i+2]) not in bigram) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #Wi-1 is in unigram but (Wi-2, Wi-1) is not in N1+((Wi-2, Wi-1, *)
                elif ((sentence[i], sentence[i+1]) not in N2_) and (sentence[i+1] in unigram):
                    KN = unigram[sentence[i+1]] / float(Nt)

                #Wi is in unigram but (Wi-1, Wi) is not in N1+((Wi-1, Wi, *)
                elif ((sentence[i+1], sentence[i+2]) not in N2_) and (sentence[i+2] in unigram):
                    KN = unigram[sentence[i+2]] / float(Nt)

                #if Wi-2, Wi-1, Wi exist in unigram
                elif (sentence[i] in unigram) and (sentence[i+1] in unigram) and (sentence[i+2] in unigram):
                    KN = Kneser_Ney2(count, bigram[(sentence[i], sentence[i+1])], N2_[(sentence[i], sentence[i+1])], \
                                           N_2[(sentence[i+1], sentence[i+2])], N12[sentence[i+1]], bigram[(sentence[i+1], sentence[i+2])], \
                                           Nb, N_1[sentence[i+2]], d, unigram[sentence[i+1]], N1_[sentence[i+1]], N000, \
                                           unigram[sentence[i+2]], Nt)                                    
                          
            else:
                #when trigram has count > 0
                KN = Kneser_Ney2(count, bigram[(sentence[i], sentence[i+1])], N2_[(sentence[i], sentence[i+1])], \
                                       N_2[(sentence[i+1], sentence[i+2])], N12[sentence[i+1]], bigram[(sentence[i+1], sentence[i+2])], \
                                       Nb, N_1[sentence[i+2]], d, unigram[sentence[i+1]], N1_[sentence[i+1]], N000, \
                                       unigram[sentence[i+2]], Nt)
                             
                                   
                    
            pro_KN = pro_KN + np.log(KN)
            num_process = num_process + 1
            KNS.append(KN)

    print 
    print 'total log(p_kn) =', pro_KN
    print 'Perplexity      =',np.exp((-1 * pro_KN) / float(num_process))
    return KNS
    

if __name__=='__main__':
     
   print 'Please wait... computing probability of Kneser-Ney Smoothing'
   print
   unigram, bigram, trigram, N_1, N1_, N2_, N_2, N12 = get_ngrams(get_words('train'))
   test = get_words('test')
   print 'Result with original version of probability of bigram Kneser-Ney Smoothing: P_2(Wi|Wi-1)' 
   list1 = get_KN1(test, unigram, bigram, trigram, N_1, N1_, N2_, 0.5)
   print
   print '-------------------' * 5
   print '-------------------' * 5
   print
   print 'Result with revised version of probability of bigram Kneser-Ney Smoothing: P_2(Wi|Wi-1)'
   list2 = get_KN2(test, unigram, bigram, trigram, N_1, N1_, N2_, N_2, N12, 0.5)

   
   
    

    


