from collections import defaultdict

#HMM-viterbi decoding algorithm
def HMM_viterbi(line, emissions, transitions):

    n_l = len(line)
    
    states = transitions.keys()
    n_s = len(states)
    
    viterbi = [[-1] * n_s for x in range(n_l)]
    prev = [[0] * n_s for x in range(n_l)]
   
    for i in range(n_l):

        for j in range(n_s):
            if i > 0:

                for k in range(n_s):
                    result = (viterbi[i - 1][k] *
                              trans_prob(transitions, states[k], states[j]) *
                              emiss_prob(emissions, line[i], states[j]))
                             

                    if result > viterbi[i][j]:
                        viterbi[i][j] = result
                        prev[i][j] = k

            else:
                viterbi[i][j] = emiss_prob(emissions, line[i], states[j])

    max_index = viterbi[-1].index(max(viterbi[-1]))

    return [states[x] for x in trellis(prev, max_index)]

#parse sentence lines into observations and states
def reader(filename):
    
    observations = []
    states = []

    with open(filename) as fout:
        for line in fout:
            line = line.strip().lower().split(" ")
            observations.append(line[1::2])
            states.append(line[2::2])
            
    return observations, states

#generate dictionaries for transition and emission for each word and state
def get_dict(observations, tags):

    words = [obs for observation in observations for obs in observation]
    states = [sts for state in tags for sts in state]

    emissions = defaultdict(lambda: defaultdict(float))
    transitions = defaultdict(lambda: defaultdict(float))

    for word, state in zip(words, states):
        emissions[word][state] += 1

    for s1, s2 in zip(states, states[1:]):
        transitions[s1][s2] += 1
        
    return get_proportions(emissions), get_proportions(transitions)
        
        
def get_proportions(dic):
    
    for vs in dic.values():
        sum_values = sum(vs.values())
        for v in vs:
            vs[v] /= sum_values
            
    return dic

#compute each emission probability
def emiss_prob(emissions, observation, state):
    
    if all([x == 0 for x in emissions[observation]]):
        return 0.01

    probability = emissions[observation][state]
    return probability

#compute each transition probability 
def trans_prob(transitions, prev, curr):
    
    probability = transitions[prev][curr]
    return probability


#generate the trail of indices that ends with that index given previous viterbi / initial index 
def trellis(prev, end):
    
    values = [end]
    curr = end
    for i in range(len(prev) - 1, 0, -1):
        curr = prev[i][curr]
        values.insert(0, curr)

    return values





if __name__ == "__main__":

    print "Start decoding."

    test_obs, test_sts = reader('test')
    train_obs, train_sts = reader('train')
    
    emissions, transitions = get_dict(train_obs, train_sts)
     

    correct = 0
    total = 0
     
    print "%d emission probabilities and %d states in the training data." % (len(emissions), len(transitions)) 
    print   

    for line, tag in zip(test_obs, test_sts):
        prediction = HMM_viterbi(line, emissions, transitions)

        #add up the number of true states in the predicted array
        correct += sum([true == pre for true, pre in zip(tag, prediction)]); total += len(line)
        accuracy = correct * 100.0 / total

    print "===" * 9 + " result " + "===" * 9
    print 
    print "Number of observations(words) = %d." % total
    print "Number of correctly predicted states = %d." % correct
    print "Accuracy of this model is %d / %d = %f%%." % (correct, total, accuracy)
    
    
