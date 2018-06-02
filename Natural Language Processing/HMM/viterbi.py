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


#produce transformed data and transition probabilities
def p_data(f_n):
    
    data = reader(f_n)
    emission = probabilities(data)[0]
    transition = probabilities(data)[1]
    
    return data, emission, transition

#parse a single sentence line into observations and states
def parser(line):
    
    lines = line.strip().lower().split(" ")
    observations = lines[1::2]
    states = lines[2::2]

    return observations, states


#read all the sentences from a file
def reader(f_n):
    
    lines = []
    with open(f_n) as fout:
        for line in fout.readlines():
            lines.append(parser(line))

    return lines


#generate transition and emission probability for each word and state
def probabilities(data):

    #initials = defaultdict(float)    
    emissions = defaultdict(lambda: defaultdict(float))
    transitions = defaultdict(lambda: defaultdict(float))

    """
       emission probabilities:
       expressing the probability of an observation t being generated
       from a state i

    """   
    for words, states in data:
        for word, state in zip(words, states):
            emissions[word][state] += 1

    for vs1 in emissions.itervalues():
        total1 = sum(vs1.values())
        for v1 in vs1:
            vs1[v1] /= total1

    """
       transition probabilities:
       representing the probability of moving from state i to state j

    """   
    for _, states in data:
        for s1, s2 in zip(states, states[1:]):
            transitions[s1][s2] += 1
  
    for vs in transitions.itervalues():
        total2 = sum(vs.values())
        for v in vs:
            vs[v] /= total2
    
    return emissions, transitions



#compute emission probabilities
def emiss_prob(emissions, observation, state):
    
    if all([x == 0 for x in emissions[observation]]):
        return 0.01
    
    probability = emissions[observation][state]
    return probability

#compute transition probabilities 
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

    test, _, _ = p_data("test")
    _, emissions, transitions = p_data("train")

    correct = 0
    total = 0
     
    print "%d emission probabilities and %d states in the training data." % (len(emissions), len(transitions)) 
    print   

    for line, tag in test:
        prediction = HMM_viterbi(line, emissions, transitions)

        #add up the number of true states in the predicted array
        correct += sum([true == pre for true, pre in zip(tag, prediction)]); total += len(line)
        accuracy = correct * 100.0 / total

    print "===" * 9 + " result " + "===" * 9
    print 
    print "Number of observations(words) = %d." % total
    print "Number of correctly predicted states = %d." % correct
    print "Accuracy of this model is %d / %d = %f%%." % (correct, total, accuracy)
