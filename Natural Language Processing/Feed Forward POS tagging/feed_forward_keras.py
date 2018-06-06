from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
import numpy as np



#import data and seperate words and tags
def reader(filename):
    
    words = []; tags = []

    with open(filename) as fin:
	for line in fin:
	    line = line.strip().split(' ')			
			
	    word = line[1::2]
	    tag = line[2::2]
			
	    words.append(word)
	    tags.append(tag)
	             
    return words, tags

#convert word to numbers
def converter(data, dct):

    for i in range(len(data)):
        for j in range(len(data[i])):
	    data[i][j] = dct[data[i][j]]


#flatten list of lists and merge two falttened-lists
def flatten_merge(X, Y):
    
    total_X = [element for list in X for element in list]
    total_Y = [element for list in Y for element in list]

    return total_X + total_Y 


#create word/tag to index dicts
def index_dicts(X, Y):
	
    w2idx = {}
        
    for i in range(len(X)):
	
        #key is i-th element of X and value is i 
        w2idx[X[i]] = i
    
    t2idx = {}

    for j in range(len(Y)):
	
        #key is j-th element of Y and value is j 
        t2idx[Y[j]] = j
        
    return w2idx, t2idx


#generate inputs as a five word window centered on the current word
def inputs(Xs, ts, total_tags):
  
    tags = []; tags_num = []; input_5 = []   
       
    for k in range(len(Xs)):
		
	word = Xs[k]; tag = ts[k]		
	
	#if the length of word is less than 5, continue
	if len(word) < 5:
	    continue

	for i in range(len(word) - 5):
	    
            #indicator of centered words of five word windows
            #corresponding centered word is 1 and others are 0's
            tag_zeros = [0] * len(total_tags)
            tag_zeros[tag[i + 2]] = 1
            tags.append(tag_zeros)
            
            #five word windows
	    input_5.append(word[i : i + 5])
                             
    return np.array(tags), np.array(input_5) 
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
 
if __name__ == "__main__":

    train_w, train_t = reader('train')
    test_w , test_t = reader('test')    
    
    #All non-overlapping elements of words and tags
    total_words = list(set(flatten_merge(train_w, test_w)))
    total_tags = list(set(flatten_merge(train_t, test_t)))
    
    lw = len(total_words)
    lt = len(total_tags)

    widx, tidx = index_dicts(total_words, total_tags)
      
    #convert each word and tag to its index number
    converter(train_w , widx)
    converter(test_w , widx)
    converter(train_t , tidx)
    converter(test_t , tidx)       
    
    train_indicator, train_input = inputs(train_w, train_t, total_tags)
    test_indicator, test_input = inputs(test_w, test_t, total_tags)
    
      
    model1 = Sequential()
    model1.add(Embedding(lw, 100, input_length=5))
    model1.add(Flatten())
    model1.add(Dense(100))
    model1.add(Dense(lt, activation = 'sigmoid'))
    #model1.add(Activation('softmax'))
    model1.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    model1.fit(train_input, train_indicator, epochs = 3 , batch_size = 512)
    score1 = model1.evaluate(test_input, test_indicator, batch_size = 512)
    

    model2 = Sequential()
    model2.add(Embedding(lw, 100, input_length=5))
    model2.add(Flatten())
    model2.add(Dense(100))
    model2.add(Dense(lt, activation = 'sigmoid'))
    model2.add(Activation('softmax'))
    model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model2.fit(train_input, train_indicator, epochs = 3 , batch_size = 512)
    score2 = model2.evaluate(test_input, test_indicator, batch_size = 512)
    
 
    
    model3 = Sequential()
    model3.add(Embedding(lw, 100, input_length=5))
    model3.add(Flatten())
    model3.add(Dense(100))
    model3.add(Dense(lt, activation = 'sigmoid'))
    model3.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model3.fit(train_input, train_indicator, epochs = 3 , batch_size = 512)
    score3 = model3.evaluate(test_input, test_indicator, batch_size = 16)


    
    model4 = Sequential()
    model4.add(Embedding(lw, 100, input_length=5))
    model4.add(Flatten())
    model4.add(Dense(100))
    model4.add(Dense(lt, activation = 'sigmoid'))
    model4.compile(loss='mean_squared_error', optimizer='sgd',metrics=['accuracy'])
    model4.fit(train_input, train_indicator, epochs = 3 , batch_size = 512)
    score4 = model4.evaluate(test_input, test_indicator, batch_size = 512)
    
 
     
    print "Test loss and accuracy with loss: binary_crossentropy',optimizer: sgd"
    print score1    

    print "Test loss and accuracy with loss: binary_crossentropy',optimizer: adam"
    print score2

    print "Test loss and accuracy with loss: mean_squared_error, optimizer: adam"
    print score3

    print "Test loss and accuracy with loss: mean_squared_error, optimizer: sgd"
    print score4
 

