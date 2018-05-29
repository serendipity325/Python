import numpy as np

# transform data
def get_data(dfile):
    
    lst = []

    with open(dfile) as lines:
        
        for line in lines:
            line = line.strip()
            point = [0,] * 120
            
            for tok in line.split()[1:]:
                a, _ = tok.split(":")
                a = int(a)
                point[0] = int(line.split()[0])
                point[a] = 1
            lst.append(point)
    
    return lst   
    
# classification function 
def sign(input, threshold = 0):
    
    return 1 if input > threshold else -1


# Obtain weights 'w' using perceptron algorithm
def perceptron(train, eta = 0.5):
 
    X = np.matrix(train)[:,1:] # input data set
    y = np.matrix(train)[:,0] # target data set                
    w = np.zeros((X.shape[1], 1)) # initialize weight vector
    
    # initialize weight vector
    for i in range(X.shape[0]):
        pred_val = sign(X[i] * w)
        if i % 20 == 0:
            print "%d..." % i,
            sys.stdout.flush()

        for j in range(w.shape[0]):
            w[j] += eta * (sign(y[i]) - pred_val) * X[i, j] 
            
    return w
    
# classify test dataset using trained wights and get accuracy
def prediction(test, w):
    
    X = np.matrix(test)[:,1:] # input data set
    y = np.matrix(test)[:,0] # target data set
    
    correct = 0
    pred = []
    for i in range(y.shape[0]):
        pred_val = sign(X[i] * w)
        pred.append(pred_val)
        
        if pred_val == y[i]:
            correct += 1 # represent number of correct cases
    
    return pred, str(correct/float(y.shape[0]) * 100) + '%'  


