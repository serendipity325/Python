import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Activation, Embedding, Flatten, LSTM
from keras.wrappers.scikit_learn import KerasClassifier

def proprocessing(data):
    """
    1. Impute missing values of ATTR03 and ATTR05 features.
    2. Remove ID, EXECUTIONSTART, ATTR01, ATTR02, ATTR04, ATTR07 features.
    3. Drop features that have only 0.
    """
    
    data = data.drop(['ATTR01', 'ATTR02', 'ATTR04', 'ATTR07'], axis = 1)
    X = data.copy()
    X = X.sort_values(by=['ID'])
    X = X.reset_index(drop=True)

    df1 = Imputer(X, 'ATTR03')
    df1 = df1.drop(['ID', 'EXECUTIONSTART', 'SCNAME'], axis = 1)

    df2 = Imputer(X, 'ATTR05')
    df2 = df2.ATTR05

    df = pd.concat([df1, df2], axis = 1)
    df['SCNAME'] = X.SCNAME

    for column in df.columns:
        if (df[column] != 0).sum() == 0:
            df = df.drop([column], axis=1)
            
    return df


def Imputer(X, column):
    """
    Replace missing values of a feature by Decision Tree algorithm.
        
    """
    
    X_test = X[X[column].isnull()]
    X_train = X[X[column].notnull()]

    y_train = X_train[column]

    X_test = X_test.drop(['ATTR03', 'ATTR05'], axis = 1)
    X_test = string_converter(X_test)
    X_train = X_train.drop(['ATTR03', 'ATTR05'], axis = 1)
    X_train = string_converter(X_train)
    
    clf = DecisionTreeClassifier(max_depth=5)   
    clf.fit(X_train, y_train)
    predicted_feature = pd.DataFrame(clf.predict(X_test))
    predicted_feature.columns = [column]    
     
    X_test = X_test.reset_index(drop=True)
    X_test = pd.concat([X_test, predicted_feature], axis = 1)
    
    X_train = pd.concat([X_train, y_train], axis = 1)
    
    newX = pd.concat([X_train, X_test])    
    imputed_X = newX.sort_values(by=['ID'])    
    imputed_X = imputed_X.reset_index(drop=True)
                               
    return imputed_X   


def string_converter(X):
    """
    Change categorical labels with some integers.
        
    """
    
    encoder = LabelEncoder()    
    
    for column in X.select_dtypes(include=['object']):
        X[column] = encoder.fit_transform(X[column])
        
    return X


def cat_converter(df):
    """
    1. Create new columns with converted numbers corresponding to categorical labels of categorical features.
    2. Drop original categorical features.
    """
    
    categoricals = df.columns[df.dtypes == object]    
    
    for column in categoricals:    
        df[column] = pd.Categorical(df[column])
        new_column = column + '_new'
        df[new_column] = df[column].cat.codes
        df = df.drop([column], axis = 1)
    
    return df


def one_hot_converter(column):
    """
    Generate one-hot encoded class vector.
    
    """
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(column)    
    encoded_ = encoder.transform(column)
    # convert integers to dummy variables, i.e., one-hot encoded
    encoded_column = to_categorical(encoded_)
    
    return encoded_column


if __name__ == "__main__":

    data = pd.read_csv('data.csv')
    data = data.replace('?', np.NaN)

    df = proprocessing(data)
    n_classes = len(set(df.NODE_MINUTES))

    df = cat_converter(df)
    y = one_hot_converter(df.NODE_MINUTES)
    df = df.values
    X = df[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    n, p = X_train.shape

    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = Sequential()
    model.add(Dense(n, input_dim=25, activation='relu'))
    model.add(Dense(256, activation='relu'))
        
    """
    Use 'softmax' and 'categorical_crossentropy' instead of 'sigmoid' and 'binary_crossentropy'. 
    By taking the latter two, we're treating the problem as a multi-label problem, not a multi-class problem.
    
    *for categorical_crossentropy: softmax activation, one-hot encoded target
    *for binary_crossentropy: sigmoid activation, scalar target 
    
    However, even if this case is a multi-label case, a combination of 'sigmoid', 'binary_crossentropy', and 'rmsprop'
    produces the best accuracy of prediction. However, if we generate predicted classes using the combination, 
    we could get just one label. 
    
    """
    
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=512)
    score = model.evaluate(X_test, y_test, batch_size = 512)    
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))   
    
    y_classes = model.predict_classes(X_test) 
    print(y_classes)
