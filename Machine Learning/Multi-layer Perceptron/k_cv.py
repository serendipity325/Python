import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.layers import Dense, Activation, Embedding, Flatten, LSTM
from keras.wrappers.scikit_learn import KerasClassifier


"""
K-Fold cross-validation
"""


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
    
def create_model():
    
    model = Sequential()
    model.add(Dense(n, input_dim=25, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
if __name__ == "__main__":
    
    data1 = pd.read_csv('data.csv')
    data1 = data1.replace('?', np.NaN)

    df1 = proprocessing(data1)
    n_classes = len(set(df1.NODE_MINUTES))

    df1 = cat_converter(df1)
    y_kf = df1.NODE_MINUTES
    y_kf = one_hot_converter(y_kf)

    df_kf = df1.copy()
    df_kf = df_kf.values
    X_kf = df_kf[:, 1:]
    

    seed = 100
    np.random.seed(seed)
    n_splits = 3

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    k2 = [] ; k3 = []
    for train, test in kf.split(X_kf, y_kf):
        k2.append(train)
        k3.append(test)    

    cvscores = []
    for i in range(n_splits):

        y_train_kf = y_kf[k2[i]]
        y_test_kf = y_kf[k3[i]]

        X_train_kf = scaler.transform(X_kf[k2[i]])
        X_test_kf = scaler.transform(X_kf[k3[i]])

        model = create_model()
        model.fit(X_train_kf, y_train_kf, epochs=100, batch_size=512)
        score = model.evaluate(X_test_kf, y_test_kf, batch_size = 512)   
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        cvscores.append(score[1] * 100)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) 
