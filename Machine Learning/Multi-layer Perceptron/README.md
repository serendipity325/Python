# Multi-layer Perceptron

Classify categorical data using Keras.

1. Preprocessing data and Imputation

    Some features/variables are dropped since their values are only/almost missing values.
    Furthermore, if variables have only one number or character, then they are also deleted.

    After that, I imputed remaining missing values using Decision Tree algorithm.
    Finally, I scaled input values because common activation functions of the network’s neurons 
    such as tanh or sigmoid are defined on the [-1, 1] or [0, 1] interval respectively.
    Scaling was accomplished using sklearn’s MinMaxScaler.


2. Multi-layer Perceptron

    I used Multi-layer perceptron to classify the target variable of test data. 
    I converted the target values as binary vectors using one-hot encoding.


3. KFold Cross Validation

    I evaluated the performance of my deep learning models using Keras and Scikit-Learn. 
    KFold CV provides a robust estimate of the performance of a model on unseen data. 
    It does this by splitting the training dataset into k subsets and takes turns training models on all subsets 
    except one which is held out, and evaluating model performance on the held out validation dataset. 
    The process is repeated until all subsets are given an opportunity to be the held out validation set. 
    The performance measure is then averaged across all models that are created. 

    However, Cross validation is often not used for evaluating deep learning models due to the greater computational expense. 
    Hence please do not exceed 10 folds to evaluate deep learning models. 


4. Hyper parameters tuning

    I used a grid search to evaluate different configurations for my neural network models and 
    report on the combination that provides the best-estimated performance.
    The create_model() function is defined to take two arguments optimizer and init, both of which must have default values. 
    This will allow us to evaluate the effect of using different optimization algorithms and weight initialization schemes for the network.

    Below are descriptions of  hyper parameters:

      * Optimizers for searching different weight values.
      * Initializers for preparing the network weights using different schemes.
      * Epochs for training the model for a different number of exposures to the training dataset.
      * Batches for varying the number of samples before a weight update.
