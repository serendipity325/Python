import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# KNN imputation model

# compute mean squared difference between two vectors
# because some points of vectors are missing
def normalized_dist(X):
    
    a, b = X.shape

    # create matrix of infinity and dimension is 'number of rows of X' by 'number of rows of X'
    D = np.ones((a, a), dtype = "float32") * np.inf

    # assign 1 to non-missing values and 0 to missing values
    observed_points = (~np.isnan(X)).astype(int)

    # create matrix that each element is number of non missing values of each vector
    non_miss = np.dot(observed_points, observed_points.T)

    # assign True to non-zeros and 0 to zeros
    n_miss_vars = (non_miss == 0)

    # find rows that not compatible with other rows
    rows_n_miss = n_miss_vars.sum(axis=1)

    # True if element is zero False otherwise
    rows_n_miss_bool = (rows_n_miss == 0)

    # True if element is 'number of rows of X' False otherwise
    rows_0_miss_bool = rows_n_miss == a

    # rows of positions that can be used 
    true_positions = ~rows_0_miss_bool

    # indices of valid positions
    position_idx = np.where(true_positions)[0]

    # create matrix of zeros and dimension is same of dimension of X
    X0 = np.zeros_like(X)

    # convert with True/False
    X0_bool = np.zeros_like(X0, dtype=bool)

    # create array of Falses and length is number of rows of X
    X1 = np.zeros(a, dtype=bool)

    # create array of zeros and length is number of rows of X
    X3 = np.zeros(a, dtype=X.dtype)

    # matrix of mean squared difference between samples
    for k, i in enumerate(position_idx):

        # X - X[i]
        X0 = np.subtract(X, X[i])
        X0_bool = np.isnan(X0)

        # assign zeros to all NA's
        X0[X0_bool] = 0

        # square each difference
        X0 **= 2

        observed_counts_i = non_miss[i]

        if rows_n_miss_bool[i]:

            # add up all the non-missing squared differences
            D[i] = X0.sum(axis=1)
            D[i] /= observed_counts_i

        else:

            # convert True->False and False->True
            X1 = np.logical_not(n_miss_vars[i])

            # add up all the non-missing squared differences
            X3 = X0.sum(axis=1)
            X3[X1] /= observed_counts_i[X1]
            D[i, X1] = X3[X1]

        if k % 20 == 0:
            print ("%d..." % k, sys.stdout.flush())

    return D


def knn(X, missing_points, k, print_interval=100):
    
    D = normalized_dist(X)
    D_finite_flat = D[np.isfinite(D)]

    # set minimum distance
    min_dist=1e-6
    # set maximum distance
    max_dist = 1e6 * D_finite_flat.max()

    # set diagonal of distance matrix to a large value since we don't want
    # points considering themselves as neighbors
    np.fill_diagonal(D, max_dist)
    # prevents 0s
    D[D < min_dist] = min_dist
    # prevents infinities
    D[D > max_dist] = max_dist  

    a, b = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    observed_points = ~missing_points
    X1 = X.copy()

    #D, max_dist = dist_matrix(X, missing_points)
    # get rid of infinities, replace them with a very large number

    sotedD = np.argsort(D, axis=1)
    D_inversed = 1.0 / D
    D_masking = D < max_dist
    row_dists = D_masking.sum(axis=1)

    # trim the number of other rows we consider to exclude those
    # with infinite distances
    sotedD = [sotedD[i, :c] for i, c in enumerate(row_dists)]

    for i in range(a):
        
        miss_rows = missing_points[i]
        miss_idx = np.where(miss_rows)[0]
        
        row_weights = D_inversed[i]
        poss_neighbor_idx = sotedD[i]

        for j in miss_idx:
            
            # observed_points
            observed = observed_points[:, j]
            # sort observed points by indices of possible neighbors
            sorted_observed = observed[poss_neighbor_idx]
            # observed neighbor indices
            obs_neighbor_idx = poss_neighbor_idx[sorted_observed]
            # k nearest indices
            KN_idx = obs_neighbor_idx[:k]
            # each k nearest points weights
            weights = row_weights[KN_idx]
            # sum of weights
            weight_sum = weights.sum()

            if weight_sum > 0:

                column = X1[:, j]
                values = column[KN_idx]
                X[i, j] = np.dot(values, weights) / weight_sum

        if i % 20 == 0:
            print ("%d..." % i, sys.stdout.flush())
            
 
    return X


if __name__ == '__main__':
        
        # import data using pandas
        df = pd.read_csv('test.csv')

        # drop first 6 columns
        df = df.drop(df.columns[0:6], axis=1)
        col_names = df.columns
          
        X_original = df.values.astype(float)         
        
        missing_points_original = np.isnan(X_original)
  
        X_result_original = knn(X_original, missing_points_original, k = 10)

        result = pd.DataFrame(X_result_original, columns = col_names)
        result.to_csv('imputed_test.csv')
        print(result)

        
