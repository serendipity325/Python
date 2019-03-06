# KNN imputation model

The basic assumption of the KNN imputation method is that the missing value can be predicted as one of the closest possible values.
However, it is impossible to calculate Euclidean distances using data that contains missing entries, because each row has different number of missing values. Thus, to overcome this problem, a standardized distance calculation is used and the method is as follows:
1.	Any two rows or vectors are selected.
2.	Only columns with values in common between two rows are used in the distance calculation.
3.  The standardized distance can be calculated as the sum of squared differences between common points divided by the number of non-missing values that exist in common between the two vectors
4.  If two vectors have no features in common, then their distance is infinity. 
5.  After standardized distances between all points are calculated, I assigned a particular value as a maximum to infinities. 
6.  Estimate the missing point by computing the sum of product of each weight and value of a point of selected vectors. In this case, the weight is the inverse of the distance between the vector of the missing point and the selected closest vector.

Consider the following examples:
1.   IF x=(NA,NA,NA,NA) and y=(NA,NA,NA,NA), then d(x,y)= ∞.
2.   IF x=(1,NA,NA,NA)  and y=(NA,2,NA,NA), then there is no point in common between two vectors, so d(x,y)= ∞.
3.   IF x=(NA,NA,5,NA)  and y=(1,8,3,NA), then both vectors have values in the third point. Hence d(x,y)= (5-3)^2 / 1 = 4.
4.   IF x=(NA,NA,5,NA)  and z=(NA,2,4,NA), then both vectors have values in the third point. Hence d(x,z)= (5-4)^2 / 1 = 1.

* Let's try to replace the second point of x=(NA,NA,5,NA) using the above two examples, y=(1,8,3,NA) and z=(NA,2,4,NA). Since d(x,y) = 4 and d(x,z) = 1, weights of vector y and z are 1/4 and 1 respectively. Hence the estimated value of the second point of x is (8 x 1/4) + (2 x 1) =  4.
