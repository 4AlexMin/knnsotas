from prda.ml.neighbors import VariableKNN

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class LVKNN(VariableKNN):
    def __init__(self, k_min=5, k_max=15):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
    
    def fit(self, X, y):
        K = lv_knn_adaptivek(X, y, k_min=self.k_min, k_max=self.k_max)
        super().fit(X, y, K=K)
        


def lv_knn_adaptivek(X, y, k_min, k_max):
    n = X.shape[0]  # Number of data points
    k_values = np.zeros(n, dtype=int)  # Vector to store local k values
    
    # Step 1: Obtain cross-validation accuracy for global values of k
    global_acc = []
    for k in range(k_min, k_max + 1):
        # Create the KFold object
        k_folds = 10
        kf = KFold(n_splits=k_folds)
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=kf)
        global_acc.append(scores.mean())
    
    for i in range(n):
        x = X[i]
        
        # Step 2-4: Determine optimal local value of k for each data point
        local_acc = []
        for k in range(k_min, k_max + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            
            
            """
            To obtain the local k value associated with a prototype x_i , we only need to consider the instances, whose nearest neighbor is x_i.
            To avoid the negative effect of considering too few neighbors, to obtain the local value of k for every instance, we use the n-nearest neighbors instead of only just the nearest one. This has the effect of smoothing the estimation of the best k value. In our experiments, we used the value of n = 3. `The results are similar with other small values`.
            [1] N. Garcia-Pedrajas, J. A. Romero Del Castillo, and G. Cerruela-Garcia, “A Proposal for Local $k$ Values for $k$-Nearest Neighbor Rule,” IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 2, pp. 470–475, 2017, doi: 10.1109/TNNLS.2015.2506821.
            """
            # Leave-one-out estimation for the 3-nearest neighbors of the current data point
            n_nbrs = max(k+1, 10)    # In original paper, n-nearest neighbors is set to 3, instead of 10. I think it's unreasonable because `n_nbrs` should always be bigger than `k`.
            x_neighbors = np.argsort(np.linalg.norm(X - x, axis=1))[:n_nbrs]
            fold_acc = cross_val_score(knn, X[x_neighbors], y[x_neighbors], cv=LeaveOneOut())
            local_acc.append(fold_acc.mean())
            
        
        # Step 3: Obtain optimal local value of k
        eval_scores = np.array(local_acc) + np.array(global_acc)
        optimal_k = k_min + np.argmax(eval_scores)
        
        # Step 4: Assign optimal value of k to ki
        k_values[i] = optimal_k
    
    # Step 5: Return the vector of local k values
    return k_values
