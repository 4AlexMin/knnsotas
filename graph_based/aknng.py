import numpy as np
from prda.ml.neighbors import VariableKNN

class AKNNG(VariableKNN):
    def __init__(self, k=10, sigma=1, method='AKNNG'):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.method = method
    

    def fit(self, X, y):
        if self.method.lower() == 'aknng':
            S = AKNNG_similarity_matrix(X, self.k, self.sigma)
        else:
            S = MAKNNG_similarity_matrix(X, self.k, self.sigma)
        super().fit(X, y, S=S)



def AKNNG_similarity_matrix(X, k, sigma):
    U = __get_norm_dist_matrix(X, k, sigma)
    S0 = (U + U.T) / 2
    
    return S0


def MAKNNG_similarity_matrix(X, k, sigma):
    U = __get_norm_dist_matrix(X, k, sigma)
    U[(U - U.T) == U] = 0
    S0 = (U + U.T) / 2
    return S0


def __get_norm_dist_matrix(X, k, sigma):
    """

    Returns
    -------
        The partial similarity matrix
    """
    X = X.astype(np.float64)
    num, _ = X.shape
    
    # Graph construction
    distX = np.sum((X[:, np.newaxis] - X) ** 2, axis=2)  # Squared Euclidean distance
    distX1 = np.sort(distX, axis=1)
    idx = np.argsort(distX, axis=1)
    
    U = np.zeros((num, num))
    for i in range(num):
        U[i, idx[i, 1:k+1]] = 1 - ((k-1-sigma) * distX1[i, 1:k+1]) / np.sum(distX1[i, 1:k+1])
    
    U[U < 0] = 0
    return U



