import numpy as np

from prda.ml.neighbors import VariableKNN
from prda.ml.neighbors import construct_adjacency_matrix

class MKNNG(VariableKNN):
    def __init__(self, k=20):
        super().__init__()
        self.k = k
    
    
    def fit(self, X, y):
        A = construct_adjacency_matrix(X, self.k)
        mutual_A = np.multiply(A, A.T)
        super().fit(X, y, A=mutual_A)




