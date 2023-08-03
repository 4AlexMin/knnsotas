from prda.ml.neighbors import VariableKNN
from prda.ml.neighbors import construct_adjacency_matrix

class KNNG(VariableKNN):
    def __init__(self, k=10):
        super().__init__()
        self.k = k
    
    
    def fit(self, X, y):
        A = construct_adjacency_matrix(X, self.k)
        super().fit(X, y, A=A)

