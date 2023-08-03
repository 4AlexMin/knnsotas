from prda.ml.neighbors import VariableKNN
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class centeredKNNG(VariableKNN):
    def __init__(self, k=10):
        super().__init__()
        self.k = k
    
    def fit(self, X, y):
        A = compute_centered_knn_graph(X, self.k)
        super().fit(X, y, A=A)




def compute_centered_knn_graph(X, k):
    n = X.shape[0]  # Number of samples
    
    # Compute the pairwise Euclidean distances between samples
    distances = euclidean_distances(X)
    
    # Compute the similarity matrix
    similarity_matrix = np.exp(-distances ** 2)  # Example similarity measure: Gaussian kernel
    
    # Center the similarity matrix
    transformation_matrix = np.eye(n) - (1/n) * np.dot(np.ones(n).reshape(-1,1), np.ones(n).reshape(1,-1))
    centered_similarity_matrix = np.dot(np.dot(transformation_matrix, similarity_matrix), transformation_matrix)
    
    # Compute the k nearest neighbors for each sample
    kNN_indices = np.argsort(-centered_similarity_matrix)[:, 1:k+1]  # Exclude self-neighbors
    
    # Generate the centered kNN graph
    graph = np.zeros((n, n), dtype=int)
    for i in range(n):
        graph[i, kNN_indices[i]] = 1
    
    return graph
