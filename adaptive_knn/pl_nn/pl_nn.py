# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:27:22 2020

@author: DANILO
"""

import numpy as np

import warnings
warnings.filterwarnings('ignore')

'''
X,y = make_classification(n_samples=50,
                             n_features=2,
                             n_classes=3,
                             n_repeated=0,
                             n_informative=2,
                             n_clusters_per_class=1,
                             class_sep=1.0,
                             n_redundant=0,random_state=42)
'''

class PlNearestNeighbors:
    def __init__(self, fake=None):
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.centers = None
        self.k = None
        self.nearest_neighbors = None
        self.__fake = fake
    
    def get_params(self, deep=True):
        return {
            "fake": self.__fake,    # Just make this algorithm agree with prda::ml.evaluate_param_combinations
            }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def __get_angle_between_three_points(self, pointA, pointB, pointC):
        '''
        Function that calculates the angle between three points.
          A
          |
          |
          |_
        B |.|___________C
        
        theta((pointB,pointA),(pointB,pointC)
        
        Code adapted from:
        https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    
        Parameters
        ----------
        pointA : array
            A n-dimensional array that represents the ending point of (pointB,pointA).
        pointB : array
            A n-dimensional array that represents the reference point of the angle between the two vectors.
        pointC : array
            A n-dimensional array that represents the ending point of (pointB,pointC).
    
        Returns
        -------
        float
            Angle in degrees.
    
        '''
        
        ba = pointA - pointB
        bc = pointC - pointB
        
        try:
            cosine_angle = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
        except:
            cosine_angle = 0
        
        return np.degrees(cosine_angle)
    
    def __get_distances(self, X, Y, check_same_idx=True):
        '''
        Function that computes the distance matrix from the samples in X and Y.

        Parameters
        ----------
        X : array
            A MxN array.
        Y : array
            A KxZ array.
        check_same_idx : bool, optional
            It indicates whether to ignore the distance of the elements in the same index in X and Y.
            If True, diagonal of the distance matrix is assigned zero. The default is True.

        Returns
        -------
        distances : array
            A MxK array with the distance between each element from X to all elements of Y.

        '''
        
        distances = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(X.shape[0]):
            p = X[i]
            for j in range(Y.shape[0]):
                if (check_same_idx and i == j):
                    continue
                
                #ed = np.linalg.norm(Y[j]-p)                
                ed = np.sum(np.abs(Y[j]-p))
                distances[i][j] = ed
        
        return distances
    
    def __get_geometric_median(self, X):
        '''
        Function that seeks for the sample whose distance to all the others in dataset is the lowest.
        The sample is considered to be center of the instances in the dataset.

        Parameters
        ----------
        X : array
            A MxN dimensional array with the samples.

        Returns
        -------
        center : array
            A 1xN dimensinal array that represents the center of the dataset.

        '''
        distances = self.__get_distances(X, X)
        
        min_dist = np.sum(distances[0])
        idx = 0
        
        for i in range(1, len(distances)):
            if (np.sum(distances[i]) < min_dist):
                min_dist = np.sum(distances[i])
                idx = i
        
        return X[idx]
            
    def fit(self, X, y):
        '''
        Function that computes the center of the classes and the weights of the samples.

        Parameters
        ----------
        X : array
            A MxN dimensional array with the samples of the training set.
        y : array
            A Mx1 dimensional array with the labels of each sample in X.

        Returns
        -------
        None.
        '''
        
        self.X_train = np.copy(X)
        
        # Adding to more columns to the X_train array to store the labels (y) and the weights of each training sample to the center of the class, respectively
        self.X_train = np.append(self.X_train,np.zeros((X.shape[0],2)),axis=1)
        self.X_train[:,-2] = np.copy(y)
        
        # Auxiliary variables to store the classes and their respective centers
        classes = np.unique(y)
        centers = []
        
        # Auxiliary variable to store the weigths of the samples (with respect to the class centers)
        w = []
        
        for c in classes:
            indices = np.where(y==c)[0]
            X_ = X[indices,:]
            
            # Getting the geometric median
            #center = self.__get_geometric_median(X_)
            #center = np.mean(X_,axis=0)
            center = np.median(X_,axis=0)
            centers.append(center)
            
            # Getting the weights of each sample
            w = []
            for s in X_:
                w.append(1 / (np.linalg.norm(s-center)+0.0001))
                # try:
                #     w.append(1 / math.sqrt((s[0]-center[0])**2 + (s[1]-center[1])**2))
                # except:
                #     w.append(1)
            
            # Adding the class weights to the last column of the X_train array
            self.X_train[indices,-1] = np.array(w)
        
        self.centers = np.vstack(centers)
        self.classes = classes
    
    def predict(self, X):
        '''
        Function that predicts the labels of each sample in test set X.

        Parameters
        ----------
        X : array
            A MxN array with the samples of the test set.

        Returns
        -------
        y_pred : array
            a Mx1 array with the predicted labels.
        '''
        y_pred = []
        
        i = 0
        for t in X:
            t = np.expand_dims(t,axis=0)
            
            # Distance of the test sample to all centers of class
            distances_center = self.__get_distances(t,self.centers,False).flatten()
            
            # Class center with the minimum distance to the test sample
            center_min = self.centers[np.argmin(distances_center)]
            
            # Distance of the test sample to all instances of the training set
            distances = self.__get_distances(t,self.X_train[:,:-2],False).flatten()
            
            # Getting the minimum distance (test sample to the class centers)
            ed = distances_center[np.argmin(distances_center)]
            
            # Getting the nearest neighbors, i.e., all training instances whose distances are less than the distances
            idx_min = np.where(distances <= ed)
            nearest_neighbors = self.X_train[idx_min]
            
            # Calculating the angle between the test sample and the nearest neighbors
            angles = []
            for n in nearest_neighbors:
                angles.append(self.__get_angle_between_three_points(center_min, t[0], n[:-2]))
            
            angles = np.nan_to_num(angles)
            
            # Concatenating the distances of the test sample to the nearest neighbors of the training set
            nearest_neighbors = np.concatenate((nearest_neighbors,distances[idx_min].reshape(-1,1)),axis=1)
            
            # Getting the neighbors inside the semi-circle
            nearest_neighbors = nearest_neighbors[np.abs(angles) <= 90]
            
            # Determining the final class based on the nearest neighbors
            if (len(nearest_neighbors) == 0):
                final_class = self.classes[np.argmin(distances_center)]
            elif (len(np.unique(nearest_neighbors[:,-3])) == 1):
                final_class = np.unique(nearest_neighbors[:,-3]).astype(int)
            else:
                # Finding the classes of the nearest neighbors                
                cl = np.unique(nearest_neighbors[:,-3]).astype(int)
                classes = dict.fromkeys(cl,0)
                
                # Weighted sum considering the considering the distances and weights of the training instances
                for n in nearest_neighbors:
                    c = int(n[-3])
                    classes[c]+=(1/n[-1]) * n[-2]
                
                final_class = max(classes,key=lambda x : classes[x])
            
            y_pred.append(final_class)
        
        y_pred = np.vstack(y_pred)
        return y_pred