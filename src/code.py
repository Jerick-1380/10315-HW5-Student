import numpy as np
from .utils import *

def consistent_scale_eigenvectors(V):
    """ Scale the columns of V such that everyone uses a consistent
        set of eigenvectors.

        Input:
        V: numpy ndarray each **column** is an eigenvector

        Returns V. V is modified in place and also returned.

        Implementation based on code from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py
    """
    max_abs_cols = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_cols, range(V.shape[1])])
    V *= signs
    return V

def pca(X, K):
    """ Return the PCA projection of X onto K-dimensions
        X: numpy ndarray of shape (N, M) where each row represents a data point
        K: integer representing the desired number of output dimensions

        Returns a numpy ndarray of shape (N, K)

        Don't forget to center your data first
        Use eigenvectors based on np.linalg.eig or np.linalg.svd.
        We suggested using np.linalg.svd because it returns sorted arrays.

        NOTE: In order to make the autograder happy, you must pass your eigenvectors
        into the consisten_scale_eigenvectors function above before applying them 
        to your data!
    """
    #Student Solution HERE
    return 0.0


def kmeans(X, K):
    """ Cluster data X into K converged clusters.
    
        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is a cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X

        The algorithm should terminate if there are no changes in labels.
        MAKE SURE TO INITIALIZE CLUSTERS TO BE THE FIRST K POINTS!!!
    """
    #Student Solution HERE
    return 0.0, 0.0


def boxcar(x, z, width):
    """ Return 1 if the Eudclidean distance between input vectors is
        less than or equal to width/2, and return zero otherwise.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        width: float value of hyperparameter
        
        Returns: float value after appying kernel to x and z
    """
    #Student Solution HERE
    return 0.0

def linear(x, z):
    """ Return the result of simply taking the dot product of the two 
        input vectors.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        
        Returns: float value after appying kernel to x and z
     """
    #Student Solution HERE
    return 0.0

def rbf(x, z, gamma):
    """ Return the result of applying the radial basis function kernel 
        on the two input vectors, given the hyperparameter gamma.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        gamma: float value of hyperparameter
        
        Returns: float value after appying kernel to x and z
    """
    #Student Solution HERE
    return 0.0
      
def polynomial(x, z, d):
    """ Return the result of applying the polynomial kernel of degree 
        up to d on the two input vectors.
        K(x, z) = (x^Tz+1)^d

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        
        Returns: float value after appying kernel to x and z
    """
    #Student Solution HERE
    return 0.0
    
