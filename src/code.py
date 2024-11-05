import numpy as np
from .utils import *

def consistent_scale_eigenvectors(V):
    """ Scale the columns of V such that everyone uses a consistent
        set of eigenvectors.

        Input:
        V: numpy ndarray each **column** is an eigenvector

        Returns V. V is modified in place and also returned.

        Implementation based on code from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py::svd_flip
    """
    max_abs_cols = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_cols, range(V.shape[1])])
    V *= signs
    return V

def pca(X, K):
    """ Return the PCA projection of X onto K-dimensions
        X: numpy ndarray of shape (N, M) where each row represents a data point
        K: integer representing the desired number of output dimensions

        Don't forget to center your data first
        Use eigenvectors based on np.linalg.eig or np.linalg.svd.
        We suggested using np.linalg.svd because it returns sorted arrays.

        NOTE: In order to make the autograder happy, you must pass your eigenvectors
        into the consisten_scale_eigenvectors function above before applying them 
        to your data!

        Returns a numpy ndarray of shape (N, K)
    """
    X_center = X - np.mean(X,axis=0)
    _, _, V = np.linalg.svd(X_center)
    V = V.T
    top_V = V[:,:K]
    consistent_scale_eigenvectors(top_V)
    return X_center @ top_V


def kmeans(X, K):
    """ Cluster data X into K converged clusters.
    
        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is a cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X
    """
    N = X.shape[0]
    M = X.shape[1]

    # Initialize cluster centers to the first K points of X
    C = np.copy(X[:K])

    # Initialize z temporarily to all -1 values
    z = -1*np.ones(N, dtype=np.int32)

    # write your solution here

    while(True):
      Cnew = C.reshape(K, 1, -1)
      distances = np.sum(np.square(X-Cnew), axis = 2)
      newZ = np.argmin(distances, axis=0)
      if(np.array_equal(newZ,z)):
        break

      z = newZ

      for i in range(K):
        C[i] = X[newZ == i].mean(axis=0)

    return (C, z)


def boxcar(x, z, width):
      return 1.0 if np.linalg.norm(x-z) <=width/2 else 0.0

def linear(x, z):
      return np.dot(x,z)

def rbf(x, z, gamma):
      return np.exp(-gamma * np.linalg.norm(x-z)**2)
      
def polynomial(x, z, d):
      x = np.ndarray.flatten(x)
      z = np.ndarray.flatten(z)
      return (np.dot(x,z)+1)**d

