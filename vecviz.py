import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as LA

def perf_PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape
    
    # mean center the data
    data -= data.mean(axis=0)
    
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    
    # sort eigenvectors according to same index
    evals = evals[idx]
    
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(data,evecs), evals, evecs

def plot_pca(data):
    clr1 =  '#2026B2'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_evals, data_evecs = perf_PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    plt.show()

