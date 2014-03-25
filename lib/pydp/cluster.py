'''
Functions for clustering the output of DP MCMC trace.

Created on 2014-03-25

@author: Andrew Roth
'''
from math import exp
from pydp.densities import log_binomial_coefficient
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import pdist, squareform

import numpy as np

def cluster_with_mpear(X):
    '''
    Args:
        X : (array) An array with as many rows as (post-burnin) MCMC iterations and columns as data points.
    '''
    X = np.array(X).T
    
    dist_mat = pdist(X, metric='hamming')
    
    sim_mat = 1 - squareform(dist_mat)
    
    Z = average(dist_mat)
    
    max_pear = 0

    best_cluster_labels = _get_flat_clustering(Z, 1)
    
    for i in range(1, len(X) + 1):
        cluster_labels = _get_flat_clustering(Z, i)
    
        pear = _compute_mpear(cluster_labels, sim_mat)
        
        if pear > max_pear:
            max_pear = pear

            best_cluster_labels = cluster_labels
    
    return best_cluster_labels

def _get_flat_clustering(Z, number_of_clusters):
    N = len(Z) + 1
    
    if number_of_clusters == N:
        return np.arange(1, N + 1)
    
    return fcluster(Z, number_of_clusters, criterion='maxclust')

def _compute_mpear(cluster_labels, sim_mat):
    N = sim_mat.shape[0]
    
    c = exp(log_binomial_coefficient(N, 2))
    
    num_term_1 = 0
    
    for j in range(N):
        for i in range(j):
            if cluster_labels[i] == cluster_labels[j]:
                num_term_1 += sim_mat[i][j]

    num_term_2 = 0
    
    for j in range(N):
        for i in range(j):
            if cluster_labels[i] == cluster_labels[j]:
                num_term_2 += sim_mat[:j - 1, j].sum()
    
    num_term_2 /= c
    
    den_term_1 = 0
    
    for j in range(N):
        for i in range(j):
            den_term_1 += sim_mat[i][j]
            
            if cluster_labels[i] == cluster_labels[j]:
                den_term_1 += 1
    
    den_term_1 /= 2
    
    num = num_term_1 - num_term_2
    
    den = den_term_1 - num_term_2
    
    return num / den