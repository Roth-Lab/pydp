'''
Created on 2012-09-21

@author: Andrew Roth
'''
from math import exp, log

from random import betavariate as beta_rvs, gammavariate as gamma_rvs, normalvariate as normal_rvs, \
    uniform as uniform_rvs
    
from pydp.utils import log_sum_exp

def bernoulli_rvs(p):
    '''
    Return a Bernoulli distributed random variable.
    
    Args:
        p : (float) Probability of success.
    
    Returns:
        x : (int) Binary indicator of success/failure.
    '''
    u = uniform_rvs(0, 1)
    
    if u <= p:
        return 1
    else:
        return 0

def binomial_rvs(n, p):
    '''
    Sample a binomial distributed random variable.
    
    Args:
        n : (int) Number of trials performed.
        p : (int) Probability of success for each trial.
    
    Returns:
        x : (int) Number of successful trials.
    '''
    if p > 0.5:
        return n - binomial_rvs(n, 1 - p)
    
    if p == 0:
        return 0
    
    u = uniform_rvs(0, 1)
    log_u = log(u)
    
    log_c = log(p) - log(1 - p)
    
    i = 0
    
    log_prob = n * log(1 - p)
    
    log_F = log_prob
    
    while True:
        if log_u < log_F:
            return i
        
        log_prob += log_c + log(n - i) - log(i + 1)
        
        log_F = log_sum_exp([log_F, log_prob])
        
        i += 1
        
def dirichlet_rvs(alpha):
    '''
    Sample a Dirichlet distributed random variable.
    
    Args:
        alpha : (list) Pseudo count parameter for Dirichlet distribution.
    
    Returns:
        pi : (list) List of probabilities for each class such that sum(pi) == 1.
    '''
    g = [gamma_rvs(a, 1) for a in alpha]
    
    norm_const = sum(g)
    
    return [x / norm_const  for x in g]

def discrete_rvs(p):
    '''
    Sample a discrete (Categorical) random variable.
    
    Args:
        p : (list) Probabilities for each class from 0 to len(p) - 1
    
    Returns:
        i : (int) Id of class sampled. 
    '''
    total = 0
    
    u = uniform_rvs(0, 1)
    
    for i, p_i in enumerate(p):
        total += p_i
        
        if u < total:
            break
    
    return i
