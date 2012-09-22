'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from math import log, lgamma as log_gamma

from pydp.utils import memoized

class Density(object):
    def log_p(self, data, params):
        '''
        Args:
            data : (nametuple) Data for density.
            
            params : (nametuple) Parameters in density.
        '''
        pass

class BetaDensity(Density):
    @memoized
    def log_p(self, data, params):
        x = data.x
        
        a = params.a
        b = params.b
        
        return log_beta_pdf(x, a, b)

class BetaBinomialDensity(Density):
    @memoized
    def log_p(self, data, params):
        x = data.x
        n = data.n
        
        a = params.a
        b = params.b
        
        return log_beta_binomial_pdf(x, n, a, b) 


class BinomialDensity(Density):
    @memoized 
    def log_p(self, data, params):
        x = data.x
        n = data.n
        
        p = params.x
        
        return log_binomial_pdf(x, n, p)
#=======================================================================================================================
# Log of probability density functions
#=======================================================================================================================
def log_beta_pdf(x, a, b):
    if x == 0 or x == 1:
        return float('-inf')    
    
    return -log_beta(a, b) + (a - 1) * log(x) + (b - 1) * log(1 - x)

def log_beta_binomial_pdf(x, n, a, b):
    return log_binomial_coefficient(n, x) + log_beta(a + x, b + n - x) - log_beta(a, b)

def log_binomial_pdf(x, n, mu):
    return log_binomial_coefficient(n, x) + x * log(mu) + (n - x) * log(1 - mu)

#=======================================================================================================================
# Helper functions
#=======================================================================================================================
def log_beta(a, b):
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)

def log_binomial_coefficient(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

def log_factorial(n):
    return log_gamma(n + 1)