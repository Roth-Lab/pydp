'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from math import log, lgamma as log_gamma

from pydp.utils import memoized
from pydp.data import GammaParameter, NegativeBinomialParameter

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

class PoissonDensity(Density):
    @memoized
    def log_p(self, data, params):
        x = data.x
        
        l = params.x
        
        return log_poisson_pdf(x, l)

class NegativeBinomialDensity(object):
    @memoized
    def log_p(self, data, params):
        x = data.x
        
        if isinstance(params, NegativeBinomialParameter):
            r = params.r
            p = params.p
        elif isinstance(params, GammaParameter):
            r = params.a
            p = 1 / (1 + params.b)
        else:
            raise Exception("BetaParameter does not accept parameters of type {0}.".format(type(params)))
            
        return log_negative_binomial(x, r, p)

#=======================================================================================================================
# Log of probability density functions
#=======================================================================================================================
def log_beta_pdf(x, a, b):
    if x == 0 or x == 1:
        return float('-inf')    
    
    return -log_beta(a, b) + (a - 1) * log(x) + (b - 1) * log(1 - x)

def log_beta_binomial_pdf(x, n, a, b):
    return log_binomial_coefficient(n, x) + log_beta(a + x, b + n - x) - log_beta(a, b)

def log_binomial_pdf(x, n, p):
    return log_binomial_coefficient(n, x) + x * log(p) + (n - x) * log(1 - p)

def log_negative_binomial(x, r, p):
    return log_binomial_coefficient(x + r - 1 , x) + r * log(1 - p) + x * log(p)
    
def log_poisson_pdf(x, l):
    return x * log(l) - l - log_factorial(x)

#=======================================================================================================================
# Helper functions
#=======================================================================================================================
def log_beta(a, b):
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)

def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)

def log_factorial(n):
    return log_gamma(n + 1)
