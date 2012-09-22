'''
Created on 2012-09-21

@author: Andrew Roth
'''
from collections import OrderedDict
from math import exp, isinf, log

import functools

#=======================================================================================================================
# Log space functions
#=======================================================================================================================
def log_sum_exp(log_X):
    '''
    Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])
    
    Numerically safer than naive method.
    '''
    max_exp = max(log_X)
    
    if isinf(max_exp):
        return max_exp
    
    total = 0

    for x in log_X:
        total += exp(x - max_exp)
    
    return log(total) + max_exp

def log_space_normalise(log_X):
    '''
    Given a list of values in log space return the values normalised such that exp(log_X[0]) + exp(log_X[1]) + ... = 1
    '''
    normalised_log_X = []
    
    log_norm_const = log_sum_exp(log_X)
    
    for x in log_X:
        normalised_log_X.append(x - log_norm_const)
    
    return normalised_log_X

#=======================================================================================================================
# Caching
#=======================================================================================================================
class memoized(object):
    def __init__(self, func, cache_size=10000):
        self.func = func
        
        self.cache = OrderedDict()

        self.cache_size = cache_size        
    
    def __call__(self, *args):
        if args in self.cache:
            value = self.cache[args]
        else:
            value = self.func(*args)
            
            self.cache[args] = value
        
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return value

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        
        return functools.partial(self.__call__, obj)
