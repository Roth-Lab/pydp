'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

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
# Integration
#=======================================================================================================================
class Integrator(object):
    def __init__(self, a=0, b=1, mesh_size=100):
        self.a = a
        self.b = b
        self.mesh_size = mesh_size
        
        self.step_size = (b - a) / mesh_size
        
        self.knots = [i * self.step_size + a for i in range(0, mesh_size + 1)]
    
class SimpsonsRuleIntegrator(Integrator):
    def __init__(self, a=0, b=1, mesh_size=100):
        if mesh_size % 2 != 0:
            raise Exception("Mesh size for Simpson's rule must be an even number.")
        
        Integrator.__init__(self, a, b, mesh_size)

    def log_integrate(self, log_f):
        log_total = []
        
        # First and last terms.
        log_total.append(log_f(self.knots[0]))
        log_total.append(log_f(self.knots[-1]))
        
        four_total = []
        
        for i in range(1, self.mesh_size, 2):
            four_total.append(log_f(self.knots[i]))
        
        log_total.append(log(4) + log_sum_exp(four_total))
        
        two_total = []
        
        for i in range(2, self.mesh_size - 1, 2):
            two_total.append(log_f(self.knots[i]))
        
        log_total.append(log(2) + log_sum_exp(two_total))
  
        return log(self.step_size) - log(3) + log_sum_exp(log_total)
    
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