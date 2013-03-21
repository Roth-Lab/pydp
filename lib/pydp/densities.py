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

from math import log, lgamma as log_gamma, pi

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
    def log_p(self, data, params):
        x = data.x
        
        a = params.a
        b = params.b
        
        return log_beta_pdf(x, a, b)

class BetaBinomialDensity(Density):
    def log_p(self, data, params):
        x = data.x
        n = data.n
        
        a = params.a
        b = params.b
        
        return log_beta_binomial_pdf(x, n, a, b) 

class BinomialDensity(Density):
    def log_p(self, data, params):
        x = data.x
        n = data.n
        
        p = params.x
        
        return log_binomial_pdf(x, n, p)

class GaussianDensity(Density):
    def log_p(self, data, params):
        x = data.x
        
        mean = params.mean        
        precision = params.precision
        
        return log_gaussian_pdf(x, mean, precision)      

class PoissonDensity(Density):
    def log_p(self, data, params):
        x = data.x
        
        l = params.x
        
        return log_poisson_pdf(x, l)

class NegativeBinomialDensity(object):
    def log_p(self, data, params):
        x = data.x
        
        if isinstance(params, NegativeBinomialParameter):
            r = params.r
            p = params.p
        elif isinstance(params, GammaParameter):
            r = params.a
            p = 1 / (1 + params.b)
        else:
            raise Exception("NegativeBinomialDensity does not accept parameters of type {0}.".format(type(params)))
            
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
    if p == 0:
        if x == 0:
            return 0
        else:
            return float('-inf')
    
    if p == 1:
        if x == n:
            return 0
        else:
            return float('-inf')
    
    return log_binomial_coefficient(n, x) + x * log(p) + (n - x) * log(1 - p)

def log_gamma_pdf(x, a, b):
    return -log_gamma(a) + a * log(b) + (a - 1) * log(x) - b * x

def log_gaussian_pdf(x, mean, precision):
    sigma2 = 1 / precision
    
    return log_normal_pdf(x, mean, sigma2)

def log_negative_binomial(x, r, p):
    return log_binomial_coefficient(x + r - 1 , x) + r * log(1 - p) + x * log(p)

def log_normal_pdf(x, mu, sigma2):
    return -1 / 2 * log(2 * pi * sigma2) - (x - mu) ** 2 / (2 * sigma2)

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
