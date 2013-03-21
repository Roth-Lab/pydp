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

from pydp.data import BetaData, BetaParameter, GammaData, GammaParameter, GaussianGammaParameter, GaussianGammaData
from pydp.rvs import beta_rvs, gamma_rvs, gaussian_rvs
from pydp.densities import log_beta_pdf, log_gamma_pdf, log_gaussian_pdf

class BaseMeasure(object):
    '''
    Base class for base measures.
    '''
    def log_p(self, data):
        '''
        Return the log probability of the density.
        
        Args:
            data : An data object of the same type as returned by self.random()
        '''
        raise NotImplemented
    
    def random(self):
        '''
        Return a random sample from the base measure.
        '''
        raise NotImplemented
    
class BetaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = BetaParameter(a, b)
        
    def log_p(self, data):
        return log_beta_pdf(data.x, self.params.a, self.params.b)      
    
    def random(self):
        x = beta_rvs(self.params.a, self.params.b)
        
        return BetaData(x)
    
class GammaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = GammaParameter(a, b)
    
    def log_p(self, data):
        return log_gamma_pdf(data.x, self.params.a, self.params.b)
    
    def random(self):
        x = gamma_rvs(self.params.a, self.params.b)
        
        return GammaData(x)

class GaussianGammaBaseMeasure(BaseMeasure):
    def __init__(self, mean, size, alpha, beta):
        self.params = GaussianGammaParameter(mean, size, alpha, beta)
    
    def log_p(self, data):
        log_p_precision = log_gamma_pdf(data.precision, self.params.alpha, self.params.beta)
        
        log_p_mean = log_gaussian_pdf(data.mean, self.params.mean, self.params.size * data.precision) 
        
        return log_p_mean + log_p_precision
        
    def random(self):
        precision = gamma_rvs(self.params.alpha, self.params.beta)
        
        mean = gaussian_rvs(self.params.mean, self.params.size * precision)
        
        return GaussianGammaData(mean, precision)
    
       
