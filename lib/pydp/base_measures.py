'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from pydp.data import BetaData, BetaParameter, GammaData, GammaParameter
from pydp.rvs import beta_rvs, gamma_rvs

class BaseMeasure(object):
    '''
    Base class for base measures.
    '''
    def random(self):
        '''
        Return a random sample from the base measure.
        '''
        pass
    
class BetaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = BetaParameter(a, b)
    
    def random(self):
        x = beta_rvs(self.params.a, self.params.b)
        
        return BetaData(x)
    
class GammaBaseMeasure(BaseMeasure):
    def __init__(self, a, b):
        self.params = GammaParameter(a, b)
    
    def random(self):
        x = gamma_rvs(self.params.a, self.params.b)
        
        return GammaData(x)
