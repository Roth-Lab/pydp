'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from pydp.rvs import beta_rvs, gamma_rvs
from pydp.data import BetaData, GammaData

class BaseMeasure(object):
    '''
    Base class for base measures.
    '''
    def __init__(self, params):
        self.params = params
    
    def random(self):
        '''
        Return a random sample from the base measure.
        '''
        pass
    
class BetaBaseMeasure(BaseMeasure):
    def random(self):
        x = beta_rvs(self.params.a, self.params.a)
        
        return BetaData(x)
    
class GammaBaseMeasure(BaseMeasure):
    def random(self):
        x = gamma_rvs(self.params.a, self.params.b)
        
        return GammaData(x)
