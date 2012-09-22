'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from pydp.rvs import beta_rvs 
from pydp.data import BetaData

class BaseMeasure(object):
    '''
    Base class for base measures.
    '''
    def __init__(self, params):
        pass
    
    def random(self):
        '''
        Return a random sample from the base measure.
        '''
        pass
    
class BetaBaseMeasure(object):
    def __init__(self, params):
        self.params = params
    
    def random(self):
        x = beta_rvs(self.params.a, self.params.a)
        
        return BetaData(x)
