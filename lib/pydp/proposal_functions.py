'''
Created on 2012-09-22

@author: Andrew Roth
'''
from pydp.data import BetaData
from pydp.rvs import beta_rvs

class ProposalFunction(object):
    def random(self, params):
        pass

class BaseMeasureProposalFunction(object):
    def __init__(self, base_measure):
        self.base_measure = base_measure
    
    def random(self, params):
        return self.base_measure.random()

class BetaProposalFunction(ProposalFunction):
    def __init__(self, s):
        self.s = s
    
    def random(self, params):
        s = self.s
        m = params.x
        
        a = s * m
        b = s - a
        
        a += 1
        b += 1
        
        return BetaData(beta_rvs(a, b))