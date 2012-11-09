'''
Created on 2012-09-22

@author: Andrew Roth
'''
from pydp.data import BetaData
from pydp.rvs import beta_rvs
from pydp.densities import log_beta_pdf

class ProposalFunction(object):
    def random(self, params):
        pass

class BaseMeasureProposalFunction(ProposalFunction):
    def __init__(self, base_measure):
        self.base_measure = base_measure
        
    def log_p(self, data, params):
        return 0
    
    def random(self, params):
        return self.base_measure.random()

class BetaProposalFunction(ProposalFunction):
    def __init__(self, s):
        self.s = s
        
    def log_p(self, data, params):
        a, b = self._get_standard_params(params)
        
        return log_beta_pdf(data.x, a, b)
    
    def random(self, params):
        a, b = self._get_standard_params(params)
        
        a += 1
        b += 1
        
        return BetaData(beta_rvs(a, b))
    
    def _get_standard_params(self, params):
        s = self.s
        m = params.x
        
        a = s * m
        b = s - a
        
        return a, b
