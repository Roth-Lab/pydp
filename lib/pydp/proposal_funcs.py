'''
Created on 2012-09-22

@author: Andrew Roth
'''
from pydp.data import BetaData
from pydp.rvs import beta_rvs

def beta_proposal(params):
    s = 10
    m = params.x
    
    a = s * m
    b = s - a
    
    a += 1
    b += 1
    
    return BetaData(beta_rvs(a, b))