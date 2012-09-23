'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from math import log

from pydp.rvs import beta_rvs, gamma_rvs, uniform_rvs

from pydp.data import BetaData, GammaData
from pydp.proposal_functions import BaseMeasureProposalFunction

class AtomSampler(object):
    '''
    Base class for samplers to update the cell values in the partition (atoms of DP).
    '''
    def __init__(self, base_measure, cluster_density):
        '''
        Args:
            base_measure : (BaseMeasure) Base measure for DP process.
            
            cluster_density : (Density) Cluster density for DP process.
        '''
        self.base_measure = base_measure
        
        self.cluster_density = cluster_density    
    
    def sample(self, data, partition):
        '''
        Sample a new partition of the data. The partition passed in will be updated in place.
        
        Args:
            data : (list) List of data points appropriate for cluster_density.
            
            partition : (Partition) Partition of dp.
        '''
        pass

#=======================================================================================================================
# Non-conjugate samplers
#=======================================================================================================================
class MetropolisHastingsAtomSampler(AtomSampler):
    '''
    Update the atom values using a Metropolis-Hastings steps with a user specified proposal function which takes
    the previous cell value as an argument.
    '''          
    def __init__(self, base_measure, cluster_density, proposal_func):
        AtomSampler.__init__(self, base_measure, cluster_density)
        
        self.proposal_func = proposal_func
    
    def sample(self, data, partition):
        for cell in partition.cells:
            old_ll = 0
            new_ll = 0
            
            old_param = cell.value
            new_param = self.proposal_func.random(old_param)
            
            for j in cell.items:
                old_ll += self.cluster_density.log_p(data[j], old_param)
                new_ll += self.cluster_density.log_p(data[j], new_param)
            
            log_ratio = new_ll - old_ll
            
            u = uniform_rvs(0, 1)
            
            if log_ratio >= log(u):
                cell.value = new_param

class BaseMeasureAtomSampler(MetropolisHastingsAtomSampler):
    '''
    Update the atom values using a Metropolis-Hastings steps with the base measure as a proposal density.
    '''
    def __init__(self, base_measure, cluster_density):
        AtomSampler.__init__(self, base_measure, cluster_density)
        
        self.proposal_func = BaseMeasureProposalFunction(base_measure)
    
#=======================================================================================================================
# Conjugate samplers
#=======================================================================================================================
class BetaBinomialGibbsAtomSampler(AtomSampler):
    '''
    Update the partition values using a Gibbs step. 
    
    Requires a Beta base measure and binomial data.
    '''  
    def sample(self, data, partition):
        for cell in partition.cells:
            a = self.base_measure.params.a
            b = self.base_measure.params.b
            
            for item in cell.items:
                a += data[item].x
                b += data[item].n - data[item].x
            
            cell.value = BetaData(beta_rvs(a, b))

class GammaPoissonGibbsAtomSampler(AtomSampler):
    '''
    Update the partition values using a Gibbs step. 
    
    Requires a Gamma base measure and Poisson data.
    '''  
    def sample(self, data, partition):
        for cell in partition.cells:
            a = self.base_measure.params.a
                        
            for item in cell.items:
                a += data[item].x
            
            n = cell.size
            b = self.base_measure.params.b
            
            b = b / (n * b + 1)
            
            cell.value = GammaData(gamma_rvs(a, b))
