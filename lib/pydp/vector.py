'''
Classes for extending scalar valued atoms to vector valued atoms.

Created on 2014-03-25

@author: Andrew Roth
'''
from collections import OrderedDict, namedtuple

from pydp.base_measures import BaseMeasure
from pydp.densities import Density
from pydp.partition import PartitionCell
from pydp.proposal_functions import ProposalFunction
from pydp.samplers.atom import AtomSampler

class VectorAtomSampler(AtomSampler):
    def __init__(self, base_measure, cluster_density, atom_samplers):
        '''
        Args:
           base_measure : (VectorBaseMeasure) Base measure.
           
           cluster_density : (VectorDensity) Emission density of clusters.
           
           atom_samplers : (dict) Mapping of dimension ID to atom sampler. 
        '''
        AtomSampler.__init__(self, base_measure, cluster_density)
        
        self.atom_samplers = atom_samplers
    
    def sample_atom(self, data, cell):
        new_atom = OrderedDict()
        
        for sample_id in self.atom_samplers:
            sample_data = [x[sample_id] for x in data]
            
            sample_cell = PartitionCell(cell.value[sample_id])
            
            sample_cell._items = cell._items
            
            new_atom[sample_id] = self.atom_samplers[sample_id].sample_atom(sample_data, sample_cell)
        
        return new_atom

class VectorBaseMeasure(BaseMeasure):
    def __init__(self, base_measures):
        '''
        Args:
            base_measures: (dict) Mapping of dimension IDs to base measures.
        '''
        self.base_measures = base_measures
    
    def log_p(self, data):
        log_p = 0
        
        for sample_id in self.base_measures:
            log_p += self.base_measures[sample_id].log_p(data[sample_id])
        
        return log_p
    
    def random(self):
        random_sample = OrderedDict()
        
        for sample_id in self.base_measures:
            random_sample[sample_id] = self.base_measures[sample_id].random()
        
        return random_sample

class VectorDensity(Density):
    '''
    Wraps a collection of univariate densities.
    '''
    def __init__(self, cluster_densities, shared_params=False):
        '''
        Args:
            cluster_densities: (dict) A collection of Density objects for each dimension.
        '''        
        self.cluster_densities = cluster_densities
        
        self.shared_params = shared_params
    
    @property
    def params(self):
        if self.shared_params:
            for cluster_id in self.cluster_densities:
                return self.cluster_densities[cluster_id].params
        
        else:
            params = OrderedDict()
            
            for cluster_id in self.cluster_densities:
                params[cluster_id] = self.cluster_densities[cluster_id].params
            
            return params
    
    @params.setter
    def params(self, value):
        if self.shared_params:
            for cluster_id in self.cluster_densities:
                self.cluster_densities[cluster_id].params = value

        elif isinstance(value, namedtuple):
            for cluster_id in self.cluster_densities:
                self.cluster_densities[cluster_id].params = value[cluster_id]
        
        else:
            raise Exception('Cannot set object type {0} as a density parameter'.format(type(value)))
            
    
    def log_p(self, data, params):
        log_p = 0
        
        for sample_id in self.cluster_densities:
            density = self.cluster_densities[sample_id]
            
            log_p += density.log_p(data[sample_id], params[sample_id])
             
        return log_p

class VectorProposalFunction(ProposalFunction):
    def __init__(self, proposal_funcs):
        '''
        Args:
            proposal_funcs: (dict) A collection of ProposalFunction for each dimension.
        '''
        self.proposal_funcs = proposal_funcs
    
    def log_p(self, data, params):
        log_p = 0
        
        for sample_id in self.proposal_funcs:
            log_p += self.proposal_funcs[sample_id].log_p(data[sample_id], params[sample_id])
        
        return log_p
    
    def random(self, params):
        random_sample = OrderedDict()
        
        for sample_id in self.proposal_funcs:
            random_sample[sample_id] = self.proposal_funcs[sample_id].random(params[sample_id])
        
        return random_sample