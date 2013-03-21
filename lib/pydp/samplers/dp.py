'''
Created on 2013-03-21

@author: Andrew Roth
'''
from pydp.partition import Partition
from pydp.samplers.concentration import GammaPriorConcentrationSampler

class DirichletProcessSampler(object):
    def __init__(self, atom_sampler, partition_sampler, alpha=None, alpha_shape=1, alpha_rate=1):
        self.atom_sampler = atom_sampler
        
        self.partition_sampler = partition_sampler           
        
        if alpha is None:
            self.alpha = 1
            
            self.update_alpha = True
            
            self.concentration_sampler = GammaPriorConcentrationSampler(alpha_shape, alpha_rate)
        else:
            self.alpha = alpha
            
            self.update_alpha = False 
        
        self.num_iters = 0
    
    @property
    def state(self):
        return {
                'alpha' : self.alpha,
                'labels' : self.partition.labels,
                'params' : [param for param in self.partition.cell_values]
                }
    
    def initialise_partition(self, data):
        self.partition = Partition()
        
        for item, _ in enumerate(data):
            self.partition.add_cell(self.partition_sampler.base_measure.random())
            
            self.partition.add_item(item, item)        
    
    def sample(self, data, trace, num_iters, print_freq=100):
        self.initialise_partition(data)
        
        for i in range(num_iters):
            if i % print_freq == 0:
                print self.num_iters, self.partition.number_of_cells, self.alpha
            
            self.interactive_sample(data)
            
            trace.update(self.state)
            
            self.num_iters += 1
    
    def interactive_sample(self, data):
        if self.update_alpha:
            self.alpha = self.concentration_sampler.sample(self.alpha,
                                                           self.partition.number_of_cells,
                                                           self.partition.number_of_items)
        
        self.partition_sampler.sample(data, self.partition, self.alpha)
        
        self.atom_sampler.sample(data, self.partition)
    
    def _init_partition(self, base_measure):
        self.partition = Partition()
        
        for item, _ in enumerate(self.data):
            self.partition.add_cell(base_measure.random())
            
            self.partition.add_item(item, item)