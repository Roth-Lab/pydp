'''
Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from math import exp, log
from random import shuffle

from pydp.rvs import discrete_rvs, uniform_rvs
from pydp.utils import log_space_normalise

class PartitionSampler(object):
    '''
    Base class for samplers which update the partition of the DP.
    

    Args:
        base_measure : (BaseMeasure) Base measure for DP process.
        
        cluster_density : (ClusterDensity) Cluster density for DP process.
    '''
    def __init__(self, base_measure, cluster_density):
        self.base_measure = base_measure
        
        self.cluster_density = cluster_density
        
    def sample(self, data, old_partition, alpha, **kwargs):
        '''
            data : (list) List of data points appropriate for cluster_density.
            
            partition : (Partition) Partition of DP.
            
            alpha : (float) Concentration parameter for the DP.
        '''
        pass

#=======================================================================================================================
# Non-conjugate samplers
#=======================================================================================================================
class AuxillaryParameterPartitionSampler(PartitionSampler):
    def sample(self, data, partition, alpha, m=2):
        '''
        Sample a new partition according to algorithm 8 of Neal "Sampling Methods For Dirichlet Process Mixture Models"
        '''
        items = range(len(data))
        
        shuffle(items)
        
        for item in items:
            data_point = data[item]
            
            old_cell_index = partition.labels[item]
            
            partition.remove_item(item, old_cell_index)
            
            if partition.counts[old_cell_index] == 0:
                num_new_tables = m - 1
            else:
                num_new_tables = m
            
            for _ in range(num_new_tables):
                partition.add_cell(self.base_measure.random())
            
            log_p = []
            
            for cell in partition.cells:
                cluster_log_p = self.cluster_density.log_p(data_point, cell.value)
                
                counts = cell.size
                
                if counts == 0:
                    counts = alpha / m
                
                log_p.append(log(counts) + cluster_log_p)
    
            log_p = log_space_normalise(log_p)
            
            p = [exp(x) for x in log_p]
            
            new_cell_index = discrete_rvs(p)
            
            partition.add_item(item, new_cell_index)
            
            partition.remove_empty_cells()

class MetropolisGibbsPartitionSampler(PartitionSampler):
    '''
    Sample a new partition according to algorithm 7 of Neal "Sampling Methods For Dirichlet Process Mixture Models"
    '''    
    def sample(self, data, partition, alpha):
        n = partition.number_of_items
        
        for item, data_point in enumerate(data):
            old_cluster_label = partition.labels[item]
            old_value = partition.item_values[item]
            
            partition.remove_item(item, old_cluster_label)
            
            if partition.counts[old_cluster_label] == 0:
                p = [x / (n - 1) for x in partition.counts]
                
                new_cluster_label = discrete_rvs(p)
                
                new_value = partition.cell_values[new_cluster_label]
                
                old_ll = self.cluster_density.log_p(data_point, old_value)
                new_ll = self.cluster_density.log_p(data_point, new_value)
                
                log_ratio = log(n - 1) - log(alpha) + new_ll - old_ll
                
                u = uniform_rvs(0, 1)
                
                if log_ratio >= log(u):
                    partition.add_item(item, new_cluster_label)
                else:
                    partition.add_item(item, old_cluster_label)
            
            else:
                new_value = self.base_measure.random()
                
                old_ll = self.cluster_density.log_p(data_point, old_value)
                new_ll = self.cluster_density.log_p(data_point, new_value)
                
                log_ratio = log(alpha) - log(n - 1) + new_ll - old_ll
                
                u = uniform_rvs(0, 1)
                
                if log_ratio >= log(u):
                    partition.add_cell(new_value)
                    
                    cell = partition.get_cell_by_value(new_value)
                    
                    cell.add_item(item)
                else:
                    partition.add_item(item, old_cluster_label)
        
        partition.remove_empty_cells()
        
        for item, data_point in enumerate(data):
            old_cluster_label = partition.labels[item]
            
            if partition.cells[old_cluster_label].size == 1:
                continue
            
            partition.remove_item(item, old_cluster_label)
            
            log_p = []
            
            for cell in partition.cells:
                cluster_log_p = self.cluster_density.log_p(data_point, cell.value)
                
                counts = cell.size
                
                log_p.append(log(counts) + cluster_log_p)
    
            log_p = log_space_normalise(log_p)
            
            p = [exp(x) for x in log_p]
            
            new_cluster_label = discrete_rvs(p)
            
            partition.add_item(item, new_cluster_label)
        
        partition.remove_empty_cells()

#=======================================================================================================================
# Conjugate Samplers
#=======================================================================================================================
class MarginalGibbsPartitionSampler(PartitionSampler):
    '''
    Update the partition using algorithm 2 of Neal "Sampling Methods For Dirichlet Process Mixture Models".
    '''
    def __init__(self, base_measure, cluster_density, posterior_predictive_density):
        '''
        Args:
            base_measure : (BaseMeasure) Base measure for DP process.
            
            cluster_density : (Density) Cluster density for DP process.
            
            posterior_predictive_density : (Density) Posterior density obtained by integrating the prior against the likelihood for
            the model.
        '''            
        PartitionSampler.__init__(self, base_measure, cluster_density)
        
        self.posterior_density = posterior_predictive_density
    
    def sample(self, data, partition, alpha):
        for item, data_point in enumerate(data):
            old_cell_index = partition.labels[item]
            
            partition.remove_item(item, old_cell_index)
            
            partition.remove_empty_cells()
            
            log_p = []    
            
            for cell in partition.cells:
                cluster_log_p = self.cluster_density.log_p(data_point, cell.value)
                
                counts = cell.size
                
                log_p.append(log(counts) + cluster_log_p)

            params = self.base_measure.params
    
            cluster_log_p = self.posterior_density.log_p(data_point, params)
            
            log_p.append(log(alpha) + cluster_log_p)
            
            log_p = log_space_normalise(log_p)    
            
            p = [exp(x) for x in log_p]
            
            new_cell_index = discrete_rvs(p)
            
            if new_cell_index == partition.number_of_cells:
                partition.add_cell(self.base_measure.random())
                
            partition.add_item(item, new_cell_index)
