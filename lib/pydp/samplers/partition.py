'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2012-09-21

@author: Andrew Roth
'''
from __future__ import division

from math import exp, log, lgamma as log_gamma
from random import sample, shuffle

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

class SequentiallyAllocatedMergeSplitSampler(PartitionSampler):
    def __init__(self, base_measure, cluster_density, proposal_func=None):        
        PartitionSampler.__init__(self, base_measure, cluster_density)
        
        if proposal_func is None:
            self.proposal_func = base_measure
        else:
            self.proposal_func = proposal_func
    
    def sample(self, data, old_partition, alpha):
        items = range(len(data))
        
        i, j = sample(items, 2)
        
        labels = old_partition.labels
        
        c_i = labels[i]
        
        c_j = labels[j]
        
        new_partition = old_partition.copy()

        if c_i == c_j:
            c = c_i
            
            old_cell = new_partition.cells[c]
            
            new_cell_i, new_cell_j, forward_log_q, reverse_log_q = self._split(i, j, old_cell, data, new_partition)
            
            forward_log_p = self._compute_partition_log_p(new_cell_i, data) + \
                            self._compute_partition_log_p(new_cell_j, data)
                            
            old_cell = old_partition.cells[c]
                            
            reverse_log_p = self._compute_partition_log_p(old_cell, data)

        else:
            cell_i = new_partition.cells[c_i]
            cell_j = new_partition.cells[c_j]
            
            new_cell, forward_log_q, reverse_log_q = self._merge(cell_i, cell_j, data, new_partition)
            
            forward_log_p = self._compute_partition_log_p(new_cell, data)
            
            old_cell_i = old_partition.cells[c_i]
            old_cell_j = old_partition.cells[c_j]
            
            reverse_log_p = self._compute_partition_log_p(old_cell_i, data) + \
                            self._compute_partition_log_p(old_cell_j, data)
        
        forward_log_prior = self._compute_prior_log_p(alpha, new_partition)        
        reverse_log_prior = self._compute_prior_log_p(alpha, old_partition)
                
        forward_log_ratio = forward_log_p + forward_log_prior - forward_log_q        
        reverse_log_ratio = reverse_log_p + reverse_log_prior - reverse_log_q

        log_ratio = forward_log_ratio - reverse_log_ratio

        u = uniform_rvs(0, 1)
        
        if log_ratio >= log(u):
            print "accepted"
            
            old_partition.cells = new_partition.cells
        else:
#             print "rejected"
#             print forward_log_p - reverse_log_p, forward_log_q - reverse_log_q
            pass

    def _merge(self, old_cell_i, old_cell_j, data, partition):
        s_i = old_cell_i.items     
        s_j = old_cell_j.items
        
        param_i = old_cell_i.value
        param_j = old_cell_j.value
        
        param_new = param_i

        forward_log_q = self.proposal_func.log_p(param_new, param_i)
        reverse_log_q = self.proposal_func.log_p(param_i, param_new) + self.proposal_func.log_p(param_j, param_new)
        
        new_cell = partition.add_cell(param_new)
        
        for k in s_i:
            old_cell_i.remove_item(k)
            
            new_cell.add_item(k)
        
        for k in s_j:
            old_cell_j.remove_item(k)
            
            new_cell.add_item(k)

        temp_s_i = set([s_i.pop(), ])
        temp_s_j = set([s_j.pop(), ])
        
        items = s_i + s_j
        
        shuffle(items)
        
        for k in items:
            n_i = len(temp_s_i)
            n_j = len(temp_s_j)
            
            log_p = [
                     log(n_i) + self.cluster_density.log_p(data[k], param_i),
                     log(n_j) + self.cluster_density.log_p(data[k], param_j)
                     ]
            
            log_p = log_space_normalise(log_p)
            
            if k in s_i:
                temp_s_i.add(k)
                
                reverse_log_q += log_p[0]
                
            else:
                temp_s_j.add(k)
                
                reverse_log_q += log_p[1]
        
        partition.remove_empty_cells()
        
        return new_cell, forward_log_q, reverse_log_q
    
    def _split(self, i, j, old_cell, data, partition):
        old_cell.remove_item(i)        
        old_cell.remove_item(j)
        
        param_i = old_cell.value      
        param_j = self.proposal_func.random(param_i)
        
        forward_log_q = self.proposal_func.log_p(param_i, param_i) + self.proposal_func.log_p(param_j, param_i)
        reverse_log_q = self.proposal_func.log_p(param_i, param_i)
        
        new_cell_i = partition.add_cell(param_i)        
        new_cell_j = partition.add_cell(param_j)
        
        new_cell_i.add_item(i)
        new_cell_j.add_item(j)
        
        s = old_cell.items
        shuffle(s)
        
        for k in s:
            old_cell.remove_item(k)
            
            n_i = new_cell_i.size
            n_j = new_cell_j.size
            
            log_p = [
                     log(n_i) + self.cluster_density.log_p(data[k], param_i),
                     log(n_j) + self.cluster_density.log_p(data[k], param_j)
                     ]
            
            log_p = log_space_normalise(log_p)
            
            p = [exp(x) for x in log_p]
            
            c_k = discrete_rvs(p)
            
            if c_k == 0:                
                new_cell_i.add_item(k)
            else:
                new_cell_j.add_item(k)
            
            forward_log_q += log_p[c_k]
        
        partition.remove_empty_cells()

        return new_cell_i, new_cell_j, forward_log_q, reverse_log_q
    
    def _compute_partition_log_p(self, cell, data):
        log_p = 0
        
        param = cell.value
        
        for item in cell.items:
            log_p += self.cluster_density.log_p(data[item], param)
        
        return log_p
    
    def _compute_prior_log_p(self, alpha, partition):
        log_p = partition.number_of_cells * log(alpha)

        for n_c in partition.counts:
            log_p += log_gamma(n_c)
        
        n = partition.number_of_items
        
        for i in range(1, n + 1):
            log_p -= log_gamma(alpha + i - 1)
        
        return log_p
    
class SplitMergeAuxillaryHybridSampler(PartitionSampler):
    def __init__(self, base_measure, cluster_density, proposal_func=None, ratio=0.1):
        PartitionSampler.__init__(self, base_measure, cluster_density)
        
        self.ratio = ratio
        
        self.auxillary_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)
        
        self.split_merge_sampler = SequentiallyAllocatedMergeSplitSampler(base_measure, cluster_density, proposal_func)
    
    def sample(self, data, partition, alpha):
        u = uniform_rvs(0, 1)
        
        if u < self.ratio:
            self.auxillary_sampler.sample(data, partition, alpha)
        else:
            self.split_merge_sampler.sample(data, partition, alpha)
        
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
