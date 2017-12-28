'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.
'''

from pydp.base_measures import BetaBaseMeasure

from pydp.data import BinomialData, BetaParameter

from pydp.densities import BinomialDensity, BetaBinomialDensity

from pydp.partition import Partition

from pydp.rvs import binomial_rvs, gamma_rvs


from pydp.samplers.atom import BetaBinomialGibbsAtomSampler, BaseMeasureAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import MarginalGibbsPartitionSampler, AuxillaryParameterPartitionSampler, \
    MetropolisGibbsPartitionSampler
    
from pydp.tests.simulators import sample_from_crp

from pydp.diagnostics import geweke_convergence_test, geweke_joint_distribution_test

class Sampler(object):
    def __init__(self, base_measure, cluster_density):
        self.base_measure = base_measure
        
        self.cluster_density = cluster_density
        
        self.concentration_sampler = GammaPriorConcentrationSampler(1, 1)
        
        posterior_density = BetaBinomialDensity()
        
#        self.partition_sampler = MarginalGibbsPartitionSampler(base_measure, cluster_density, posterior_density)
        self.partition_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)
        
        self.atom_sampler = BetaBinomialGibbsAtomSampler(base_measure, cluster_density)
#        self.atom_sampler = BaseMeasureAtomSampler(base_measure, cluster_density)
            
    def sample(self, alpha, partition, data):        
#        alpha = self.concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
        
#        self.partition_sampler.sample(data, partition, alpha)
        
        self.atom_sampler.sample(data, partition)
        
        return alpha, partition

def draw_data(params, n):
    data = []
    
    for p in params:
        x = binomial_rvs(n, p)
        
        data.append(BinomialData(x, n))
    
    return data

def draw_from_prior(base_measure, size):
    alpha = gamma_rvs(1, 1)
    
#    partition = sample_from_crp(alpha, size, base_measure)
    
    partition = Partition()
    
    partition.add_cell(base_measure.random())
    
    for item in range(size):
        partition.add_item(item, 0)
    
    return alpha, partition

#=======================================================================================================================
# Simulators
#=======================================================================================================================
def marginal_conditional_simulator(base_measure, n, size, num_iters): 
    params = []
    
    for _ in range(num_iters):
        alpha, partition = draw_from_prior(base_measure, size) 
        
        data_params = [value.x for value in partition.item_values]
        
        params.append({
                       'alpha' : alpha,
                       'p' : data_params
                       })
        
#        draw_data(data_params, n)
    
    return params

def successive_conditional_simulator(base_measure, cluster_density, n, size, num_iters):
    params = []
    
    sampler = Sampler(base_measure, cluster_density)
    
    alpha, partition = draw_from_prior(base_measure, size)
    
    for _ in range(num_iters):
        data = draw_data([param.x for param in partition.item_values], n)
        
        alpha, partition = sampler.sample(alpha, partition, data)
        
        params.append({
                       'alpha' : alpha,
                       'p' : [value.x for value in partition.item_values]
                       })
                
    return params

size = 10
n = 100

num_iters = int(2e5)
burnin = int(1e5)
thin = int(100)

base_measure = BetaBaseMeasure(1, 1)

cluster_density = BinomialDensity()

params_1 = marginal_conditional_simulator(base_measure, n, size, num_iters)
#params_2 = marginal_conditional_simulator(base_measure, n, size, num_iters)

#params_1 = successive_conditional_simulator(base_measure, cluster_density, n, size, num_iters)
params_2 = successive_conditional_simulator(base_measure, cluster_density, n, size, num_iters)

#trace_1 = [x['alpha'] for x in params_1][burnin::thin]
#trace_2 = [x['alpha'] for x in params_2][burnin::thin]
#
#print compare_trace(trace_1, trace_2, lambda x: x)
#print compare_trace(trace_1, trace_2, lambda x: x ** 2)

for i in range(size):
    trace_1 = [x['p'][i] for x in params_1][burnin::thin]
    trace_2 = [x['p'][i] for x in params_2][burnin::thin]
    
    print()
    print(geweke_convergence_test(trace_2))
    print()
    
    print(geweke_joint_distribution_test(trace_1, trace_2, lambda x: x))
    print(geweke_joint_distribution_test(trace_1, trace_2, lambda x: x ** 2))
