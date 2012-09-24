from __future__ import division

from pydp.base_measures import BetaBaseMeasure

from pydp.data import BinomialData, BetaParameter

from pydp.densities import BinomialDensity, BetaBinomialDensity

from pydp.partition import Partition

from pydp.rvs import binomial_rvs, gamma_rvs, normal_rvs, uniform_rvs


from pydp.samplers.atom import BetaBinomialGibbsAtomSampler, BaseMeasureAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import MarginalGibbsPartitionSampler, AuxillaryParameterPartitionSampler, \
    MetropolisGibbsPartitionSampler
    
from pydp.tests.simulators import sample_from_crp

from pydp.diagnostics import geweke_convergence_test, geweke_joint_distribution_test
from pydp.stats import chi_square_cdf, inverse_normal_cdf, mean

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
        alpha = self.concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
        
        self.partition_sampler.sample(data, partition, alpha)
        
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
    
    partition = sample_from_crp(alpha, size, base_measure)
    
#    partition = Partition()
#    
#    partition.add_cell(base_measure.random())
#    
#    for item in range(size):
#        partition.add_item(item, 0)
    
    return alpha, partition

#=======================================================================================================================
# Simulators
#=======================================================================================================================
size = 10
n = 100

num_iters = int(2e4)
burnin = int(1e4)
thin = 1
num_replicates = 1000

base_measure = BetaBaseMeasure(1, 1)

cluster_density = BinomialDensity()

test_stat = []

for _ in range(num_replicates):
    alpha, partition = draw_from_prior(base_measure, size)
    
    observed_p = partition.item_values[0].x
    
    params = [param.x for param in partition.item_values]
    
    data = draw_data(params, n)
    
    sampler = Sampler(base_measure, cluster_density)
    
    posterior = []
    
    for _ in range(num_iters):
        alpha, partition = sampler.sample(alpha, partition, data)

        posterior.append(partition.item_values[0].x)
    
    q = 0
    
    for value in posterior:
        if observed_p > value:
            q += 1
    
    q = q / len(posterior)
    
    print q, inverse_normal_cdf(q), observed_p, mean(posterior)
    
    test_stat.append(inverse_normal_cdf(q) ** 2)
    
print 1 - chi_square_cdf(sum(test_stat), len(test_stat))
