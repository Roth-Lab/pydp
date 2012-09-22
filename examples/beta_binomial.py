from pydp.base_measures import BetaBaseMeasure

from pydp.data import BinomialData, BetaParameter

from pydp.densities import BinomialDensity, BetaBinomialDensity

from pydp.partition import Partition

from pydp.rvs import binomial_rvs


from pydp.samplers.atom import BetaBinomialGibbsAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import MarginalGibbsPartitionSampler, AuxillaryParameterPartitionSampler, \
    MetropolisGibbsPartitionSampler

num_iters = 10000
n = 1000
data = []

for i in range(100):
    x = binomial_rvs(n, 0.5)
    data.append(BinomialData(x, n))

for i in range(50):
    x = binomial_rvs(n, 0.6)
    data.append(BinomialData(x, n))

alpha = 1

base_measure = BetaBaseMeasure(BetaParameter(1, 1))

cluster_density = BinomialDensity()

partition = Partition()

for item, data_point in enumerate(data):
    partition.add_cell(base_measure.random())
    partition.add_item(item, item)

concentration_sampler = GammaPriorConcentrationSampler(1, 1)

posterior_density = BetaBinomialDensity()
partition_sampler = MarginalGibbsPartitionSampler(base_measure, cluster_density, posterior_density)

atom_sampler = BetaBinomialGibbsAtomSampler(base_measure, cluster_density)

for i in range(num_iters):
    alpha = concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
    
    partition_sampler.sample(data, partition, alpha)
    
    atom_sampler.sample(data, partition)
    
    if i % 100 == 0:
        print alpha, partition.cell_values

