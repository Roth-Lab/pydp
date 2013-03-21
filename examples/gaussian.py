from pydp.base_measures import GaussianGammaBaseMeasure
from pydp.data import GaussianData, GaussianGammaParameter
from pydp.densities import GaussianDensity
from pydp.partition import Partition
from pydp.rvs import gaussian_rvs

from pydp.samplers.atom import BaseMeasureAtomSampler, GaussianGammaGaussianAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import AuxillaryParameterPartitionSampler

class Sampler(object):
    def __init__(self, base_measure, cluster_density):
        self.base_measure = base_measure
        
        self.cluster_density = cluster_density
        
        self.concentration_sampler = GammaPriorConcentrationSampler(1, 1)
        
        posterior_density = GaussianDensity()
        
        self.partition_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)
        
        self.atom_sampler = BaseMeasureAtomSampler(base_measure, cluster_density)
            
    def sample(self, data, num_iters=100):
        partition = Partition()
        
        for item, data_point in enumerate(data):
            partition.add_cell(self.base_measure.random())
            partition.add_item(item, item)
        
        alpha = 1
        
        for i in range(num_iters):
            self.atom_sampler.sample(data, partition)
            
            self.partition_sampler.sample(data, partition, alpha)
            
            alpha = self.concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
        
        return alpha, partition

size = 10
n = 100

num_iters = int(1e3)
burnin = int(1e1)
thin = int(1)

data = [GaussianData(gaussian_rvs(-100, 100)) for _ in range(10)] + [GaussianData(gaussian_rvs(100, 100)) for _ in range(10)] 

base_measure = GaussianGammaBaseMeasure(0, 1, 1, 1)

cluster_density = GaussianDensity()

sampler = Sampler(base_measure, cluster_density)

alpha, partition = sampler.sample(data, num_iters)

print [x.value.mean for x in partition.cells]
print [x.value.precision for x in partition.cells]