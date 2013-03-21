from pydp.base_measures import GaussianGammaBaseMeasure
from pydp.data import GaussianData
from pydp.densities import GaussianDensity
from pydp.rvs import gaussian_rvs
from pydp.trace import MemoryTrace

from pydp.samplers.atom import GaussianGammaGaussianAtomSampler
from pydp.samplers.dp import DirichletProcessSampler
from pydp.samplers.partition import AuxillaryParameterPartitionSampler

size = 10
n = 100

num_iters = int(1e2)
burnin = int(1e1)
thin = int(1)

data = [GaussianData(gaussian_rvs(-100, 2)) for _ in range(100)] + [GaussianData(gaussian_rvs(100, 100)) for _ in range(100)] 

base_measure = GaussianGammaBaseMeasure(0, 1, 1, 1)

cluster_density = GaussianDensity()

atom_sampler = GaussianGammaGaussianAtomSampler(base_measure, cluster_density)

partition_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)

sampler = DirichletProcessSampler(atom_sampler, partition_sampler)

trace = MemoryTrace()

sampler.sample(data, trace, num_iters)

print [x.mean for x in trace.params[-1]]