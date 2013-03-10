'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.
'''

from pydp.base_measures import GammaBaseMeasure

from pydp.data import PoissonData, GammaParameter

from pydp.densities import PoissonDensity, NegativeBinomialDensity

from pydp.partition import Partition

from pydp.rvs import poisson_rvs

from pydp.samplers.atom import GammaPoissonGibbsAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import MarginalGibbsPartitionSampler, AuxillaryParameterPartitionSampler, \
    MetropolisGibbsPartitionSampler

num_iters = 10000
n = 1000
data = []

for i in range(1000):
    x = poisson_rvs(100)
    data.append(PoissonData(x))

for i in range(50):
    x = poisson_rvs(25)
    data.append(PoissonData(x))

alpha = 1

base_measure = GammaBaseMeasure(1, 1)

cluster_density = PoissonDensity()

partition = Partition()

for item, data_point in enumerate(data):
    partition.add_cell(base_measure.random())
    partition.add_item(item, item)

concentration_sampler = GammaPriorConcentrationSampler(1, 1)

posterior_density = NegativeBinomialDensity()
partition_sampler = MarginalGibbsPartitionSampler(base_measure, cluster_density, posterior_density)

#partition_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)

atom_sampler = GammaPoissonGibbsAtomSampler(base_measure, cluster_density)

for i in range(num_iters):
    alpha = concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
    
    partition_sampler.sample(data, partition, alpha)
    
    atom_sampler.sample(data, partition)
    
    if i % 100 == 0:
        print alpha, partition.cell_values

