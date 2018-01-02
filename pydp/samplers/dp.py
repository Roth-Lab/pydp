'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2013-03-21

@author: Andrew Roth
'''
from collections import OrderedDict

from pydp.partition import Partition
from pydp.samplers.concentration import GammaPriorConcentrationSampler


class DirichletProcessSampler(object):

    def __init__(self, atom_sampler, partition_sampler, alpha=1.0, alpha_priors=None, global_params_sampler=None):
        self.atom_sampler = atom_sampler

        self.partition_sampler = partition_sampler

        self.alpha = alpha

        if alpha_priors is None:
            self.update_alpha = False

        else:
            self.update_alpha = True

            self.concentration_sampler = GammaPriorConcentrationSampler(
                alpha_priors['shape'], alpha_priors['rate']
            )

        if global_params_sampler is None:
            self.update_global_params = False

        else:
            self.update_global_params = True

            self.global_params_sampler = global_params_sampler

        self.num_iters = 0

        self.partition = None

    @property
    def state(self):
        return {
            'alpha': self.alpha,
            'labels': self.partition.labels,
            'params': [param for param in self.partition.item_values],
            'global_params': self.atom_sampler.cluster_density.params
        }

    @state.setter
    def state(self, value):
        self.alpha = value['alpha']

        self.partition = Partition()

        cell_idx = 0

        label_to_cell = {}

        for item, (label, param) in enumerate(zip(value['labels'], value['params'])):
            if label not in label_to_cell:
                label_to_cell[label] = cell_idx

                cell_idx += 1

                self.partition.add_cell(param)

            self.partition.add_item(item, label_to_cell[label])

        self.atom_sampler.cluster_density.params = value.get('global_params', None)

    def initialise_partition(self, init_method, num_data_points):
        """ Initialize the partition of data points.

        Parameters
        ----------
        init_method: str
            How to initialise the partition. Options are `connected` which puts all data points in the same cluster or
                `disconnected` which puts them all data points in separate clusters.
        num_data_points: int
            Number of data points in the partition.

        """
        self.partition = Partition()

        if init_method == 'disconnected':
            for item in range(num_data_points):
                self.partition.add_cell(self.partition_sampler.base_measure.random())

                self.partition.add_item(item, item)

        elif init_method == 'connected':
            self.partition.add_cell(self.partition_sampler.base_measure.random())

            for item in range(num_data_points):
                self.partition.add_item(item, 0)

    def sample(self, data, trace, num_iters, init_method='disconnected', print_freq=100):
        self.initialise_partition(data, init_method)

        for i in range(num_iters):
            if i % print_freq == 0:
                print(self.num_iters, self.partition.number_of_cells, self.alpha)

                if self.update_global_params:
                    params = self.atom_sampler.cluster_density.params

                    if isinstance(params, OrderedDict):
                        print(','.join([str(x[0]) for x in list(self.atom_sampler.cluster_density.params.values())]))

                    elif isinstance(params, tuple):
                        print(params[0])

                    else:
                        raise Exception('Object type {0} is not a valid cluster parameter'.format(type(params)))

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

        if self.update_global_params:
            self.global_params_sampler.sample(data, self.partition)
