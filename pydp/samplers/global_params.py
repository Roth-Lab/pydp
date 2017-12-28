'''
This file is part of PyDP.

PyDP is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PyDP.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2013-04-27

@author: Andrew Roth
'''
from math import log

from pydp.rvs import uniform_rvs


class GlobalParameterSampler(object):
    '''
    Base class for samplers to update the cell values in the partition (atoms of DP).
    '''

    def __init__(self, base_measure, cluster_density):
        '''
        Args:
            base_measure : (BaseMeasure) Prior density for parameter.
            cluster_density : (Density) Cluster density for DP process.
        '''
        self.base_measure = base_measure

        self.cluster_density = cluster_density

    def sample(self, data, partition):
        '''
        Sample new values for global parameters.

        Args:
            data : (list) List of data points appropriate for cluster_density.

            partition : (Partition) Partition of dp.
        '''
        pass

#=======================================================================================================================
# Non-conjugate samplers
#=======================================================================================================================


class MetropolisHastingsGlobalParameterSampler(GlobalParameterSampler):
    '''
    Update the atom values using a Metropolis-Hastings steps with a user specified proposal function which takes
    the previous cell value as an argument.
    '''

    def __init__(self, base_measure, cluster_density, proposal_func):
        GlobalParameterSampler.__init__(self, base_measure, cluster_density)

        self.proposal_func = proposal_func

    def sample(self, data, partition):
        old_param = self.cluster_density.params
        new_param = self.proposal_func.random(old_param)

        old_ll = self.base_measure.log_p(old_param)
        new_ll = self.base_measure.log_p(new_param)

        for cell in partition.cells:
            atom_params = cell.value

            for j in cell.items:
                old_ll += self.cluster_density.log_p(data[j], atom_params)

        self.cluster_density.params = new_param

        for cell in partition.cells:
            atom_params = cell.value

            for j in cell.items:
                new_ll += self.cluster_density.log_p(data[j], atom_params)

        forward_log_ratio = new_ll - self.proposal_func.log_p(new_param, old_param)
        reverse_log_ratio = old_ll - self.proposal_func.log_p(old_param, new_param)

        log_ratio = forward_log_ratio - reverse_log_ratio

        u = uniform_rvs(0, 1)

        if log_ratio >= log(u):
            self.cluster_density.params = new_param
        else:
            self.cluster_density.params = old_param
