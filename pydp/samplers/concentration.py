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
from math import log

from pydp.rvs import beta_rvs, discrete_rvs, gamma_rvs


class ConcentrationSampler(object):
    '''
    Base class for samplers to update the concentration parameter of the DP.
    '''

    def sample(self, old_value, num_cells, num_items):
        '''
        Args:
            old_value : (float) Previous value of concentration parameter.
            num_clusters : (int) Number of cells (clusters) in partition.
            num_items : (int) Number of items (data_points) in partition.
        '''
        pass


class GammaPriorConcentrationSampler(ConcentrationSampler):
    '''
    Gibbs update assuming a gamma prior on the concentration parameter.
    '''

    def __init__(self, a, b):
        '''
        Args :
            a : (float) Shape parameter of the gamma prior.
            b : (float) Rate parameter of the gamma prior.
        '''
        self.a = a
        self.b = b

    def sample(self, old_value, num_clusters, num_data_points):
        a = self.a
        b = self.b

        k = num_clusters
        n = num_data_points

        eta = beta_rvs(old_value + 1, n)

        x = (a + k - 1) / (n * (b - log(eta)))

        pi = x / (1 + x)

        label = discrete_rvs([pi, 1 - pi])

        scale = b - log(eta)

        if label == 0:
            new_value = gamma_rvs(a + k, scale)
        else:
            new_value = gamma_rvs(a + k - 1, scale)

        return new_value
