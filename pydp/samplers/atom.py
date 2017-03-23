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

from math import log

from pydp.data import BetaData, GammaData, GaussianGammaData
from pydp.proposal_functions import BaseMeasureProposalFunction
from pydp.rvs import beta_rvs, gamma_rvs, uniform_rvs, gaussian_rvs
from pydp.stats import mean, variance


class AtomSampler(object):
    '''
    Base class for samplers to update the cell values in the partition (atoms of DP).
    '''

    def __init__(self, base_measure, cluster_density):
        '''
        Args:
            base_measure : (BaseMeasure) Base measure for DP process.

            cluster_density : (Density) Cluster density for DP process.
        '''
        self.base_measure = base_measure

        self.cluster_density = cluster_density

    def sample(self, data, partition):
        '''
        Sample a new value for atoms in the partition. The partition passed in will be updated in place.

        Args:
            data : (list) List of data points appropriate for cluster_density.

            partition : (Partition) Partition of dp.
        '''
        for cell in partition.cells:
            cell.value = self.sample_atom(data, cell)

    def sample_atom(self, data, cell):
        '''
        Sample a new value for the atom associated with the cell. Returns a suitable value for the cell.
        '''
        raise NotImplemented

#=======================================================================================================================
# Non-conjugate samplers
#=======================================================================================================================


class MetropolisHastingsAtomSampler(AtomSampler):
    '''
    Update the atom values using a Metropolis-Hastings steps with a user specified proposal function which takes
    the previous cell value as an argument.
    '''

    def __init__(self, base_measure, cluster_density, proposal_func):
        AtomSampler.__init__(self, base_measure, cluster_density)

        self.proposal_func = proposal_func

    def sample_atom(self, data, cell):
        old_param = cell.value
        new_param = self.proposal_func.random(old_param)

        old_ll = self.base_measure.log_p(old_param)
        new_ll = self.base_measure.log_p(new_param)

        for j in cell.items:
            old_ll += self.cluster_density.log_p(data[j], old_param)
            new_ll += self.cluster_density.log_p(data[j], new_param)

        forward_log_ratio = new_ll - self.proposal_func.log_p(new_param, old_param)
        reverse_log_ratio = old_ll - self.proposal_func.log_p(old_param, new_param)

        log_ratio = forward_log_ratio - reverse_log_ratio

        u = uniform_rvs(0, 1)

        if log_ratio >= log(u):
            return new_param
        else:
            return old_param


class BaseMeasureAtomSampler(MetropolisHastingsAtomSampler):
    '''
    Update the atom values using a Metropolis-Hastings steps with the base measure as a proposal density.
    '''

    def __init__(self, base_measure, cluster_density):
        proposal_func = BaseMeasureProposalFunction(base_measure)

        MetropolisHastingsAtomSampler.__init__(self, base_measure, cluster_density, proposal_func)

#=======================================================================================================================
# Conjugate samplers
#=======================================================================================================================


class BetaBinomialGibbsAtomSampler(AtomSampler):
    '''
    Update the partition values using a Gibbs step. 

    Requires a Beta base measure and binomial data.
    '''

    def sample_atom(self, data, cell):
        a = self.base_measure.params.a
        b = self.base_measure.params.b

        for item in cell.items:
            a += data[item].x
            b += data[item].n - data[item].x

        return BetaData(beta_rvs(a, b))


class GammaPoissonGibbsAtomSampler(AtomSampler):
    '''
    Update the partition values using a Gibbs step. 

    Requires a Gamma base measure and Poisson data.
    '''

    def sample_atom(self, data, cell):
        a = self.base_measure.params.a

        for item in cell.items:
            a += data[item].x

        n = cell.size
        b = self.base_measure.params.b

        b = b / (n * b + 1)

        return GammaData(gamma_rvs(a, b))


class GaussianGammaGaussianAtomSampler(AtomSampler):
    '''
    Update the partition values using a Gibbs step. 

    Requires a GammaGaussian base measure and GammaGaussian data.
    '''

    def sample_atom(self, data, cell):
        sample_size = cell.size

        sample_mean = mean([data[item].x for item in cell.items])

        sample_variance = variance([data[item].x for item in cell.items], sample=False)

        posterior_precision = self._sample_precision(sample_size, sample_mean, sample_variance)

        posterior_mean = self._sample_mean(sample_size, sample_mean, sample_variance, posterior_precision)

        return GaussianGammaData(posterior_mean, posterior_precision)

    def _sample_mean(self, sample_size, sample_mean, sample_variance, tau):
        prior_size = self.base_measure.params.size

        prior_mean = self.base_measure.params.mean

        posterior_precision = prior_size * tau + sample_size * tau

        posterior_mean = (prior_size * tau * prior_mean) / posterior_precision + \
                         (sample_size * tau * sample_mean) / posterior_precision

        return gaussian_rvs(posterior_mean, posterior_precision)

    def _sample_precision(self, sample_size, sample_mean, sample_variance):
        prior_alpha = self.base_measure.params.alpha

        prior_beta = self.base_measure.params.beta

        prior_mean = self.base_measure.params.mean

        prior_size = self.base_measure.params.size

        posterior_alpha = prior_alpha + sample_size / 2

        posterior_beta = prior_beta + \
            (sample_size * sample_variance) / 2 + \
            ((prior_size * sample_size) / (2 * (prior_size + sample_size))) * \
            (sample_mean - prior_mean) ** 2

        return gamma_rvs(posterior_alpha, posterior_beta)
