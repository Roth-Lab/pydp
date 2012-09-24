from __future__ import division

from collections import namedtuple
from math import lgamma as log_gamma

from pydp.base_measures import BetaBaseMeasure
from pydp.data import BetaParameter, BetaData
from pydp.densities import Density, log_binomial_pdf, log_beta_pdf
from pydp.partition import Partition
from pydp.rvs import beta_rvs
from pydp.samplers.atom import BaseMeasureAtomSampler, MetropolisHastingsAtomSampler
from pydp.samplers.concentration import GammaPriorConcentrationSampler
from pydp.samplers.partition import AuxillaryParameterPartitionSampler
from pydp.utils import log_sum_exp, memoized, SimpsonsRuleIntegrator

import csv

def main(file_name, num_iters=100000):
    data = load_pyclone_data(file_name)
    
    alpha = 1
    
    base_measure = BetaBaseMeasure(BetaParameter(1, 1))
    
    cluster_density = PyCloneDensity()
    
    partition = Partition()
    
    for item, data_point in enumerate(data):
        partition.add_cell(base_measure.random())
        partition.add_item(item, item)
    
    concentration_sampler = GammaPriorConcentrationSampler(1, 1)

    partition_sampler = AuxillaryParameterPartitionSampler(base_measure, cluster_density)
    
    atom_sampler = MetropolisHastingsAtomSampler(base_measure, cluster_density, beta_proposal)
    
    for i in range(num_iters):
        alpha = concentration_sampler.sample(alpha, partition.number_of_cells, partition.number_of_items)
        
        partition_sampler.sample(data, partition, alpha)
        
        atom_sampler.sample(data, partition)
        
        if i % 100 == 0:
            print alpha, sorted([param.x  for param in partition.cell_values])   

PyCloneData = namedtuple('PyCloneData', ['a', 'd', 'mu_r', 'mu_v', 'log_pi_r', 'log_pi_v'])

def load_pyclone_data(file_name):
    '''
    Load data from PyClone formatted input file.
    '''
    data = []
    
    reader = csv.DictReader(open(file_name), delimiter='\t')

    for row in reader:
        gene = row['gene']
        
        a = int(row['a'])
        
        d = int(row['d'])
        
        mu_r = [float(x) for x in row['mu_r'].split(',')]
        mu_v = [float(x) for x in row['mu_v'].split(',')]
        
        delta_r = [float(x) for x in row['delta_r'].split(',')]
        delta_v = [float(x) for x in row['delta_v'].split(',')]
        
        log_pi_r = get_log_mix_weights(delta_r)
        log_pi_v = get_log_mix_weights(delta_v)
        
        data.append(PyCloneData(a, d, tuple(mu_r), tuple(mu_v), tuple(log_pi_r), tuple(log_pi_v)))

    return data

def beta_proposal(params):
    s = 10
    m = params.x
    
    a = s * m
    b = s - a
    
    a += 1
    b += 1
    
    return BetaData(beta_rvs(a, b))

class PyCloneDensity(Density):
    def __init__(self):
        self.s = 100
        
        self.integrator = SimpsonsRuleIntegrator()
    
    @memoized
    def log_p(self, data, params):        
        ll = []
        
        for mu_r, log_pi_r in zip(data.mu_r, data.log_pi_r):
            for mu_v, log_pi_v in zip(data.mu_v, data.log_pi_v):
                a = mu_v * self.s
                b = self.s - a
                
                log_f = lambda x: self._log_binomial_likelihood(data.a, data.d, params.x, mu_r, x) + \
                                  log_beta_pdf(x, a, b)
                
                temp = log_pi_r + log_pi_v + self.integrator.log_integrate(log_f)
                
                ll.append(temp)
        
        return log_sum_exp(ll)
    
    def _log_binomial_likelihood(self, a, d, phi, mu_r, mu_v):
        mu = (1 - phi) * mu_r + phi * mu_v
        
        return log_binomial_pdf(a, d, mu)

def get_log_mix_weights(delta):
    log_denominator = log_gamma(sum(delta) + 1)
    
    log_mix_weights = []
    
    for i, d_i in enumerate(delta):
        log_numerator = log_gamma(d_i + 1)
        
        for j, d_j in enumerate(delta):
            if i != j:
                log_numerator += log_gamma(d_j)
        
        log_mix_weights.append(log_numerator - log_denominator)
    
    return log_mix_weights

if __name__ == "__main__":
    import sys
    
    file_name = sys.argv[1]
    
    main(file_name)
