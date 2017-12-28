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
from math import log, sqrt
from random import betavariate as beta_rvs, gammavariate as _gamma_rvs, normalvariate as _normal_rvs, \
    uniform as uniform_rvs

from pydp.utils import log_sum_exp


def bernoulli_rvs(p):
    '''
    Return a Bernoulli distributed random variable.

    Args:
        p : (float) Probability of success.

    Returns:
        x : (int) Binary indicator of success/failure.
    '''
    u = uniform_rvs(0, 1)

    if u <= p:
        return 1
    else:
        return 0


def beta_binomial_rvs(n, a, b):
    p = beta_rvs(a, b)

    x = binomial_rvs(n, p)

    return x


def binomial_rvs(n, p):
    '''
    Sample a binomial distributed random variable.

    Args:
        n : (int) Number of trials performed.
        p : (int) Probability of success for each trial.

    Returns:
        x : (int) Number of successful trials.
    '''
    if p > 0.5:
        return n - binomial_rvs(n, 1 - p)

    if p == 0:
        return 0

    u = uniform_rvs(0, 1)
    log_u = log(u)

    log_c = log(p) - log(1 - p)

    i = 0

    log_prob = n * log(1 - p)

    log_F = log_prob

    while True:
        if log_u < log_F:
            return i

        log_prob += log_c + log(n - i) - log(i + 1)

        log_F = log_sum_exp([log_F, log_prob])

        i += 1


def dirichlet_rvs(alpha):
    '''
    Sample a Dirichlet distributed random variable.

    Args:
        alpha : (list) Pseudo count parameter for Dirichlet distribution.

    Returns:
        pi : (list) List of probabilities for each class such that sum(pi) == 1.
    '''
    g = [gamma_rvs(a, 1) for a in alpha]

    norm_const = sum(g)

    return [x / norm_const for x in g]


def discrete_rvs(p):
    '''
    Sample a discrete (Categorical) random variable.

    Args:
        p : (list) Probabilities for each class from 0 to len(p) - 1

    Returns:
        i : (int) Id of class sampled. 
    '''
    total = 0

    u = uniform_rvs(0, 1)

    for i, p_i in enumerate(p):
        total += p_i

        if u < total:
            break

    return i


def gamma_rvs(a, b):
    '''
                        a ** b    x ** (a - 1) * math.exp(-x * b)
            pdf(x) =  ----------------- 
                      math.gamma(a) 
    '''
    shape = a

    scale = 1 / b

    value = _gamma_rvs(shape, scale)

    if value < 1e-100:
        value = 1e-100

    return value


def multinomial_rvs(n, p):
    x = [0 for _ in p]

    if len(p) / n > 1:
        for _ in range(n):
            index = discrete_rvs(p)

            x[index] += 1
    else:
        total = 0
        denom = 1

        for i, p_i in enumerate(p):
            if p_i <= 0:
                continue

            # Need min to avoid numeirc issues
            p_scale = min(p_i / denom, 1)

            x[i] = binomial_rvs(n - total, p_scale)

            denom -= p_i

            total += x[i]

            if total == n:
                break

    return x


def gaussian_rvs(mean, precision):
    '''
    Draw a random variable from a univariate Gaussian (normal) distribution.

    Args:
        mean : (float) The mean value of distribution.
        precision: (float > 0) The precision (inverse of variance) of the distribution.
    '''
    std_dev = 1 / sqrt(precision)

    return _normal_rvs(mean, std_dev)


def normal_rvs(mean, precision):
    return gaussian_rvs(mean, precision)


def poisson_rvs(l):
    u = uniform_rvs(0, 1)
    log_u = log(u)

    i = 0

    log_l = log(l)

    log_p = -l

    log_F = log_p

    while True:
        if log_u < log_F:
            return i

        log_p += log_l - log(i + 1)

        log_F = log_sum_exp([log_F, log_p])

        i += 1


def inverse_sample_rvs(log_f, a, b, mesh_size=100):
    '''
    Sample from a continuous univariate density using the inverse transform method.

    Args:
        log_f : (function) A function which computes the unnormalised of the density.
        a : (float) Left side of the support for the denstiy.
        b : (float) Right side of the support for the density.

    Kwargs:
        mesh_size : (int) How many points to use to approximate the integral for computing CDF.

    Returns:
        x : (float) Sampled value
        log_q : (float) The value of the density at x.
    '''
    u = uniform_rvs(0, 1)

    log_u = log(u)

    step_size = (b - a) / mesh_size

    log_step_size = log(b - a) - log(mesh_size)

    knots = [i * step_size + a for i in range(0, mesh_size + 1)]

    mid_points = [(x_l + x_r) / 2 for x_l, x_r in zip(knots[:-1], knots[1:])]

    log_likelihood = [log_f(x) for x in mid_points]

    log_riemann_sum = []

    for y in log_likelihood:
        log_riemann_sum.append(y + log_step_size)

    log_norm_const = log_sum_exp(log_riemann_sum)

    log_cdf = None

    for x, y in zip(mid_points, log_likelihood):
        log_q = y - log_norm_const

        log_partial_pdf_riemann_sum = log_q + log_step_size

        if log_cdf is None:
            log_cdf = log_partial_pdf_riemann_sum
        else:
            log_cdf = log_sum_exp([log_cdf, log_partial_pdf_riemann_sum])

        if log_u < log_cdf:
            break

    return x, log_q
