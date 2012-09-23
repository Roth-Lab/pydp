'''
Created on 2012-09-23

@author: Andrew Roth
'''
from math import erf, sqrt

#=======================================================================================================================
# Descriptive Statistics
#=======================================================================================================================
def correlation(x, y):
    '''
    Compute the coefficient between the iterables x and y.
    '''
    c_xy = covariance(x, y)
    
    s_x = standard_deviation(x)
    s_y = standard_deviation(x)
    
    return c_xy / (s_x * s_y)

def covariance(x, y, sample=True):
    '''
    Compute the covariance of the iterables x and y.
    
    Args:
        x, y : (iterable) Values to compute covariance for.
    
    Kwargs:
        sample : (bool) Whether to compute the sample covariance. If true the normalisation (N-1) is used otherwise the
        normalisation N is used.
    '''
    if len(x) != len(y):
        raise Exception('Covariance can only be computed on iterables of the same length.')
    
    N = len(x)
    
    m_x = mean(x)
    m_y = mean(y)
    
    c = 0 
    
    for x_i, y_i in zip(x, y):
        c += (x_i - m_x) * (y_i - m_y)
    
    if sample:
        return c / (N - 1)
    else:
        return c / N

def mean(x):
    '''
    Compute the sample mean value of the iterable x.
    '''
    return sum(x) / len(x)

def standard_deviation(x, sample=True):
    '''
    Compute the standard deviation of the iterable x.
    
    Args:
        x : (iterable) Values to compute standard deviation for.
    
    Kwargs:
        sample : (bool) Whether to compute the sample variance. If true the normalisation (N-1) is used otherwise the
        normalisation N is used.
    '''
    v_x = variance(x, sample)
    
    return sqrt(v_x)

def variance(x, sample=True):
    '''
    Compute the variance of the iterable x.
    
    Args:
        x : (iterable) Values to compute variance for.
    
    Kwargs:
        sample : (bool) Whether to compute the sample variance. If true the normalisation (N-1) is used otherwise the
        normalisation N is used.
    '''
    return covariance(x, x, sample)

#=======================================================================================================================
# Time Series Descriptive Statistics
#=======================================================================================================================
def autocorrelation(x, lag=1):
    '''
    Compute the sample autocorrelation at a specified lag
    
    Args:
        x : (iterable) Values to compute autocorrelation for.
        lag : (int) Lag value to compute autocorrelation with.
    '''
    if lag <= 0: 
        raise Exception("Autocorrelation tag must be >= 1. Value of {0} passed.".format(lag))
    
    return correlation(x[:-lag], x[lag:], sample=False)

def autocovariance(x, lag=1):
    '''
    Compute the sample autocovariance at a specified lag.

    Args:
        x : (iterable) Values to compute autocovariance for.
        lag : (int) Lag value to compute autocovariance with.    
    '''
    if lag <= 0: 
        raise Exception("Autocorrelation tag must be >= 1. Value of {0} passed.".format(lag))
    
    return covariance(x[:-lag], x[lag:], sample=False)
    
#=======================================================================================================================
# Test Statistics
#=======================================================================================================================
def two_sample_z_score(x, y, expected_mean_x=0, expected_mean_y=0):
    '''
    Compute the two sample z-score of of two iterables x and y.
    
    Args:
        x, y : (iterable) Values to compute z-score for.
        
    Kwargs:
        expected_mean_x, expected_mean_y : Expected means for x and y respectively.
    '''
    m_x = mean(x)
    m_y = mean(y)
    
    observed_diff = m_x - m_y
    expected_diff = expected_mean_x - expected_mean_y
    
    diff = observed_diff - expected_diff
    
    v_x = variance(x, sample=False)
    v_y = variance(y, sample=False)
    
    N_x = len(x)
    N_y = len(y)
    
    norm_const = sqrt((v_x / N_x) + (v_y / N_y))
    
    return diff / norm_const

#=======================================================================================================================
# Density Related Functions
#=======================================================================================================================
def inverse_normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2)))
