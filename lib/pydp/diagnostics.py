'''
Created on 2012-09-23

@author: Andrew Roth
'''
from pydp.stats import inverse_normal_cdf, two_sample_z_score

#=======================================================================================================================
# Convergence Checking
#=======================================================================================================================
def geweke_convergence_test(trace, first=0.1, last=0.5):
    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError("Length of intervals must sum to <= 1. Values {0} and {1} sum to {2}.".format(first,
                                                                                                       last,
                                                                                                       first + last))

    # Calculate starting indices
    end_of_first_slice = int(len(trace) * first)
    
    start_of_last_slice = int(len(trace) * (1 - last))

    # Calculate slices
    first_slice = trace[:end_of_first_slice]
    last_slice = trace[start_of_last_slice:]
    
    z = two_sample_z_score(first_slice, last_slice)
    
    return 1 - inverse_normal_cdf(z)

#=======================================================================================================================
# Model Checking
#=======================================================================================================================
def geweke_joint_distribution_test(trace_1, trace_2, g):
    '''
    Compare whether the trace of two samples have converged to the same stationary distribution.
    
    Returns :
        p : p-value two simulators are drawing from same joint distribution.
    
    ''' 
    g_1 = [g(x) for x in trace_1]
    g_2 = [g(x) for x in trace_2]
    
    z = two_sample_z_score(g_1, g_2)
    
    return 1 - inverse_normal_cdf(z)
