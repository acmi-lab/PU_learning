import numpy as np 
from scipy.stats import binom


def bininv(m,e,c): 
    plo = 0
    phi = 1
    p = .5

    max_iter = 20 
    tol = c*0.001   

    iter = 0;

    while iter <= max_iter:
        iter = iter + 1;
        bintail = binom.cdf(e,m,p)
        if abs(bintail - c) <= tol: 
            return p

        if bintail < c: 
            phi = p
        else: 
            plo = p
        p = (phi + plo)/2

    return p

def scott_estimator(pdata_probs, udata_probs):
    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    u_indices = np.argsort(udata_probs[:,0])
    sorted_u_probs = udata_probs[:,0][u_indices]

    sorted_u_probs = sorted_u_probs[::-1]
    sorted_p_probs = sorted_p_probs[::-1]
    num = len(sorted_u_probs)

    j = 0
    num_u_samples = 0
    ratios = []
    delta=0.1
    n = len(sorted_u_probs)
    m = len(sorted_p_probs)
    i = 0
    while (i < num):
        start_interval =  sorted_u_probs[i]   
        k = i 
        if (i<num-1 and start_interval> sorted_u_probs[i+1]): 
            pass
        else: 
            i += 1
            continue

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = bininv(n, i, delta) / (1.0 - bininv(m, m-j, delta))
            # t = ((i*1.0/n) + np.sqrt(np.log(1/delta)/2/n))/( (j*1.0/m) - + np.sqrt(np.log(1/delta)/2/m))
            # if t > 0: 
            ratios.append(t)
        i+=1

    if len(ratios)!= 0: 
        return np.min(ratios)
    else: 
        return 0.0
