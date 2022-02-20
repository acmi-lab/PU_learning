import numpy as np

def estimator_CM_EN(pdata_probs, pudata_probs):
    return np.sum(pudata_probs)*len(pdata_probs)/len(pudata_probs)/np.sum(pdata_probs)

def estimator_prob_EN(pdata_probs):
    return np.sum(pdata_probs, axis=0)/len(pdata_probs)

def estimator_max_EN(pdata_probs, pudata_probs):
    return np.max(np.concatenate((pdata_probs, pudata_probs) ))
