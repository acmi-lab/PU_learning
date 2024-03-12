import random
import numpy as np
import sys 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from baselines import * 
from data_helper import *
from models import *

from helper import * 
from model_helper import * 

def p_probs(net, device, p_loader): 
    net.eval()
    pp_probs = None
    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(p_loader):
           
            inputs = inputs.to(device)
            outputs = net(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=-1)[:,0] 
#             probs = torch.stack((probs, 1-probs), dim=1)
#             probs = probs.to(torch.int32)
            if pp_probs is None: 
                pp_probs = probs.detach().cpu().numpy().squeeze()
            else:
                pp_probs = np.concatenate((pp_probs, \
                    probs.detach().cpu().numpy().squeeze()), axis=0)
    
    return pp_probs    

def u_probs(net, device, u_loader):
    net.eval()
    pu_probs = None
    pu_targets = None
    with torch.no_grad():
        for batch_idx, (_, inputs, _, targets) in enumerate(u_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
                
            probs = torch.nn.functional.softmax(outputs, dim=-1) 
            
            if pu_probs is None: 
                pu_probs = probs.detach().cpu().numpy().squeeze()
                pu_targets = targets.numpy().squeeze()
                
            else:
                pu_probs = np.concatenate( (pu_probs, \
                    probs.detach().cpu().numpy().squeeze()))
                pu_targets = np.concatenate( (pu_targets, \
                    targets.numpy().squeeze()))
                
    
    return pu_probs, pu_targets

def DKW_bound(x,y,t,m,n,delta=0.1, gamma= 0.01):

    temp = np.sqrt(np.log(4/delta)/2/n) + np.sqrt(np.log(4/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound


def BBE_estimator(pdata_probs, udata_probs, udata_targets):

    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    u_indices = np.argsort(udata_probs[:,0])
    sorted_u_probs = udata_probs[:,0][u_indices]
    sorted_u_targets = udata_targets[u_indices]

    sorted_u_probs = sorted_u_probs[::-1]
    sorted_p_probs = sorted_p_probs[::-1]
    sorted_u_targets = sorted_u_targets[::-1]
    num = len(sorted_u_probs)

    estimate_arr = []

    upper_cfb = []
    lower_cfb = []            

    i = 0
    j = 0
    num_u_samples = 0

    while (i < num):
        start_interval =  sorted_u_probs[i]   
        k = i 
        if (i<num-1 and start_interval> sorted_u_probs[i+1]): 
            pass
        else: 
            i += 1
            continue
        if (sorted_u_targets[i]==1):
            num_u_samples += 1

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
            estimate, lower , upper = DKW_bound(i, j, t, len(sorted_u_probs), len(sorted_p_probs))
            estimate_arr.append(estimate)
            upper_cfb.append( upper)
            lower_cfb.append( lower)
        i+=1

    if (len(upper_cfb) != 0): 
        idx = np.argmin(upper_cfb)
        mpe_estimate = estimate_arr[idx]

        return mpe_estimate, lower_cfb, upper_cfb
    else: 
        return 0.0, 0.0, 0.0





