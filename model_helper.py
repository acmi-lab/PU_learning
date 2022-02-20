import random
from models.bert import initialize_bert_based_model
import numpy as np
import sys
from numpy.core.numeric import False_ 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *


def get_model(model_type, input_dim=None): 
    if model_type == 'FCN':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 2, bias=True)
        )
        return net
    elif model_type == 'UCI_FCN':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 2, bias=True)
        )
        return net
    elif model_type == 'linear':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 2, bias=True),
        )
        return net
    elif model_type == 'ResNet':
        net = ResNet18(num_classes=2)
        return net
    elif model_type == 'LeNet':
        net = LeNet(num_classes=2)
        return net
    elif model_type == 'AllConv': 
        net = AllConv()
        return net
    elif model_type == "DistilBert":
        net = initialize_bert_based_model("distilbert-base-uncased", num_classes=2)
        return net 
    else:
        print("Model type must be one of FCN | CNN | linear ... ")
        sys.exit(0)


def train_penultimate(net, model_type): 
    if model_type == 'FCN': 
        for param in net.parameters(): 
            param.requires_grad = False

        for param in net.module[-1].parameters():
            param.requires_grad = True

    
    return net
