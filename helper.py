import random
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
from mmd import *

try:
    from utils import progress_bar
except: 
    pass 

import matplotlib
matplotlib.use('Agg')

try:
    from transformers import DistilBertForSequenceClassification
except:
    pass

import matplotlib.pyplot as plt

def ramp_loss(out, y, device): 
    loss = torch.max(torch.min(1 - torch.mul(2*y -1, out),\
        other= torch.tensor([2],dtype=torch.float).to(device)),\
        other=torch.tensor([0],dtype=torch.float).to(device)).mean()
    return loss

def sigmoid_loss(out, y): 
    # loss = torch.gather(out, dim=1, index=y).sum()
    loss = out.gather(1, 1- y.unsqueeze(1)).mean()
    return loss


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
    elif model_type =="IMDb_BERT": 
        net = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
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

def train(epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device, show_bar=True,isIMDbBERT=False):
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)

        if isIMDbBERT:
            vinput_ids = torch.cat((p_inputs['input_ids'], u_inputs['input_ids']), dim=0)
            vinput_ids = vinput_ids.to(device)

            targets =  torch.cat((p_targets, u_targets), dim=0)
            
            vattention_mask = torch.cat((p_inputs['attention_mask'], u_inputs['attention_mask']), dim=0)
            vattention_mask = vattention_mask.to(device)

            voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=targets)
            outputs = voutputs[1]
        
        else: 
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = inputs.to(device)

            outputs = net(inputs)

        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        
        # import pdb; pdb.set_trace()

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)
        loss = (p_loss + u_loss)/2.0
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if isIMDbBERT and (batch_idx%50==49):
            break
          
    return 100.*correct/total

def train_PN(epoch, net, u_trainloader, optimizer, criterion, device, show_bar=True, isIMDbBERT=False):

    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ( _, inputs, _, targets ) in enumerate(u_trainloader):
        optimizer.zero_grad()
        
        if isIMDbBERT:
                vinput_ids = inputs['input_ids'].to(device)
                targets =  targets.to(device)                
                vattention_mask = inputs['attention_mask'].to(device)

                voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=targets)
                outputs = voutputs[1]
        
        else:
            inputs , targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar: 
            progress_bar(batch_idx, len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total


def validate(epoch, net, u_validloader, criterion, device, threshold, logistic=True, show_bar=True, separate=False,isIMDbBERT=False):
    
    if show_bar:     
        print('\nTest Epoch: %d' % epoch)
    
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0

    pos_correct = 0
    neg_correct = 0

    pos_total = 0
    neg_total = 0

    if not logistic: 
        # print("here")
        criterion = sigmoid_loss

    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            
            if isIMDbBERT:
                vinput_ids = inputs['input_ids'].to(device)
                true_targets =  true_targets.to(device)                
                vattention_mask = inputs['attention_mask'].to(device)

                voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=true_targets)
                outputs = voutputs[1]
        
            else:

                inputs , true_targets = inputs.to(device), true_targets.to(device)
                outputs = net(inputs)
            

            predicted  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] \
                    <= torch.tensor([threshold]).to(device)

            if not logistic: 
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                
            loss = criterion(outputs, true_targets)

            # import pdb; pdb.set_trace(); 


            test_loss += loss.item()
            total += true_targets.size(0)
            
            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if separate: 

                true_numpy = true_targets.cpu().numpy().squeeze()
                pos_idx = np.where(true_numpy==0)[0]
                neg_idx = np.where(true_numpy==1)[0]

                pos_correct += np.sum(correct_preds[pos_idx])
                neg_correct += np.sum(correct_preds[neg_idx])

                pos_total += len(pos_idx)
                neg_total += len(neg_idx)

            if show_bar: 
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if not separate: 
        return 100.*correct/total
    else: 
        return 100.*correct/total, 100.*pos_correct/pos_total, 100.*neg_correct/neg_total

def train_mmd(epoch, net,  p_trainloader, u_trainloader, optimizer,\
        criterion, device, lambda_ce, lambda_mmd, K_XX, K_XY, K_YY,\
        alpha, beta, sqrt=True, show_bar=True):

    if show_bar:
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_mmd_loss = 0
    train_ce_loss = 0
    correct = 0
    total = 0
    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        p_index, p_inputs, p_targets = p_data
        u_index, u_inputs, u_targets, u_true_target = u_data

        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs , targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss_ce = criterion(outputs, targets)

        p_index = p_index.numpy()
        u_index = u_index.numpy()

        batch_KXX = torch.from_numpy(K_XX[np.ix_(p_index, p_index)]).to(device)
        batch_KXY = torch.from_numpy(K_XY[np.ix_(p_index, u_index)]).to(device)
        batch_KYY = torch.from_numpy(K_YY[np.ix_(u_index, u_index)]).to(device)
        probs  = torch.nn.functional.softmax(net(u_inputs), dim=-1)[:,0]        

        scaled_probs = torch.clamp(alpha* (1- beta)/ beta * probs / (1-probs), min=0.0,max=1.0)


        # print(scaled_probs[:20])
        # print(targets[200:220])
        # print(u_true_target[:20])
        loss_mmd = mix_rbf_mmd2(batch_KXX, batch_KXY, batch_KYY, probs=scaled_probs)
        # print(loss_mmd)
        # print(1.0 - u_true_target)
        # print(scaled_probs)
        # import pdb; pdb.set_trace()


        loss_mmd = F.relu(loss_mmd)

        if sqrt: 
            loss_mmd = torch.sqrt(loss_mmd) 

        # print(loss_mmd)
        # print(loss_ce)
        loss = loss_mmd*lambda_mmd  + lambda_ce*loss_ce


        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_mmd_loss += loss_mmd.item() 
        train_ce_loss += loss_ce.item() 

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | CE Loss: %.3f | MMD Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), train_ce_loss/(batch_idx+1), \
                    train_mmd_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total

def validate_transformed(epoch, net, u_validloader, criterion, device, alpha, beta, show_bar=True):

    if show_bar:
        print('\nTest Epoch: %d' % epoch)
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)
            probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] 
            scaled_probs = alpha* (1- beta)/ beta * probs / (1-probs)
            predicted = scaled_probs <= torch.tensor([0.5]).to(device)

            loss = criterion(outputs, true_targets)
           
            test_loss += loss.item()
            total += true_targets.size(0)
            
            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if show_bar: 
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total

def train_PU_discard(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, keep_sample=None, show_bar=True,isIMDbBERT=False):
    
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        
        optimizer.zero_grad()
        
        _, p_inputs, p_targets = p_data
        u_index, u_inputs, u_targets, u_true_targets = u_data

        u_idx = np.where(keep_sample[u_index.numpy()]==1)[0]
        # u_idx = np.where(keep_sample[u_index.numpy()]==1)[0]
        # print(len(u_idx))
        if len(u_idx) <1: 
            # print("here")
            continue
        u_targets = u_targets[u_idx]

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)
        

        if isIMDbBERT:
            vinput_ids = torch.cat((p_inputs['input_ids'], u_inputs['input_ids'][u_idx]), dim=0)
            vinput_ids = vinput_ids.to(device)

            targets =  torch.cat((p_targets, u_targets), dim=0)
            
            vattention_mask = torch.cat((p_inputs['attention_mask'], u_inputs['attention_mask'][u_idx]), dim=0)
            vattention_mask = vattention_mask.to(device)

            voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=targets)
            outputs = voutputs[1]
        
        else: 
            u_inputs = u_inputs[u_idx]        
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = inputs.to(device)

            outputs = net(inputs)

        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        
        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)

        loss = (p_loss + u_loss)/2.0

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

        if show_bar:
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if isIMDbBERT and (batch_idx%50==49):
            break

    return 100.*correct/total

def rank_inputs(_, net, u_trainloader, device, alpha, u_size,isIMDbBERT=False):

    net.eval() 
    output_probs = np.zeros(u_size)
    keep_samples = np.ones_like(output_probs)
    true_targets_all = np.zeros(u_size)
    # if alpha >=0.99: 
    #     alpha = 0.0

    with torch.no_grad():
        for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
            idx = idx.numpy()
            
            if isIMDbBERT:
                vinput_ids = inputs['input_ids'].to(device)
                vlabels =  true_targets.to(device)                
                vattention_mask = inputs['attention_mask'].to(device)

                voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=vlabels)
                outputs = voutputs[1]
            
            else:
                inputs = inputs.to(device)
                outputs = net(inputs)

            # import pdb; pdb.set_trace(); 

            probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0]         
            output_probs[idx] = probs.detach().cpu().numpy().squeeze()
            true_targets_all[idx] = true_targets.numpy().squeeze()

            # print(probs)
            # print(true_targets)

    # print(batch_num)
    sorted_idx = np.argsort(output_probs)

    keep_samples[sorted_idx[u_size - int(alpha*u_size):]] = 0

    neg_reject = np.sum(true_targets_all[sorted_idx[u_size - int(alpha*u_size):]]==1.0)

    neg_reject = neg_reject/ int(alpha*u_size)
    return keep_samples, neg_reject

def train_PU_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha,logistic=True, show_bar=True, isIMDbBERT=False):
    
    if show_bar:
        print('\nTrain Epoch: %d' % epoch)
        
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if not logistic: 
        criterion = sigmoid_loss

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()

        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets_sub = torch.ones_like(p_targets)
        p_targets, p_targets_sub, u_targets = p_targets.to(device), p_targets_sub.to(device), u_targets.to(device)


        if isIMDbBERT:
            vinput_ids = torch.cat((p_inputs['input_ids'], u_inputs['input_ids']), dim=0)
            vinput_ids = vinput_ids.to(device)

            targets =  torch.cat((p_targets, u_targets), dim=0)
            
            vattention_mask = torch.cat((p_inputs['attention_mask'], u_inputs['attention_mask']), dim=0)
            vattention_mask = vattention_mask.to(device)

            voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=targets)
            outputs = voutputs[1]

        else: 
            p_inputs , u_inputs = p_inputs.to(device), u_inputs.to(device)
            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = torch.cat((p_inputs, u_inputs), axis=0)
            outputs = net(inputs)
        
        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]

        if not logistic: 
            p_outputs = torch.nn.functional.softmax(p_outputs, dim=-1) 
            u_outputs = torch.nn.functional.softmax(u_outputs, dim=-1)

        loss_pos = criterion(p_outputs, p_targets)
        loss_pos_neg = criterion(p_outputs, p_targets_sub)
        loss_unl = criterion(u_outputs, u_targets)

        # import pdb; pdb.set_trace()
        loss = alpha * (loss_pos - loss_pos_neg) + loss_unl

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if isIMDbBERT and (batch_idx%50==49):
            break

    return 100.*correct/total


def train_PU_nn_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha, logistic=True, show_bar=True,isIMDbBERT=False):
    
    if show_bar:
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if not logistic: 
        criterion = sigmoid_loss

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets_sub = torch.ones_like(p_targets)
        p_targets, p_targets_sub, u_targets = p_targets.to(device), p_targets_sub.to(device), u_targets.to(device)


        if isIMDbBERT:
            vinput_ids = torch.cat((p_inputs['input_ids'], u_inputs['input_ids']), dim=0)
            vinput_ids = vinput_ids.to(device)

            targets =  torch.cat((p_targets, u_targets), dim=0)
            
            vattention_mask = torch.cat((p_inputs['attention_mask'], u_inputs['attention_mask']), dim=0)
            vattention_mask = vattention_mask.to(device)

            voutputs = net(vinput_ids, attention_mask=vattention_mask, labels=targets)
            outputs = voutputs[1]
        else: 
            p_inputs , u_inputs = p_inputs.to(device), u_inputs.to(device)

            targets =  torch.cat((p_targets, u_targets), dim=0)
            inputs = torch.cat((p_inputs, u_inputs), axis=0)
            outputs = net(inputs)
        
        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]

        if not logistic: 
            p_outputs = torch.nn.functional.softmax(p_outputs, dim=-1) 
            u_outputs = torch.nn.functional.softmax(u_outputs, dim=-1)

            # print(p_outputs)

        loss_pos = criterion(p_outputs, p_targets)
        loss_pos_neg = criterion(p_outputs, p_targets_sub)
        loss_unl = criterion(u_outputs, u_targets)


        if torch.gt((loss_unl - alpha* loss_pos_neg ), 0):
            loss = alpha * (loss_pos - loss_pos_neg) + loss_unl
        else: 
            loss = alpha * loss_pos_neg - loss_unl
        
        loss.backward()

        optimizer.step()
        # loss = alpha * (loss_pos - loss_pos_neg) + loss_unl
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar:
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if isIMDbBERT and (batch_idx%50==49):
            break

    return 100.*correct/total

def train_PU_unbiased_hinge(epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device, alpha, show_bar=True):
    
    if show_bar:

        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data


        p_inputs , u_inputs = p_inputs.to(device), u_inputs.to(device)
        p_targets, u_targets = p_targets.to(device), u_targets.to(device)

        targets =  torch.cat((p_targets, u_targets), dim=0)

        optimizer.zero_grad()

        p_outputs = net(p_inputs)[:,0]
        u_outputs = net(u_inputs)[:,0]

        outputs = torch.cat((p_outputs, u_outputs), dim=0)
        
        loss_pos = ramp_loss(p_outputs, p_targets, device)
        loss_unl = ramp_loss(u_outputs, u_targets, device)

        loss = alpha * loss_pos + (1 - alpha)* loss_unl
        
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item()

        predicted = outputs > torch.tensor([0]).to(device)

        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar:
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total

def validate_hinge(epoch, net, u_validloader, criterion, device, threshold, show_bar=True):
    if show_bar:
    
        print('\nTest Epoch: %d' % epoch)
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)[:,0]

            loss = ramp_loss(outputs, true_targets, device)
           
            test_loss += loss.item()
            total += true_targets.size(0)
            
            predicted = outputs > torch.tensor([0]).to(device)

            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)
            
            if show_bar:
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return 100.*correct/total


def visualize(pos_data, u_data, net, device, alpha, beta, name):

    l=2.0
    fc=15

    X = np.random.uniform(size=10000)*4.0 - 2.0
    X = X.reshape((5000,2)).astype(np.float32)

    inputs = torch.tensor(X).to(device)

    outputs = net(inputs)
    probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] 
    scaled_probs = alpha* (1- beta)/ beta * probs / (1-probs)

    scaled_probs = scaled_probs.detach().cpu().numpy().squeeze()
    idx = np.where( (scaled_probs <= 0.52) & (scaled_probs >=0.48))[0]

    plt.scatter(X[:,0], X[:,1], c=scaled_probs,alpha=0.5)

    plt.scatter(X[idx,0], X[idx,1], c='black', marker='.')

    plt.scatter(pos_data.data[:, 0], pos_data.data[:, 1], color='blue')
    plt.scatter(u_data.data[4:, 0], u_data.data[4:, 1], color='orange')
    plt.plot([-1,1],[0,0], '--', linewidth=l, color='black', label="true separator")

    plt.legend(loc="best", frameon=True, prop={'size': fc})

    plt.xticks(fontsize=fc)
    plt.yticks(fontsize=fc)

    plt.grid()
    plt.savefig(name + ".png",transparent=True,bbox_inches='tight')
    plt.clf()
    # plt.show()