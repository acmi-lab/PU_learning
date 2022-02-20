import torch
from utils import progress_bar
import numpy as np


def ramp_loss(out, y, device): 
    loss = torch.max(torch.min(1 - torch.mul(2*y -1, out),\
        other= torch.tensor([2],dtype=torch.float).to(device)),\
        other=torch.tensor([0],dtype=torch.float).to(device)).mean()
    return loss

def sigmoid_loss(out, y): 
    # loss = torch.gather(out, dim=1, index=y).sum()
    loss = out.gather(1, 1- y.unsqueeze(1)).mean()
    return loss


def train_PN(epoch, net, u_trainloader, optimizer, criterion, device, show_bar=True):

    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ( _, inputs, _, targets ) in enumerate(u_trainloader):
        optimizer.zero_grad()
        
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


def validate(epoch, net, u_validloader, criterion, device, threshold, logistic=True, show_bar=True, separate=False):
    
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
            
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)
            

            predicted  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] \
                    <= torch.tensor([threshold]).to(device)

            if not logistic: 
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                
            loss = criterion(outputs, true_targets)

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



def train(epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device, show_bar=True):
    
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
            progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

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

def train_PU_discard(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, keep_sample=None, show_bar=True):
    
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

        if len(u_idx) <1: 
            continue

        u_targets = u_targets[u_idx]

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)
        

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

    return 100.*correct/total

def rank_inputs(_, net, u_trainloader, device, alpha, u_size):

    net.eval() 
    output_probs = np.zeros(u_size)
    keep_samples = np.ones_like(output_probs)
    true_targets_all = np.zeros(u_size)

    with torch.no_grad():
        for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
            idx = idx.numpy()
            
            inputs = inputs.to(device)
            outputs = net(inputs)


            probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0]         
            output_probs[idx] = probs.detach().cpu().numpy().squeeze()
            true_targets_all[idx] = true_targets.numpy().squeeze()

    sorted_idx = np.argsort(output_probs)

    keep_samples[sorted_idx[u_size - int(alpha*u_size):]] = 0

    neg_reject = np.sum(true_targets_all[sorted_idx[u_size - int(alpha*u_size):]]==1.0)

    neg_reject = neg_reject/ int(alpha*u_size)
    return keep_samples, neg_reject

def train_PU_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha,logistic=True, show_bar=True):
    
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

    return 100.*correct/total


def train_PU_nn_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha, logistic=True, show_bar=True):
    
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
        
    return 100.*correct/total