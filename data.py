import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

from models import *
from mmd import *
from helper import * 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try: 
    from pathlib import Path
except: 
    pass


def dummy_encode(df):
    """
   Auto encodes any dataframe column of type category or object.
   """

    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

def normalize_col(s):
    std = s.std()
    mean = s.mean()
    if std > 0:
        return (s - mean) / std
    else:
        return s - mean

def normalize_cols(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = normalize_col(df[col])
    return df
    
def reg_to_class(s):
    return (s > s.mean()).astype(int)

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


def mul_to_bin(s, border=None):
    if border is None:
        border = s.median()
    return (s > border).astype(int)


def uci_data(data_mode):

    if data_mode == 'bank':
        df = pd.read_csv('UCI//bank//bank-full.csv', sep=';')
        df['balance'] = normalize_col(df['balance'])
        df = dummy_encode(df)
        df.rename(columns={'y': 'target'}, inplace=True)

    elif data_mode == 'concrete':
        df = pd.read_excel('UCI//concrete//Concrete_Data.xls')
        df = normalize_cols(df)
        df.rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'housing':
        df = pd.read_fwf('UCI//housing//housing.data.txt', header=None)
        df = normalize_cols(df)
        df.rename(columns={13: 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'landsat':
        df = pd.read_csv('UCI//landsat//sat.trn.txt', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//landsat//sat.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(36)])
        df.rename(columns={36: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'])

    elif data_mode == 'mushroom':
        df = pd.read_csv('UCI//mushroom//agaricus-lepiota.data.txt', header=None)
        df = dummy_encode(df)
        df.rename(columns={0: 'target'}, inplace=True)

    elif data_mode == 'pageblock':
        df = pd.read_fwf('UCI//pageblock//page-blocks.data', header=None)
        df = normalize_cols(df, columns=[x for x in range(10)])
        df.rename(columns={10: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'shuttle':
        df = pd.read_csv('UCI//shuttle//shuttle.trn', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//shuttle//shuttle.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(9)])
        df.rename(columns={9: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'spambase':
        df = pd.read_csv('UCI//spambase//spambase.data.txt', header=None, sep=',')
        df = normalize_cols(df, columns=[x for x in range(57)])
        df.rename(columns={57: 'target'}, inplace=True)

    elif data_mode == 'wine':
        df = pd.read_csv('UCI//wine//winequality-red.csv', sep=';')
        df_w = pd.read_csv('UCI//wine//winequality-white.csv', sep=';')
        df['target'] = 1
        df_w['target'] = 0
        df = pd.concat([df, df_w])
        df = normalize_cols(df, [x for x in df.columns if x != 'target'])

    df_neg = df[df['target'] == 0]
    n_data = df_neg.drop(['target'], axis=1).values
    n_shuffle = np.random.permutation(len(n_data))
    n_data = n_data[n_shuffle]

    df_pos = df[df['target'] == 1]
    p_data = df_pos.drop(['target'], axis=1).values
    p_shuffle = np.random.permutation(len(p_data))
    p_data = p_data[p_shuffle]

    return p_data, n_data 

class ToyData(torchvision.datasets.CIFAR10): 
    def __init__(self): 
        margin = 0.4
        pos_x = -0
        neg_x = 2
        y_scale = 2
        X = np.array(
            [[-1, margin],
            [1, margin],
            [pos_x, y_scale],
            [2 * pos_x, 2 * y_scale],
            [-1, -margin],
            [1, -margin],
            [neg_x, -y_scale],
            [2 * neg_x, -2 * y_scale]
            ]
        )

        temp = np.concatenate((X[:4,:], X[:4,:]), axis=0)
        self.p_data = np.concatenate((temp, X[:4,:]), axis=0).astype(np.float32)    
        self.n_data = np.array(X[4:,:]).astype(np.float32)
        self.transform = None
        self.target_transform = None

    def __len__(self): 
        return len(self.n_data) + len(self.p_data)



class ToyDataContinuous(torchvision.datasets.CIFAR10): 

    def _points_on_triangle(self, v, n):
        """
        Give n random points uniformly on a triangle.

        The vertices of the triangle are given by the shape
        (2, 3) array *v*: one vertex per row.
        """
        x = np.sort(np.random.rand(2, n), axis=0)
        return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v

    def __init__(self): 
        margin = 0.1
        pos_x = -0
        neg_x = 2
        y_scale = 2
        
        p_v = np.array([(-1, margin), (2 * pos_x, 2 * y_scale), (1, margin)])
        points_p = self._points_on_triangle(p_v, 20000)

        n_v = np.array([ (-1, -margin), (1, -margin) , (2 * neg_x, -2 * y_scale) ])
        points_n =  self._points_on_triangle(n_v, 10000)

        self.p_data = np.array( points_p).astype(np.float32)    
        self.n_data = np.array(points_n).astype(np.float32)
        self.transform = None
        self.target_transform = None

    def __len__(self): 
        return len(self.n_data) + len(self.p_data)

    


class BinarizedCifarData(torchvision.datasets.CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__( root, train, transform, target_transform,
                 download)
        targets = np.array(self.targets)
        data = np.array(self.data)

        p_data_idx = np.where(targets<=4)[0]
        # p_data_idx = np.where( (targets == 0) | (targets == 1) | (targets == 8) | (targets == 9))[0]
        self.p_data = data[p_data_idx]
        
        n_data_idx = np.where(targets>4)[0]
        # n_data_idx = np.where( (targets == 2) | (targets == 3) | (targets == 4) | (targets == 5) | (targets == 6) | (targets == 7))[0]
        self.n_data = data[n_data_idx]


    def __len__(self): 
        return len(self.n_data) + len(self.p_data)
        

class DogCatData(torchvision.datasets.CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__( root, train, transform, target_transform,
                 download)
        targets = np.array(self.targets)
        data = np.array(self.data)
        
        p_data_idx = np.where(targets==3)[0]
        self.p_data = data[p_data_idx]
        
        n_data_idx = np.where(targets==5)[0]
        self.n_data = data[n_data_idx]


    def __len__(self): 
        return len(self.n_data) + len(self.p_data)


class BinarizedMNISTData(torchvision.datasets.MNIST): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
        super().__init__( root, train, transform, target_transform,
                 download)
        targets = np.array(self.targets)
        data = np.array(self.data)
        
        p_data_idx = np.where(targets<=4)[0]
        self.p_data = data[p_data_idx]
        
        n_data_idx = np.where(targets>4)[0]
        self.n_data = data[n_data_idx]


    def __len__(self): 
        return len(self.n_data) + len(self.p_data)


class OverlapMNISTData(torchvision.datasets.MNIST): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
        super().__init__( root, train, transform, target_transform,
                 download)
        targets = np.array(self.targets)
        data = np.array(self.data)
        
        p_data_idx = np.where(targets<=2)[0]
        self.p_data = data[p_data_idx]
    
        n_data_idx = np.where(targets>=7)[0]
        self.n_data = data[n_data_idx]

        overlap_idx = np.where((targets<7) & (targets>2))[0]
        self.overlap_data = data[overlap_idx]
        shuffle = np.random.permutation(len(self.overlap_data))

        self.overlap_data = self.overlap_data[shuffle]

        self.p_data = np.concatenate((self.p_data, self.overlap_data[:len(self.overlap_data)//2]), axis=0)
        self.n_data = np.concatenate((self.n_data, self.overlap_data[len(self.overlap_data)//2:]), axis=0)

        p_shuffle = np.random.permutation(len(self.p_data))
        self.p_data = self.p_data[p_shuffle]


        n_shuffle = np.random.permutation(len(self.n_data))
        self.n_data = self.n_data[n_shuffle]



    def __len__(self): 
        return len(self.n_data) + len(self.p_data)


class MNIST17Data(torchvision.datasets.MNIST): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
        super().__init__( root, train, transform, target_transform,
                 download)
        targets = np.array(self.targets)
        data = np.array(self.data)
        
        p_data_idx = np.where(targets==1)[0]
        self.p_data = data[p_data_idx]
        
        n_data_idx = np.where(targets==7)[0]
        self.n_data = data[n_data_idx]


    def __len__(self): 
        return len(self.n_data) + len(self.p_data)

class IMDbBERTData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        labels = np.array(labels)
        encodings =  {key: np.array(val) for key, val in encodings.items()}
        p_data_idx = np.where(labels==1)[0]
        n_data_idx = np.where(labels==0)[0]
        
        # print(labels)
        # # print(encodings)
        # print(p_data_idx)
        # print(n_data_idx)

        # for key, val in encodings.items(): 
        #     print(len(val))
        #     print(type(val))
            # print(val[p_data_idx])
        self.p_data = {key: val[p_data_idx] for key, val in encodings.items()}
        self.n_data = {key: val[n_data_idx] for key, val in encodings.items()}

        self.labels = labels
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)


class Gaussain_data(torch.utils.data.Dataset):
    def __init__(self, mu, sigma, size, dim):
        input_pos = np.concatenate((np.random.multivariate_normal(mu*np.ones(dim), np.eye(dim)*(sigma**2), size//2),
                        np.random.multivariate_normal(mu*np.zeros(dim), np.eye(dim)*(sigma**2), size//2)), axis=1)
        
        input_neg = np.concatenate((np.random.multivariate_normal(-mu*np.ones(dim), np.eye(dim)*(sigma**2), size//2), 
                        np.random.multivariate_normal(mu*np.zeros(dim), np.eye(dim)*(sigma**2), size//2)), axis=1)

        self.p_data  = input_pos.astype(np.float32)
        self.n_data  = input_neg.astype(np.float32)
        self.transform = None
        self.target_transform = None

    def __len__(self): 
        return len(self.n_data) + len(self.p_data)

class UCI_data(torch.utils.data.Dataset):
    def __init__(self, p_data, n_data, train=True):

        if train:
            self.p_data = p_data[ :len(p_data)*2//3].astype(np.float32)
            self.n_data = n_data[ :len(n_data)*2//3].astype(np.float32)
        else:  
            self.p_data = p_data[len(p_data)*2//3:].astype(np.float32)
            self.n_data = n_data[len(n_data)*2//3:].astype(np.float32)

        self.transform = None
        self.target_transform = None

    def __len__(self): 
        return len(self.n_data) + len(self.p_data)
 


class PosData(torch.utils.data.Dataset): 
    def __init__(self, transform=None, target_transform=None, data=None, \
            index=None, data_type=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data=data
        if data_type=="IMDb_BERT": 
            for _,b in data.items(): 
                size_pos = len(b)
            self.targets = np.zeros(size_pos, dtype= np.int_)
        else: 
            self.targets = np.zeros(data.shape[0], dtype= np.int_)
        self.data_type = data_type
        self.index = index

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, idx):
        if self.data_type == "IMDb_BERT":
            data = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
            target = torch.tensor(self.targets[idx])
            index = self.index[idx]

            return index, data, target
        else: 
            index, img, target = self.index[idx],  self.data[idx], self.targets[idx]
            
            if self.data_type == 'cifar' : 
                img = Image.fromarray(img)
            
            elif self.data_type =='mnist': 
                img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        
            return index, img, target
    

class UnlabelData(torch.utils.data.Dataset): 
    def __init__(self, transform=None, target_transform=None, pos_data=None, \
            neg_data=None, index=None, data_type=None):
        self.transform = transform
        self.target_transform = target_transform

        # import pdb; pdb.set_trace();
        if data_type =="IMDb_BERT": 
            for _,b in pos_data.items(): 
                size_pos = len(b)
            for _,b in neg_data.items(): 
                size_neg = len(b)   
            self.data = {key: np.concatenate((pos_data[key], val), axis=0) for key, val in neg_data.items()}
            self.true_targets = np.concatenate((np.zeros(size_pos,  dtype= np.int_), np.ones(size_neg,  dtype= np.int_)), axis=0)
        else: 
            self.data=np.concatenate((pos_data, neg_data), axis=0)
            self.true_targets = np.concatenate((np.zeros(pos_data.shape[0],  dtype= np.int_), np.ones(neg_data.shape[0],  dtype= np.int_)), axis=0)
        self.targets = np.ones_like(self.true_targets, dtype= np.int_)

        self.data_type = data_type
        self.index = index

    def __len__(self): 
        return len(self.targets)


    def __getitem__(self, idx):

        if self.data_type == "IMDb_BERT":
            data = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
            target = torch.tensor(self.targets[idx])
            index = self.index[idx]
            true_target = self.true_targets[idx]
            return index, data, target,true_target
        
        else:
            index, img, target, true_target = self.index[idx],  self.data[idx], self.targets[idx], self.true_targets[idx]
            
            if self.data_type == 'cifar' : 
                img = Image.fromarray(img)
            
            elif self.data_type =='mnist': 
                img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        
            return index, img, target, true_target

def get_PUDataSplits(data_obj, pos_size, alpha, beta, data_type=None): 

    unlabel_size = int((1-beta)*pos_size/beta)
    
    if data_type =="IMDb_BERT": 
        assert ((pos_size + int(unlabel_size*alpha)) < len(data_obj)), "Check sizes again"
        assert ((int(unlabel_size*(1-alpha))) < len(data_obj)), "Check sizes again"
    else: 
        assert ((pos_size + int(unlabel_size*alpha)) < len(data_obj.p_data)), "Check sizes again"
        assert ((int(unlabel_size*(1-alpha))) < len(data_obj.n_data)), "Check sizes again"

    if data_type =="IMDb_BERT": 
        pos_data = {key: val[:pos_size] for key, val in data_obj.p_data.items()}
        unlabel_pos_data = {key: val[pos_size: pos_size+ int(unlabel_size*alpha)] for key, val in data_obj.p_data.items()}
        unlabel_neg_data = {key: val[:int(unlabel_size*(1-alpha))] for key, val in data_obj.n_data.items()}


    else: 
        pos_data = data_obj.p_data[:pos_size]
        unlabel_pos_data = data_obj.p_data[pos_size: pos_size+ int(unlabel_size*alpha)]
        unlabel_neg_data = data_obj.n_data[:int(unlabel_size*(1-alpha))]

    return PosData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                data=pos_data, index=np.array(range(pos_size)), data_type=data_type), \
            UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(unlabel_size)),data_type=data_type)

def get_PNDataSplits(data_obj, pos_size, neg_size, data_type=None): 

    if data_type =="IMDb_BERT": 
        unlabel_pos_data = {key: val[:pos_size] for key, val in data_obj.p_data.items()}
        unlabel_neg_data = {key: val[:neg_size] for key, val in data_obj.n_data.items()}
        
    else:    
        unlabel_pos_data = data_obj.p_data[:pos_size]
        unlabel_neg_data = data_obj.n_data[:neg_size]

    return UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(pos_size + neg_size)),data_type=data_type)


def get_dataset(data_type,net_type, device, alpha, beta, batch_size): 

    p_trainloader=None
    u_trainloader=None
    p_validloader=None
    u_validloader=None
    net=None
    X=None
    Y=None

    if data_type=='gaussian': 
        '''
        Gaussian Data hyperparamters and data
        '''
        num_points = 6000
        input_size =200
        pos_size = 2000

        gauss_traindata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)
        gauss_testdata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)

        p_traindata, u_traindata = get_PUDataSplits(gauss_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata , u_validdata = get_PUDataSplits(gauss_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 

        net = get_model(net_type, input_dim = input_size)
        net = net.to(device)

    elif data_type=='toy_continuous':       
        '''
        Toy dataset from P vs U failure for domain discrimination
        '''

        toy_traindata = ToyDataContinuous()
        toy_testdata = ToyDataContinuous()

        pos_size = 50
        p_traindata, u_traindata = get_PUDataSplits(toy_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = get_PUDataSplits(toy_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)

    elif data_type=='toy_discrete': 

        toy_traindata = ToyData()
        toy_testdata = ToyData()

        pos_size = 8
        p_traindata, u_traindata = get_PUDataSplits(toy_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = get_PUDataSplits(toy_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)


    elif data_type=='mnist_17': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = MNIST17Data(root='./data_files', train=True, transform=transform_train)
        testdata = MNIST17Data(root='./data_files', train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=3000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)


    elif data_type=='mnist_binarized': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = BinarizedMNISTData(root='./data_files', train=True, transform=transform_train)
        testdata = BinarizedMNISTData(root='./data_files', train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=15000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='mnist_overlap': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = OverlapMNISTData(root='./data_files', train=True, transform=transform_train)
        testdata = OverlapMNISTData(root='./data_files', train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=15000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='cifar_DogCat': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = DogCatData(root='./data_files', train=True, transform=transform_train)
        testdata = DogCatData(root='./data_files', train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=2500, alpha=alpha, beta=beta,data_type='cifar')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=500, alpha=alpha, beta=beta,data_type='cifar')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=='cifar_binarized': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = BinarizedCifarData(root='./data_files', train=True, transform=transform_train)
        testdata = BinarizedCifarData(root='./data_files', train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=12500, alpha=alpha, beta=beta,data_type='cifar')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='cifar')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)

    elif data_type.startswith("UCI"): 
        
        uci_data_type = data_type.split("_")[1]
        p_data, n_data = uci_data(uci_data_type)

        keep_num = min(len(p_data), len(n_data))
        p_data = p_data[:keep_num]
        n_data = n_data[:keep_num]

        train_p_size= keep_num//3
        test_p_size= keep_num//6


        traindata = UCI_data(p_data=p_data, n_data=n_data, train=True)
        testdata = UCI_data(p_data=p_data, n_data=n_data, train=False)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=train_p_size, alpha=alpha, beta=beta,data_type='uci')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=test_p_size, alpha=alpha, beta=beta,data_type='uci')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = p_data.shape[-1])
        net = net.to(device)

    elif data_type=="IMDb_BERT": 
        train_texts, train_labels = read_imdb_split('aclImdb/train')
        test_texts, test_labels = read_imdb_split('aclImdb/test')


        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = IMDbBERTData(train_encodings, train_labels)
        test_dataset = IMDbBERTData(test_encodings, test_labels)

        p_traindata, u_traindata = get_PUDataSplits(train_dataset, pos_size=6250, alpha=alpha, beta=beta,data_type='IMDb_BERT')
        p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=5000, alpha=alpha, beta=beta,data_type='IMDb_BERT')

        X = p_traindata.targets
        Y = u_traindata.targets

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=8, \
            shuffle=True)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, \
            shuffle=True)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    
        return p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata

    return p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata

    

def get_PN_dataset(data_type,net_type, device,  alpha, beta, batch_size): 

    u_trainloader=None
    u_validloader=None
    net=None

    if data_type=='gaussian': 
        '''
        Gaussian Data hyperparamters and data
        '''
        num_points = 6000
        input_size =200
        pos_size = 2000

        gauss_traindata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)
        gauss_testdata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)

        u_traindata = get_PNDataSplits(gauss_traindata, unlabeled_size=int(num_points//2))
        u_validdata = get_PNDataSplits(gauss_traindata, unlabeled_size=int(num_points//2))


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 

        net = get_model(net_type, input_dim = input_size)
        net = net.to(device)

    elif data_type=='toy_continuous':       
        '''
        Toy dataset from P vs U failure for domain discrimination
        '''

        toy_traindata = ToyDataContinuous()
        toy_testdata = ToyDataContinuous()

        pos_size = 50
        u_traindata = get_PNDataSplits(toy_traindata, unlabeled_size=pos_size*2)
        u_validdata = get_PNDataSplits(toy_testdata, unlabeled_size=pos_size*2)

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size*2, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size*2, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)

    elif data_type=='toy_discrete': 

        toy_traindata = ToyData()
        toy_testdata = ToyData()

        pos_size = 8
        u_traindata =  get_PNDataSplits(toy_traindata, unlabeled_size=pos_size)
        u_validdata = get_PNDataSplits(toy_testdata, unlabeled_size=pos_size)


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)


    elif data_type=='mnist_17': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = MNIST17Data(root='./data_files', train=True, transform=transform_train)
        testdata = MNIST17Data(root='./data_files', train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=3000, neg_size=int(3000*(1-alpha)*(1-beta)/beta), data_type='mnist')
        u_validdata = get_PNDataSplits(testdata,pos_size=int(500*alpha), neg_size=int(500*(1-alpha)),data_type='mnist')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)


    elif data_type=='mnist_binarized': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = BinarizedMNISTData(root='./data_files', train=True, transform=transform_train)
        testdata = BinarizedMNISTData(root='./data_files', train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=15000, neg_size=int(15000*(1-alpha)*(1-beta)/beta), data_type='mnist')
        u_validdata = get_PNDataSplits(testdata, pos_size=int(2500*alpha), neg_size=int(2500*(1 - alpha)), data_type='mnist')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='cifar_DogCat': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = DogCatData(root='./data_files', train=True, transform=transform_train)
        testdata = DogCatData(root='./data_files', train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=2500, neg_size=int(2500*(1-alpha)*(1-beta)/beta), data_type='cifar')
        u_validdata = get_PNDataSplits(testdata, pos_size=int(500*alpha), neg_size=int(500*alpha), data_type='cifar')

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=='cifar_binarized': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = BinarizedCifarData(root='./data_files', train=True, transform=transform_train)
        testdata = BinarizedCifarData(root='./data_files', train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata,pos_size=12500, neg_size=int(12500*(1-alpha)*(1-beta)/beta),data_type='cifar')
        u_validdata = get_PNDataSplits(testdata,pos_size=int(2500*alpha), neg_size=int(2500*(1-alpha)),data_type='cifar')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=="IMDb_BERT": 
        train_texts, train_labels = read_imdb_split('aclImdb/train')
        test_texts, test_labels = read_imdb_split('aclImdb/test')


        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = IMDbBERTData(train_encodings, train_labels)
        test_dataset = IMDbBERTData(test_encodings, test_labels)

        u_traindata = get_PNDataSplits(train_dataset, pos_size=6250,  neg_size=int(6250*(1-alpha)*(1-beta)/beta) ,data_type='IMDb_BERT')
        u_validdata = get_PNDataSplits(test_dataset, pos_size=int(5000*alpha), neg_size=int(5000*(1-alpha)) ,data_type='IMDb_BERT')

        # X = p_traindata.targets
        # Y = u_traindata.targets

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    return u_trainloader, u_validloader, net