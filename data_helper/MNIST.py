import numpy as np
import torchvision


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
