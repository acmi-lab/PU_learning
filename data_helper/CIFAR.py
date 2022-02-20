import torchvision
import numpy as np 

class BinarizedCifarData(torchvision.datasets.CIFAR10): 
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
        

class DogCatData(torchvision.datasets.CIFAR10): 
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True):
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


