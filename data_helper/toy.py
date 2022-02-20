
import torchvision
import numpy as np
import torch 



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

    