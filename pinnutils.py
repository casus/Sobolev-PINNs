import os
import time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn

from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

# network definition

class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

class BatchNorm(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self, x):
        return (x-self.mean)/self.std

class Layer(nn.Module):
    def __init__(self, in_features, out_features, seed, activation):
        super(Layer, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
            
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x)

class PINN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.bn = BatchNorm(mean, std)
        
        layer = []
        for i in range(len(sizes)-2):
            linear = Layer(sizes[i], sizes[i+1], seed, activation)
            layer += [linear, activation]
            
        layer += [Layer(sizes[-2], sizes[-1], seed, activation)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(self.bn(x))
    
# dynamic weighting methods
    
def loss_grad_std(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return torch.std(grad_)

def loss_grad_max(loss, net, lambg=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
    
    grad_ = torch.abs(lambg*grad_)
        
    return torch.max(grad_), torch.mean(grad_)

def network_gradient(loss, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    grad_ = torch.zeros((0), dtype=torch.float32, device=device)
    for elem in grad(loss, net.parameters(), retain_graph=True):
        grad_ = torch.cat((grad_, elem.view(-1)))
        
    return grad_

def reweight(d_, num_tasks, eps):
    nz_ind = np.where(d_>eps)
    z_ind  = np.delete(np.arange(num_tasks),nz_ind)
    if(len(z_ind)==0):
        return d_
    d_[nz_ind] = d_[nz_ind] - eps/len(d_[nz_ind])
    d_[z_ind]  = eps/len(z_ind)
    return d_

def mgda_solver(Q, num_tasks, tol, maxiter=500):
    alphas = (1./num_tasks)*np.ones((num_tasks,))
    direct = np.zeros((num_tasks,2))

    for it in range(0, maxiter):
        ind_vec = np.zeros((num_tasks,));
        grad = Q @ alphas
        idx_oracle   = np.argmin(grad);
        ind_vec[idx_oracle] = 1.0;

        direct[:,0] = ind_vec; direct[:,1] = alphas;
        MM = (direct.T @ Q) @ direct

        if(MM[0,1] >= MM[0,0]):
            step_size = 1.0;
        elif(MM[0,1] >= MM[1,1]):
            step_size = 0;
        else:
            step_size = (MM[1,1] - MM[0,1])/(MM[0,0] + MM[1,1] - MM[0,1] - MM[1,0])

        alphas = (1. - step_size) * alphas
        alphas[idx_oracle] = alphas[idx_oracle] + step_size * ind_vec[idx_oracle]

    return reweight(alphas,num_tasks,tol)

# data loader

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
