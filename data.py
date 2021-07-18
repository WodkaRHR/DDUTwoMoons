import numpy as np
import sklearn.datasets
import torch
import torch.utils.data

class TwoMoonsDataset(torch.utils.data.Dataset):
    """ Two Moons Toy Dataset (interleaving half circles), generated using sklearn 
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
    """

    def __init__(self, num_samples, noise=0.1):
        """ Creates the toy dataset fully in memory. 
        
        Parameters:
        -----------
        num_samples : int
            The number of samples in the dataset.
        noise : float
            Standard deviation of noise added to perturb the data.
        """
        self.num_samples = num_samples
        self.noise = noise

        self.X, self.labels = sklearn.datasets.make_moons(num_samples, noise=noise)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]
    
    def __len__(self):
        return self.num_samples
    