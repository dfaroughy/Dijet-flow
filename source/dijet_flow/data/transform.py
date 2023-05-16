import time
import numpy as np
import pandas as pd
import torch

from dijet_flow.data.plots import plot_data_projections


class Transform:

    def __init__(self, data, args):

        self.args = args
        self.data = data    
        self.mean = torch.zeros(args.dim)
        self.std = torch.zeros(args.dim)

    @property
    def leading(self):
        return self.data[:, :4]
    @property
    def subleading(self):
        return self.data[:, 7:11]
    @property
    def jj(self):
        return np.concatenate((self.data[:, :4], self.data[:, 7:11]), axis=1)
    @property
    def mass(self):
        return self.data[:, -2]
    @property
    def truth_label(self):
        return self.data[:, -1]
    @property
    def num_events(self):
        return self.data.shape[0]

    def get_sidebands(self):
        m0, m1, m2, m3 = self.args.mass_window
        mjj = self.data[:,-2]
        self.data = self.data[ ((m0 < mjj) & (mjj < m1)) | ((m2 < mjj) & (mjj < m3))]
        return self


    def logit(self, alpha=1e-6, inverse=False):
        
        if not inverse:
            self.jj = alpha + (1 - 2 * alpha) * self.jj
            self.jj = torch.log(self.jj/(1-self.jj))

        return self