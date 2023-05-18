import time
import numpy as np
import pandas as pd
import torch

from dijet_flow.data.plots import jet_plot_routine
from dijet_flow.utils.base import em2ptepm, logit, expit, inv_mass


class Transform:

    def __init__(self, data, args, convert_to_ptepm=True):

        self.args = args
        self.data = data  
        if convert_to_ptepm:
            self.data[:, :4] = em2ptepm(data[:, :4])    # input is in 'em' coords: (px,py,pz,m)
            self.data[:, 4:8] = em2ptepm(data[:, 4:8])   
        self.min = None
        self.max = None 
        self.mean = torch.zeros(8)
        self.std = torch.zeros(8)

    @property
    def leading(self):
        return self.data[:, :4]
    @property
    def subleading(self):
        return self.data[:, 4:8]
    @property
    def dijet(self):
        return self.data[:, :8]
    @property
    def mjj(self):
        return self.data[:, -1]
    @property
    def num_jets(self):
        return self.data.shape[0]

    def compute_mjj(self):
        self.data[:,-1] = inv_mass(self.data[:, :4], self.data[:, 4:8], coord='ptepm')
        return self

    def get_sidebands(self):
        m0, m1, m2, m3 = self.args.mass_window
        mjj = self.data[:,-1]
        self.data = self.data[ ((m0 < mjj) & (mjj < m1)) | ((m2 < mjj) & (mjj < m3))]
        return self

    def get_signal_region(self):
        _, m1, m2, _ = self.args.mass_window
        mjj = self.data[:,-1]
        self.data = self.data[(m1 <= mjj) & (mjj <= m2)]
        return self

    def normalize(self, inverse=False):
        if not inverse:
            self.max = torch.max(self.dijet)
            self.min = torch.min(self.dijet)
            self.data[:, :4] = (self.data[:, :4] - self.min) / (self.max - self.min) 
            self.data[:, 4:8] = (self.data[:, 4:8] - self.min) / (self.max - self.min)
        else:
            self.data[:, :4] = self.data[:, :4] * (self.max - self.min) + self.min
            self.data[:, 4:8] = self.data[:, 4:8] * (self.max - self.min) + self.min 
        return self

    def standardize(self, inverse=False):
        if not inverse:
            self.mean[:4] = torch.mean(self.data[:, :4], dim=0)
            self.std[:4] = torch.std(self.data[:, :4], dim=0)
            self.mean[4:] = torch.mean(self.data[:, 4:8], dim=0)
            self.std[4:] = torch.std(self.data[:, 4:8], dim=0)
            self.data[:, :4] = (self.leading - self.mean[:4]) / self.std[:4]
            self.data[:, 4:8] = (self.subleading - self.mean[4:]) / self.std[4:]
        else:
            self.data[:, :4] = self.leading * self.std[:4] + self.mean[:4]
            self.data[:, 4:8] = self.subleading * self.std[4:] + self.mean[4:]
        return self

    def logit_transform(self, alpha=1e-6, inverse=False):
        if not inverse:
            self.data[:, :4] = logit(self.data[:, :4], alpha=alpha)
            self.data[:, 4:8] = logit(self.data[:, 4:8], alpha=alpha)
        else:
            self.data[:, :4] = expit(self.data[:, :4], alpha=alpha)
            self.data[:, 4:8] = expit(self.data[:, 4:8], alpha=alpha)
        return self

    def preprocess(self, alpha=1e-6, reverse=False, verbose=True):
        if not reverse:
            if verbose: print('INFO: reversing preprocessed data')
            self.normalize()
            self.logit_transform(alpha=alpha)
            self.standardize()
        else:
            if verbose: print('INFO: preprocessing data')
            self.standardize(inverse=True)
            self.logit_transform(alpha=alpha, inverse=True)
            self.normalize(inverse=True)
        return self

    def plot_jet_features(self, title, bins=100, save_dir=None):
        if not save_dir: save_dir = self.args.workdir
        jet_plot_routine(self.data, bins=bins, title=title, save_dir=save_dir)


