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
