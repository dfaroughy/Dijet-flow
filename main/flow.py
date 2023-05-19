import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import copy
from copy import deepcopy
import argparse
import json
import h5py
import pandas as pd

from dijet_flow.utils.base import make_dir, copy_parser, save_arguments, get_gpu_memory
from dijet_flow.models.flows.norm_flows import masked_autoregressive_flow, coupling_flow
from dijet_flow.models.training import Model
from dijet_flow.models.loss import neglogprob_loss
from dijet_flow.data.transform import EventTransform 

sys.path.append("../")
torch.set_default_dtype(torch.float64)


'''
    Description:

    Normalizing flow (maf or coupling layer) for learning the 
    features of the dijet distribution for lhco data. 

    The flow generates context for a diffusion model for anomaly detection.

 
'''

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the flow model')

params.add_argument('--device',       default='cuda:2',           help='where to train')
params.add_argument('--dim',          default=8,                  help='dim of data: (pT1,eta1,phi1,m1,pT2,eta2,phi2,m2)', type=int)
params.add_argument('--loss',         default=neglogprob_loss,    help='loss function')

#...flow params:

params.add_argument('--flow',         default='coupling',   help='type of flow model: coupling or MAF', type=str)
params.add_argument('--dim_flow',     default=8,            help='dimension of input features for flow, usually same as --dim', type=int)
params.add_argument('--flow_func',    default='RQSpline',   help='type of flow transformation: affine or RQSpline', type=str)
params.add_argument('--coupl_mask',   default='mid-split',  help='mask type [only for coupling flows]: mid-split or checkers', type=str)
params.add_argument('--permutation',  default='inverse',    help='type of fixed permutation between flows: n-cycle or inverse', type=str)
params.add_argument('--num_flows',    default=32,           help='num of flow layers', type=int)
params.add_argument('--dim_hidden',   default=128,          help='dimension of hidden layers', type=int)
params.add_argument('--num_spline',   default=30,           help='num of spline for rational_quadratic', type=int)
params.add_argument('--num_blocks',   default=2,            help='num of MADE blocks in flow', type=int)
params.add_argument('--dim_context',  default=1,            help='dimension of context features', type=int)

#...training params:

params.add_argument('--batch_size',    default=2048,         help='size of training/testing batch', type=int)
params.add_argument('--num_steps',     default=0,            help='split batch into n_steps sub-batches + gradient accumulation', type=int)
params.add_argument('--test_size',     default=0.2,          help='fraction of testing data', type=float)
params.add_argument('--max_epochs',    default=20,           help='max num of training epochs', type=int)
params.add_argument('--max_patience',  default=20,           help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',            default=1e-4,         help='learning rate of generator optimizer', type=float)
params.add_argument('--activation',    default=F.leaky_relu, help='activation function for neural networks')
params.add_argument('--batch_norm',    default=True,         help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',       default=0.1,          help='dropout probability', type=float)
params.add_argument('--seed',          default=999,          help='random seed for data split', type=int)

#... data params:

params.add_argument('--mass_window', default=(0,3300,3700,13000), help='bump hunt mass window: SB1, SR, SB2', type=tuple)
params.add_argument('--num_gen',     default=10000,                  help='number of sampled events from model', type=int)
# params.add_argument('--mean',        default=[],                    help='data mean (for preprocessing)', type=list)
# params.add_argument('--std',         default=[],                    help='data covariance (for preprocessing)', type=list)
# params.add_argument('--min',         default=None,                  help='data min (for preprocessing)', type=float)
# params.add_argument('--max',         default=None,                  help='data max (for preprocessing)', type=float)


####################################################################################################################

if __name__ == '__main__':

    #...create working folders 

    args = params.parse_args()
    args.workdir = make_dir('Results_dijet_density', sub_dirs=['data_plots', 'results_plots'], overwrite=False)

    #...get datasets

    file =  "./data/events_anomalydetection_v2.features_with_jet_constituents.h5"
    data = torch.tensor(pd.read_hdf(file).to_numpy())
    data = torch.cat((data[:, :4], data[:, 7:11], torch.unsqueeze(data[:, -2], dim=1)), dim=1)  # d=9: (jet1, jet2, mjj)

    #...get SB events and preprocess data

    events_bckg = EventTransform(data, args)
    events_bckg.get_sidebands()
    events_bckg.preprocess()

    #...store parser arguments

    args.num_jets = events_bckg.num_jets
    args.num_gen = events_bckg.num_jets
    args.mean = events_bckg.mean.tolist()
    args.std = events_bckg.std.tolist()
    args.max = events_bckg.max.tolist()
    args.min = events_bckg.min.tolist()
    print("INFO: num sideband jets: {}".format(args.num_jets))
    save_arguments(args, name='inputs.json')

    # #...Prepare train/test samples from sidebands

    train, test  = train_test_split(events_bckg.data, test_size=args.test_size, random_state=args.seed)

    # #...define model

    if args.flow == 'MAF': flow = masked_autoregressive_flow(args)
    elif args.flow == 'coupling': flow = coupling_flow(args)
    flow = flow.to(args.device)
    model = Model(flow, args) 

    #...train flow for density estimation.

    train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args.batch_size, shuffle=False) 
    model.train(train_sample, test_sample)
    
    #...get context form signal region

    context = EventTransform(data, args)
    context.get_signal_region()
    context.plot_jet_features(title='jet features truth SR', save_dir=args.workdir+'/data_plots')

    # sample from model with mjj context:

    sample = model.sample(context=context.mjj)
    sample = torch.cat((sample, torch.zeros(sample.shape[0],1)), dim=1)
    sample_SR = EventTransform(sample, args, convert_to_ptepm=False)
    sample_SR.mean = events_bckg.mean
    sample_SR.std = events_bckg.std
    sample_SR.min = events_bckg.min
    sample_SR.max = events_bckg.max
    sample_SR.preprocess(reverse=True)
    sample_SR.compute_mjj()
    sample_SR.plot_jet_features(title='jet features generated SR', save_dir=args.workdir+'/results_plots')



