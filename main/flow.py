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

from dijet_flow.utils.base import make_dir, copy_parser, save_arguments
from dijet_flow.models.flows.norm_flows import masked_autoregressive_flow, coupling_flow
from dijet_flow.models.training import Model
from dijet_flow.models.loss import neglogprob_loss
from dijet_flow.data.transform import Transform

sys.path.append("../")
# torch.set_default_dtype(torch.float64)


'''
    Description:

    Normalizing flow (maf or coupling layer) for learning the 
    features of the dijet distribution for lhco data. 

    The flow generates context for a diffusion model for anomaly detection.

    tips:



'''

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the flow model')

params.add_argument('--workdir')
params.add_argument('--device',       default='cuda:1',           help='where to train')
params.add_argument('--dim',          default=8,                  help='dimensionalaty of data: (x,y,z,vx,vy,vz)', type=int)
params.add_argument('--loss',         default=neglogprob_loss,    help='loss function')

#...flow params:

params.add_argument('--flow',         default='coupling',   help='type of flow model: coupling or MAF', type=str)
params.add_argument('--dim_flow',     default=8,            help='dimension of input features for flow, usually same as --dim', type=int)
params.add_argument('--flow_func',    default='RQSpline',   help='type of flow transformation: affine or RQSpline', type=str)
params.add_argument('--coupl_mask',   default='mid-split',  help='mask type [only for coupling flows]: mid-split or checkers', type=str)
params.add_argument('--permutation',  default='inverse',    help='type of fixed permutation between flows: n-cycle or inverse', type=str)
params.add_argument('--num_flows',    default=10,            help='num of flow layers', type=int)
params.add_argument('--dim_hidden',   default=128,          help='dimension of hidden layers', type=int)
params.add_argument('--num_spline',   default=30,           help='num of spline for rational_quadratic', type=int)
params.add_argument('--num_blocks',   default=2,            help='num of MADE blocks in flow', type=int)
params.add_argument('--dim_context',  default=None,         help='dimension of context features', type=int)

#...training params:

params.add_argument('--batch_size',    default=1024,          help='size of training/testing batch', type=int)
params.add_argument('--num_steps',     default=0,            help='split batch into n_steps sub-batches + gradient accumulation', type=int)
params.add_argument('--test_size',     default=0.2,          help='fraction of testing data', type=float)
params.add_argument('--max_epochs',    default=1000,         help='max num of training epochs', type=int)
params.add_argument('--max_patience',  default=20,           help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',            default=1e-4,         help='learning rate of generator optimizer', type=float)
params.add_argument('--activation',    default=F.leaky_relu, help='activation function for neural networks')
params.add_argument('--batch_norm',    default=True,         help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',       default=0.1,          help='dropout probability', type=float)

#... data params:

params.add_argument('--mass_window',  default=(3000,3300,
                                               3400,3700),  help='mass window: SB1, SR, SB2', type=tuple)

# params.add_argument('--',       default=               help='', type=)
# params.add_argument('--',       default=               help='', type=)
# params.add_argument('--',       default=               help='', type=)
# params.add_argument('--',       default=               help='', type=)
# params.add_argument('--',       default=               help='', type=)
# params.add_argument('--',       default=               help='', type=)



####################################################################################################################

if __name__ == '__main__':

    #...create working folders and save args

    args = params.parse_args()
    args.workdir = make_dir('Results_Dijets_density', sub_dirs=['data_plots', 'results_plots'], overwrite=True)
    print("#================================================")
    print("INFO: working directory: {}".format(args.workdir))
    print("#================================================")

    #...get datasets, preprocess them

    data_file =  "./data/events_anomalydetection_v2.features_with_jet_constituents.h5"
    lhco = pd.read_hdf(data_file)
    lhco = torch.tensor(lhco.to_numpy())

    dijets = Transform(lhco, args)

    dijets.get_sidebands() # dijets from both Side bands
    dijets.logit() # dijets from both Side bands

    dijets.get_sidebands() # dijets from both Side bands

    print(dijets.leading[0], dijets.subleading[0], dijets.jj[0])
    #...smear and preprocess data    

    # dijet = Transform(data)






    # gaia.get_stars_near_sun(self, R=args.radius)

    # if args.data == 'noisy': gaia.smear()

    # gaia.plot('x', title='target positions', save_dir=args.workdir+'/data_plots') 
    # gaia.plot('v', title='target velocities', save_dir=args.workdir+'/data_plots') 
    
    # gaia.preprocess()

    # #...store parser arguments

    # args.num_stars = gaia.num_stars
    # args.mean = gaia.mean.tolist()
    # args.std = gaia.std.tolist()
    # args.Rmax = gaia.Rmax.tolist()
    # print("INFO: num stars: {}".format(args.num_stars))
    # save_arguments(args, name='inputs.json')

    # #...Prepare train/test samples

    # train, test  = train_test_split(gaia.data, test_size=args.test_size, random_state=9999)
    # train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    # test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args.batch_size, shuffle=False)

    # #...define model

    # if args.flow == 'MAF': flow = masked_autoregressive_flow(args)
    # elif args.flow == 'coupling': flow = coupling_flow(args)

    # flow = flow.to(args.device)

    # #...train flow for phase-space density estimation.

    # Train_Model(flow, train_sample, test_sample, args , show_plots=False, save_best_state=False)
    
    # #...sample from flow model

    # sample = sampler(flow, num_samples=args.num_gen)

    # #...transofrm back to phase-space amd plot

    # gaia_sample = GaiaTransform(sample, torch.zeros(sample.shape), args) 
    # gaia_sample.mean = gaia.mean
    # gaia_sample.std =  gaia.std
    # gaia_sample.preprocess(R=gaia.Rmax, reverse=True) # invert preprocess transformations
    # gaia_sample.plot('x', title='position density', save_dir=args.workdir+'/results_plots') 
    # gaia_sample.plot('v', title='velocity density', save_dir=args.workdir+'/results_plots') 

    # 