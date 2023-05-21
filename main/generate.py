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

from dijet_flow.utils.base import make_dir, copy_parser, save_arguments, shuffle
from dijet_flow.models.flows.norm_flows import masked_autoregressive_flow, coupling_flow
from dijet_flow.models.training import Model
from dijet_flow.models.loss import neglogprob_loss
from dijet_flow.data.transform import EventTransform 
from dijet_flow.data.plots import jet_plot_routine

sys.path.append("../")
torch.set_default_dtype(torch.float64)

#########################################

params = argparse.ArgumentParser()
params.add_argument('--dir', type=str)

#########################################

if __name__ == '__main__':

    #...get model params
    args = params.parse_args()
    make_dir(args.dir+'/plots', overwrite=True)
    with open(args.dir + '/inputs.json', 'r') as f: model_inputs = json.load(f)
    args = argparse.Namespace(**model_inputs)
    args.activation = getattr(F, args.activation)

    #...get datasets

    file =  "./data/events_anomalydetection_v2.features_with_jet_constituents.h5"
    data = torch.tensor(pd.read_hdf(file).to_numpy())
    data = torch.cat((data[:, :4], data[:, 7:11], data[:, -2:]), dim=1)  # d=9: (jet1, jet2, mjj, truth_label)
    data = shuffle(data)
    context, bckg_truth = train_test_split(data, test_size=0.5, random_state=2487162)

    #...get SB events and preprocess data

    bckg_truth = EventTransform(bckg_truth, args)
    bckg_truth.get_truth(kind='background')
    bckg_truth.get_signal_region()

    context = EventTransform(context, args)
    context.get_signal_region()
    context.preprocess()

    args.num_jets = context.num_jets
    args.num_gen = context.num_jets
    args.mean = context.mean.tolist()
    args.std = context.std.tolist()
    args.max = context.max.tolist()
    args.min = context.min.tolist()
    print("INFO: num context jets: {}".format(args.num_jets))

    #...define template model

    if args.flow == 'MAF': flow = masked_autoregressive_flow(args)
    elif args.flow == 'coupling': flow = coupling_flow(args)
    flow = flow.to(args.device)
    model = Model(flow, args) 

    #...load model

    model.load_state(path=args.workdir+'/best_model.pth')

    #...define template model

    sample = model.sample(num_batches=10, context=context.mjj[:bckg_truth.num_jets])

    sample = torch.cat((sample, torch.zeros(sample.shape[0],1)), dim=1)
    sample = EventTransform(sample, args, convert_to_ptepm=False)
    sample.mean = context.mean
    sample.std = context.std
    sample.min = context.min
    sample.max = context.max
    sample.preprocess(reverse=True)
    sample.compute_mjj()

    jet_plot_routine((sample.data, bckg_truth.data), 
                     title='jet features generated SR ', save_dir=args.workdir+'/plots')




