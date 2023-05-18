
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from dijet_flow.data.transform import Transform

def calculate_loss(model, data, args, loss_func=None, reduction=torch.mean):
	if not loss_func: loss_func = args.loss
	loss = reduction(loss_func(model, data, args))
	return loss

def neglogprob_loss(model, batch, args):
	if args.context: 
		context = batch[:, args.dim:]
		context = context.to(args.device)
	else: context = None
	batch = batch[:, :args.dim]
	batch = batch.to(args.device)
	loss = - model.log_prob(input=batch, context=context)
	return loss
