
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy

from dijet_flow.models.loss import calculate_loss
from dijet_flow.data.plots import plot_loss
from dijet_flow.data.transform import EventTransform

class Model:

    def __init__(self, models, args):
        super(Model, self).__init__()
        self.model = models
        self.args = args

    def train(self, training_sample, validation_sample, show_plots=True, save_best_state=True):        
        train = Train_Step(self.model, self.args)
        test = Test_Step(self.model, self.args)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs)
        print("INFO: start training") 
        print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        for epoch in tqdm(range(self.args.max_epochs), desc="epochs"):
            train.fit(training_sample, optimizer)       
            test.validate(validation_sample)
            scheduler.step() 
            if test.check_patience(show_plots=show_plots, save_best_state=save_best_state): 
                break
        plot_loss(train, test, self.args)
        torch.cuda.empty_cache()
        return test.best_model

    @torch.no_grad()
    def sample(self, num_samples=10000, num_batches=1, context=False):

        if torch.is_tensor(context): 
            num_samples = context.shape[0]
            chunks = torch.tensor_split(context, num_batches)
        else: 
            chunks = [False] * num_batches

        print("INFO: generating {} jets from model".format(num_samples))

        n, r = divmod(num_samples, num_batches)
        num_samples = [n] * num_batches
        num_samples[-1] += r
        samples=[]

        for i in range(num_batches):

            num = num_samples[i]
            chunk = chunks[i]
            chunk = chunk.to(self.args.device) if torch.is_tensor(context) else None
            batch_sample = self.model.sample(num_samples=(1 if torch.is_tensor(context) else num), context=chunk)
            samples.append(batch_sample.cpu().detach())

        samples = torch.squeeze(torch.cat(samples, dim=0), 1)
        return samples

    def load_state(self, path):
        self.model.load_state_dict(torch.load(path))
        return self

class Train_Step(nn.Module):

    def __init__(self, model, args):
        super(Train_Step, self).__init__()
        self.model = model
        self.loss = 0
        self.loss_per_epoch = []
        self.args = args

    def fit(self, data, optimizer):
        self.model.train()
        self.loss = 0
        for batch in tqdm(data, desc=" batch"):
            if self.args.num_steps <= 1: 
                loss_current = calculate_loss(self.model, batch, self.args)
                loss_current.backward()
                optimizer.step()  
                optimizer.zero_grad()
                self.loss += loss_current.item() / len(data)
            else: 
                # sub-batch and accumulate gradient (use if data does not fit in GPU memory)  
                sub_batches = torch.tensor_split(batch, self.args.num_steps)
                sub_batch_loss = 0
                for sub_batch in tqdm(sub_batches, desc="  sub-batch"):
                    loss_current = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum)
                    loss_current.backward()
                    sub_batch_loss += loss_current.item() / self.args.batch_size
                optimizer.step()
                optimizer.zero_grad()
                self.loss += sub_batch_loss / len(data) 
        self.loss_per_epoch.append(self.loss)
        print("\t Training loss: {}".format(self.loss))

class Test_Step(nn.Module):

    def __init__(self, model, args):
        super(Test_Step, self).__init__()
        self.model = model
        self.loss = 0
        self.loss_per_epoch = []
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.best_model = None
        self.terminate = False
        self.args = args

    @torch.no_grad()
    def validate(self, data):
        self.model.eval()
        self.loss = 0
        self.epoch += 1
        for batch in data:
            if self.args.num_steps <= 1: 
                loss_current = calculate_loss(self.model, batch, self.args)
                self.loss += loss_current.item() / len(data)
            else:
                sub_batches = torch.tensor_split(batch, self.args.num_steps)
                sub_batch_loss = 0
                for sub_batch in sub_batches:
                    loss_current = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum) 
                    sub_batch_loss += loss_current.item() / self.args.batch_size
                self.loss += sub_batch_loss / len(data)
        self.loss_per_epoch.append(self.loss)

    @torch.no_grad()
    def check_patience(self, show_plots=True, save_best_state=True):
        self.model.eval()
        if self.loss < self.loss_min:
            self.loss_min = self.loss
            self.patience = 0
            self.best_model = deepcopy(self.model)
            if save_best_state:
                torch.save(self.best_model.state_dict(), self.args.workdir + '/best_model.pth')       
        else: self.patience += 1
        if self.patience >= self.args.max_patience: self.terminate = True
        print("\t Test loss: {}  (min: {})".format(self.loss, self.loss_min))
        return self.terminate
