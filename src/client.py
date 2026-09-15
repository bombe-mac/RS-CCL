import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .losses import *

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True, drop_last=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.criterion_cl = client_config["criterion_cl"]


    def client_update_cl(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data[0] = data[0].float().to(self.device)
                data[1] = data[1].float().to(self.device)

                labels =  labels.long().to(self.device)
  
                optimizer.zero_grad()
                features, labels, logits = self.model(im_q=data[0], im_k=data[1], labels=labels)
                loss = self.criterion_cl(features, labels, logits)

                loss.backward()
                optimizer.step() 

                if self.device.startswith("cuda"): torch.cuda.empty_cache()
        self.model.to("cpu")
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


