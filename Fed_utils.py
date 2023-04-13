from tkinter import N
import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from torch.utils.data import DataLoader
import random

from train import Trainer
from train_rcil import Trainer_rcil
from apex.parallel import DistributedDataParallel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def local_train(args, clients, index, model_g, current_step, ep_g):
    
    clients[index].beforeTrain(args, current_step)
    if args.base_weights == False:
        if args.use_entropy_detection==True:
            clients[index].update_entropy_signal(model_g)
        local_model = clients[index].train(args, model_g, ep_g)
    else:
        local_model = None

    return local_model

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg

def model_global_eval(args, model_g, test_loader, current_step, val_metrics,device,rank):

    tmp_model_g = copy.deepcopy(model_g)

    tmp_model_g = DistributedDataParallel(tmp_model_g.cuda(device))

    if args.incremental_method != 'RCIL':
        trainer = Trainer(tmp_model_g, None, device=device, rank=rank,opts=args, step=current_step)
    else:
        trainer = Trainer_rcil(tmp_model_g, None, device=device, rank=rank,opts=args, step=current_step)


    tmp_model_g.eval()

    _, val_score, _ = trainer.validate(
        loader=test_loader, metrics=val_metrics, end_task=True
    )

    if rank==0:
        print(val_metrics.to_str(val_score))

    tmp_model_g = tmp_model_g.to('cpu')
    torch.cuda.empty_cache() 

    del tmp_model_g
    del trainer
   
    return val_score

