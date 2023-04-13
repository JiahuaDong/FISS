from re import S
from selectors import EpollSelector
from signal import signal
from tkinter import N
import torch.nn as nn
import torch
from myNetwork import *
from myNetwork_rcil import *
from Fed_utils import * 
import tasks

from apex import amp
from apex.parallel import DistributedDataParallel
from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from dataset import (AdeSegmentationIncremental,
                     VOCSegmentationIncremental, transform)

import utils
from train import Trainer
from train_rcil import Trainer_rcil

import time



def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot


def entropy(probabilities):

    entropy = -probabilities * torch.log(probabilities + 1e-8)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_trainset(opts, step, client_index, class_ratio, sample_ratio2, sample_ratio1):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose(
        [
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    labels, labels_old, _ = tasks.get_task_labels(opts.dataset, opts.task, step)

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    else:
        raise NotImplementedError

    # if opts.overlap:
    #     path_base += "-ov"

    # path_base = path_base + f"/seed_{opts.seed}"

    # if not os.path.exists(path_base):
    #     os.makedirs(path_base, exist_ok=True)
        
    train_dst = dataset(
        root=opts.data_root,
        train=True,
        transform=train_transform,
        labels=list(labels),
        labels_old=list(labels_old),
        masking=not opts.no_mask,
        overlap=opts.overlap,
        disable_background=opts.disable_background,
        data_masking=opts.data_masking,
        test_on_val=opts.test_on_val, 
        class_ratio = class_ratio,
        sample_ratio2 = sample_ratio2,
        sample_ratio1 = sample_ratio1,
        step=step
    )

    return train_dst


class FC_model:

    def __init__(self, client_index, batch_size, num_workers, loss_de, pod, world_size, rank, device, entropy_threshold):

        super(FC_model, self).__init__()
   

        self.last_learning_rate = 0.0 

        self.client_index = client_index 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_de = loss_de
        self.pod = pod


        self.old_model = None

        self.learned_step = -1

        self.signal = False 

        self.current_trainset = None  
    

        self.device = device
        self.world_size = world_size
        self.rank = rank


        self.last_entropy = -1
        self.cur_epoch = 0

        self.trainer_state = None

        self.entropy_threshold = entropy_threshold

    # get incremental train data
    def beforeTrain(self, args, current_step):
        if self.rank==0:
            print("Current Client Index: ", self.client_index)

        if current_step != self.learned_step: 
            self.learned_step = current_step 

            self.current_trainset = get_trainset(args, self.learned_step, self.client_index, args.class_ratio, args.sample_ratio2, args.sample_ratio1)

            if args.use_entropy_detection==False:
                self.signal = True # rule

        
    def update_entropy_signal(self, model_g):
        
        
        tmp_model = copy.deepcopy(model_g)
        
        tmp_model = DistributedDataParallel(tmp_model.cuda(self.device))
        tmp_model.eval()
        tmp_model.module.in_eval = True

        train_loader = data.DataLoader(
            self.current_trainset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.current_trainset, num_replicas=self.world_size, rank=self.rank),
            num_workers=self.num_workers,
            drop_last=True
            )

        self.signal = self.entropy_signal(tmp_model,train_loader)

        tmp_model = tmp_model.to('cpu')

        torch.cuda.empty_cache() 

        del tmp_model

        del train_loader


    # train model
    def train(self,args, model_g, ep_g): 

        model = copy.deepcopy(model_g)

        params = []  
        if not args.freeze: 
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                    'weight_decay': args.weight_decay
                }
            )

        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.head.parameters()),
                'weight_decay': args.weight_decay
            }
        )

        if args.lr_old is not None and self.learned_step > 0:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[:-1].parameters()),
                    'weight_decay': args.weight_decay,
                    "lr": args.lr_old * args.lr
                }
            )
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[-1:].parameters()),
                    'weight_decay': args.weight_decay
                }
            )
        else:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                    'weight_decay': args.weight_decay
                }
            )

        if model.scalar is not None:
            params.append({"params": model.scalar, 'weight_decay': args.weight_decay})
            

        if self.signal:
            self.cur_epoch = 0
            if ep_g // args.steps_global == 0:
                self.last_learning_rate = args.lr1
            else:
                self.last_learning_rate = args.lr2

        optimizer = torch.optim.SGD(params, lr=self.last_learning_rate, momentum=0.9, nesterov=True)

        train_loader = data.DataLoader(
            self.current_trainset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.current_trainset, num_replicas=self.world_size, rank=self.rank),
            num_workers=self.num_workers,
            drop_last=True
            )

        if args.lr_policy == 'poly': 
            scheduler = utils.PolyLR(
                optimizer, max_iters=(args.epochs_local *(args.steps_global-(ep_g%args.steps_global))) * len(train_loader), power=args.lr_power
            )  
        elif args.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor
            )
        else:
            raise NotImplementedError

        if self.signal:

            step = self.learned_step -1 
            ckpt_path = f"{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{step}.pth"
     

            if ckpt_path is not None and os.path.exists(ckpt_path):
                if self.rank==0:
                    print('load old model')

                if args.name != 'RCIL':
                    self.old_model = make_model(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step))  
                else:
                    self.old_model = make_model_rcil(args, classes=tasks.get_per_task_classes(args.dataset, args.task, step))  

                    # add bias to the left branch of model_old
                    if self.learned_step > 1:
                        for name, mm in self.old_model.named_modules():
                            if hasattr(mm, 'convs'):
                                mm.convs.conv2.bias = nn.Parameter(torch.zeros(mm.convs.conv2.weight.shape[0]).to(mm.convs.conv2.weight.device))

                            if hasattr(mm, 'map_convs'):
                                for kk in range(4):
                                    mm.map_convs[kk].bias = nn.Parameter(torch.zeros(mm.map_convs[kk].weight.shape[0]).to(mm.map_convs[kk].weight.device))
                
                self.old_model.load_state_dict(torch.load(ckpt_path), strict=True) 

        if self.old_model is not None:
            model_old = copy.deepcopy(self.old_model)
                
            [model, model_old], optimizer = amp.initialize([model.to(self.device), model_old.to(self.device)], optimizer, opt_level=args.opt_level)
            model_old = DistributedDataParallel(model_old)

            for par in model_old.parameters():
                par.requires_grad = False
            model_old.eval()
            
        else:
            model_old = None
            model, optimizer = amp.initialize(model.to(self.device), optimizer, opt_level=args.opt_level)  

        model = DistributedDataParallel(model,delay_allreduce=True)



        if args.name != 'RCIL':
            trainer = Trainer(
                model,
                model_old,
                device=self.device,
                rank = self.rank,
                opts=args,
                trainer_state=self.trainer_state,
                classes=tasks.get_per_task_classes(args.dataset, args.task, self.learned_step),
                step=self.learned_step
                )
        else:
            trainer = Trainer_rcil(
                model,
                model_old,
                device=self.device,
                rank = self.rank,
                opts=args,
                trainer_state=self.trainer_state,
                classes=tasks.get_per_task_classes(args.dataset, args.task, self.learned_step),
                step=self.learned_step
                )
        
 
        for cur_epoch in range(args.epochs_local):
            # =====  Train  =====
            if args.name != 'RCIL':
                trainer.before(cur_epoch=self.cur_epoch,train_loader=train_loader)
                self.cur_epoch = self.cur_epoch + 1
            else:
                trainer.before(train_loader=train_loader)
            model.train()

            if args.name == 'RCIL':
                if self.learned_step > 0:
                    for name, mm in model.named_modules():
                        if hasattr(mm, 'convs'):
                            for params in mm.convs.conv2.parameters(): params.requires_grad = False 
                            for params in mm.convs.bn2.parameters(): params.requires_grad = False
                            mm.convs.bn2.eval()
                        if hasattr(mm, 'map_convs'):
                            for params in mm.map_convs.parameters(): params.requires_grad = False
                            for params in mm.map_bn.parameters(): params.requires_grad = False
                            mm.map_bn.eval()

            epoch_loss = trainer.train(
                cur_epoch=cur_epoch,
                optim=optimizer,
                train_loader=train_loader,
                scheduler=scheduler,
            )
            
            if self.rank==0:
                print(
                    f"Clinet index {self.client_index}, End of Epoch {cur_epoch+1}/{args.epochs_local}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}"
                )


        self.trainer_state = trainer.state_dict()
        self.last_learning_rate = optimizer.param_groups[0]['lr']


        model = model.to('cpu')

        torch.cuda.empty_cache() 


        torch.distributed.barrier()

        if args.use_entropy_detection==False:
            self.signal = False   # rule

        local_model = model.module.state_dict() 

        del model
        del params
        del optimizer
        del scheduler
        del train_loader
        del trainer

        if model_old is not None:
            model_old = model_old.to('cpu')
            torch.cuda.empty_cache() 
            del model_old

        return local_model

        

    def entropy_signal(self, tmp_model, loader):
        start_ent = True
        res = False

        for (images, labels) in loader:
            #time1 = time.time()
            images = images.to(self.device, dtype=torch.float32) # (bs,channel,height,width)
            labels = labels.to(self.device, dtype=torch.long) # (bs,height,width)
          
            with torch.no_grad():
                ret_intermediate = self.loss_de or (self.pod is not None) 
                outputs, _ = tmp_model(images, ret_intermediate=ret_intermediate) #(batch,classes,height,weight)

              
            softmax_out = nn.Softmax(dim=1)(outputs) #(batch,classes,height,weight)
            ent = entropy(softmax_out) 

            if start_ent: 
                all_ent = ent.float().cpu()
                all_label = labels.long().cpu()
                start_ent = False
            else: 
                all_ent = torch.cat((all_ent, ent.float().cpu()), 0)  # (b+,h,w)
                all_label = torch.cat((all_label, labels.long().cpu()), 0) # (b+,h,w)
            #print("time: ", time.time()-time1)

        overall_avg = torch.mean(all_ent.view(-1,1).squeeze(-1)).item()

        if overall_avg - self.last_entropy > self.entropy_threshold:  
            res = True
        
        self.last_entropy = overall_avg


        return res



