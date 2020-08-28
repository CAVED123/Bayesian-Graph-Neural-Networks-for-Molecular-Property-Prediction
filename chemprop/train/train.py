import logging
from typing import Callable
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import wandb

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from torch.optim.lr_scheduler import OneCycleLR

def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          gp_switch: bool = False,
          likelihood = None,
          bbp_switch = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    
    

        
    debug = logger.debug if logger is not None else print
    
    model.train()
    if likelihood is not None:
        likelihood.train()
    
    loss_sum = 0
    if bbp_switch is not None:
        data_loss_sum = 0
        kl_loss_sum = 0

    #for batch in tqdm(data_loader, total=len(data_loader)):
    for batch in data_loader:
        # Prepare batch
        batch: MoleculeDataset

        # .batch_graph() returns BatchMolGraph
        # .features() returns None if no additional features
        # .targets() returns list of lists of floats containing the targets
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
        
        # mask is 1 where targets are not None
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        # where targets are None, replace with 0
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # Move tensors to correct device
        mask = mask.to(args.device)
        targets = targets.to(args.device)
        class_weights = torch.ones(targets.shape, device=args.device)
        
        # zero gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        
        ##### FORWARD PASS AND LOSS COMPUTATION #####
        
        
        if bbp_switch == None:
        
            # forward pass
            preds = model(mol_batch, features_batch)
    
            # compute loss
            if gp_switch:
                loss = -loss_func(preds, targets)
            else:
                loss = loss_func(preds, targets, torch.exp(model.log_noise))
        
        
        ### bbp non sample option
        if bbp_switch == 1:    
            preds, kl_loss = model(mol_batch, features_batch, sample = False)
            data_loss = loss_func(preds, targets, torch.exp(model.log_noise))
            kl_loss /= args.train_data_size
            loss = data_loss + kl_loss  
            
        ### bbp sample option
        if bbp_switch == 2:

            if args.samples_bbp == 1:
                preds, kl_loss = model(mol_batch, features_batch, sample=True)
                data_loss = loss_func(preds, targets, torch.exp(model.log_noise))
                kl_loss /= args.train_data_size
        
            elif args.samples_bbp > 1:
                data_loss_cum = 0
                kl_loss_cum = 0
        
                for i in range(args.samples_bbp):
                    preds, kl_loss_i = model(mol_batch, features_batch, sample=True)
                    data_loss_i = loss_func(preds, targets, torch.exp(model.log_noise))                    
                    kl_loss_i /= args.train_data_size                    
                    
                    data_loss_cum += data_loss_i
                    kl_loss_cum += kl_loss_i
        
                data_loss = data_loss_cum / args.samples_bbp
                kl_loss = kl_loss_cum / args.samples_bbp
            
            loss = data_loss + kl_loss

        ### DUN non sample option
        if bbp_switch == 3:    
            cat = model.categorical / (model.categorical.sum())
            _, preds_list, kl_loss = model(mol_batch, features_batch, sample=False)
            data_loss = loss_func(preds_list, targets, torch.exp(model.log_noise), cat)
            kl_loss /= args.train_data_size
            loss = data_loss + kl_loss  

        ### DUN sample option
        if bbp_switch == 4:

            cat = model.categorical / (model.categorical.sum())

            if args.samples_dun == 1:
                _, preds_list, kl_loss = model(mol_batch, features_batch, sample=True)
                data_loss = loss_func(preds_list, targets, torch.exp(model.log_noise), cat)
                kl_loss /= args.train_data_size
        
            elif args.samples_dun > 1:
                data_loss_cum = 0
                kl_loss_cum = 0
        
                for i in range(args.samples_dun):
                    _, preds_list, kl_loss_i = model(mol_batch, features_batch, sample=True)
                    data_loss_i = loss_func(preds_list, targets, torch.exp(model.log_noise), cat)
                    kl_loss_i /= args.train_data_size                    
                    
                    data_loss_cum += data_loss_i
                    kl_loss_cum += kl_loss_i
        
                data_loss = data_loss_cum / args.samples_dun
                kl_loss = kl_loss_cum / args.samples_dun
            
            loss = data_loss + kl_loss

            print('-----')
            print(data_loss)
            print(kl_loss)
            print(cat)
            
        #############################################
        
        
        # backward pass; update weights
        loss.backward()
        optimizer.step()
        
        
        #for name, parameter in model.named_parameters():
            #print(name)#, parameter.grad)
            #print(np.sum(np.array(parameter.grad)))

        # add to loss_sum and iter_count
        loss_sum += loss.item() * len(batch)
        if bbp_switch is not None:
            data_loss_sum += data_loss.item() * len(batch)
            kl_loss_sum += kl_loss.item() * len(batch)

        # update learning rate by taking a step
        if isinstance(scheduler, NoamLR) or isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # increment n_iter (total number of examples across epochs)
        n_iter += len(batch)

        
        ########### REPORTING
        
        # determine reporting frequency
        if gp_switch:
            batch_size = args.batch_size_gp
        else:
            batch_size = args.batch_size
        
        # determine log freq
        if gp_switch:
            log_frequency = args.log_frequency_gp
        else:
            log_frequency = args.log_frequency

        # Log and/or add to tensorboard
        #if (n_iter // batch_size) % log_frequency == 0:
            
        # per epoch reporting
        if n_iter % args.train_data_size == 0:
            lrs = scheduler.get_last_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            
            loss_avg = loss_sum / args.train_data_size

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')
            
            if bbp_switch is not None:
                data_loss_avg = data_loss_sum / args.train_data_size
                kl_loss_avg = kl_loss_sum / args.train_data_size
                wandb.log({"Total loss": loss_avg}, commit=False)
                wandb.log({"Likelihood cost": data_loss_avg}, commit=False)
                wandb.log({"KL cost": kl_loss_avg}, commit=False)
            else:
                wandb.log({"Negative log likelihood (scaled)": loss_avg}, commit=False)
            
            
            wandb.log({"Learning rate": lrs[0]}, commit=False)
            


    return n_iter















