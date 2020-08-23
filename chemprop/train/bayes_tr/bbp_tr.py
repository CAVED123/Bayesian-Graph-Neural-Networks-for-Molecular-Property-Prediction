
import torch
from torch import nn
import copy
import numpy as np
import os
import sys
import wandb

from chemprop.models import MoleculeModelBBP
from chemprop.bayes import BayesLinear
from chemprop.bayes_utils import scheduler_const
from chemprop.utils import save_checkpoint, load_checkpoint
from chemprop.data import MoleculeDataLoader

from ..train import train
from ..evaluate import evaluate



def train_bbp(
    model,
    train_data,
    val_data,
    num_workers,
    cache,
    loss_func,
    metric_func,
    scaler,
    features_scaler,
    args,
    save_dir
    ):

    # data loaders for bbp
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size_bbp,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size_bbp,
        num_workers=num_workers,
        cache=cache
    )

    #################################################################################
    # PHASE 1 = INSTANTIATE BBP MODEL AND TRAIN MEANS + NOISE WITH PRIOR SIGMA
    #################################################################################
    
    # instantiate BBP model with Bayesian linear layers (includes log noise)
    model_bbp = MoleculeModelBBP(args)
    
    # copy over parameters from pretrained to BBP model
    # we take the transpose because the Bayes linear layers have transpose shapes
    for (_, param_bbp), (_, param_pre) in zip(model_bbp.named_parameters(), model.named_parameters()):
        param_bbp.data = copy.deepcopy(param_pre.data.T)
    
    # optimiser
    optimizer = torch.optim.Adam(model_bbp.parameters(), lr=args.lr1_bbp)
    
    # scheduler
    scheduler = scheduler_const([args.lr1_bbp])
    
    print("----------BBP training PHASE 1----------")
    
    # training loop
    n_iter = 0
    for epoch in range(args.epochs1_bbp):
        continue
        print(f'BBP phase 1 epoch {epoch}')
    
        n_iter = train(
                model=model_bbp,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=1
            )
        
        val_scores = evaluate(
                model=model_bbp,
                data_loader=val_data_loader,
                args=args,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler
            )
        
        # Average validation score
        avg_val_score = np.nanmean(val_scores)
        print(f'Validation {args.metric} = {avg_val_score:.6f}')
        wandb.log({"Validation MAE": avg_val_score})
        
    #print('saving phase 1 model_bbp')
    #save_checkpoint(os.path.join(save_dir, 'model_phase1.pt'), model_bbp, scaler, features_scaler, args)        
    
    print('loading phase 1 model_bbp')
    template = MoleculeModelBBP(args)
    model_bbp = load_checkpoint(os.path.join(save_dir, 'model_phase1.pt'), device=args.device, logger=None, template = template)    


        
    #################################################################################
    # PHASE 2 = INSTANTIATE RHOS, FREEZE LOG NOISE, TRAIN MEANS AND BANDWIDTHS
    #################################################################################
    
    # instantiate rho for each weight
    for layer in model_bbp.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
    for layer in model_bbp.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
            
    # freeze log noise
    for name, parameter in model_bbp.named_parameters():
        if name == 'log_noise':
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # optimiser
    optimizer = torch.optim.Adam(model_bbp.parameters(), lr=args.lr2_bbp)
    
    # scheduler
    scheduler = scheduler_const([args.lr2_bbp])
    
    print("----------BBP training PHASE 2----------")
    
    # training loop
    n_iter = 0
    for epoch in range(args.epochs2_bbp):
        print(f'BBP phase 2 epoch {epoch}')
    
        n_iter = train(
                model=model_bbp,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=2
            )
        
        val_scores = evaluate(
                model=model_bbp,
                data_loader=val_data_loader,
                args=args,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler
            )
        
        # Average validation score
        avg_val_score = np.nanmean(val_scores)
        print(f'Validation {args.metric} = {avg_val_score:.6f}')
        wandb.log({"Validation MAE": avg_val_score})
        
    print('saving phase 2 model_bbp')
    save_checkpoint(os.path.join(save_dir, 'model_phase2.pt'), model_bbp, scaler, features_scaler, args)        
    
    return model_bbp 











