import os
import torch
import numpy as np
import wandb

from ..train import train
from ..evaluate import evaluate
from chemprop.data import MoleculeDataLoader
from chemprop.utils import save_checkpoint
from chemprop.bayes.swag import SWAG
from chemprop.bayes_utils import scheduler_const
from torch.optim.lr_scheduler import OneCycleLR


def train_swag_pdts(
        model_core,
        train_data_loader,
        loss_func,
        scaler,
        features_scaler,
        args,
        save_dir,
        batch_no):

    # define no_cov_mat from cov_mat
    if args.cov_mat:
        no_cov_mat = False
    else:
        no_cov_mat = True
    
    # instantiate SWAG model (wrapper)
    swag_model = SWAG(
        model_core,
        args,
        no_cov_mat,
        args.max_num_models,
        var_clamp=1e-30
    )

    ############## DEFINE COSINE OPTIMISER AND SCHEDULER ##############
    
    # define optimiser
    optimizer = torch.optim.SGD([
        {'params': model_core.encoder.parameters()},
        {'params': model_core.ffn.parameters()},
        {'params': model_core.log_noise, 'lr': args.lr_swag/5/25, 'weight_decay': 0}
        ], lr=args.lr_swag/25, weight_decay=args.weight_decay_swag, momentum=args.momentum_swag)

    # define scheduler
    num_param_groups = len(optimizer.param_groups)
    if batch_no == 0:
        scheduler = OneCycleLR(
            optimizer,
            max_lr = [args.lr_swag, args.lr_swag, args.lr_swag/5],
            epochs=args.epochs_swag,
            steps_per_epoch=-(-args.train_data_size // args.batch_size), 
            pct_start=5/args.epochs_swag,
            anneal_strategy='cos', 
            cycle_momentum=False, 
            div_factor=25.0,
            final_div_factor=1/25)
    else:
        scheduler = scheduler_const([args.lr_swag])

    ###################################################################

    # freeze log noise
    for name, parameter in model_core.named_parameters():
        if name == 'log_noise':
            parameter.requires_grad = False

    print("----------SWAG training----------")
    
    # training loop
    n_iter = 0
    for epoch in range(args.epochs_swag):

        print(f'SWAG epoch {epoch}')
    
        loss_avg, n_iter = train(
                model=model_core,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter
            )

        # SWAG update
        if (epoch >= args.burnin_swag) and (loss_avg < args.loss_threshold):
            swag_model.collect_model(model_core)
            print('***collection***')

    # save final swag model
    save_checkpoint(os.path.join(save_dir, f'model_{batch_no}.pt'), swag_model, scaler, features_scaler, args)

    return swag_model

    
