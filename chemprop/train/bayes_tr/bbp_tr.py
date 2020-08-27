
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
    
    # instantiate BBP model with Bayesian linear layers (includes log noise)
    model_bbp = MoleculeModelBBP(args)

    # copy over parameters from pretrained to BBP model
    # we take the transpose because the Bayes linear layers have transpose shapes
    for (_, param_bbp), (_, param_pre) in zip(model_bbp.named_parameters(), model.named_parameters()):
        param_bbp.data = copy.deepcopy(param_pre.data.T)
        
    # instantiate rho for each weight
    for layer in model_bbp.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
    for layer in model_bbp.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)

    # move bbp model to cuda
    if args.cuda:
        print('Moving bbp model to cuda')
        model_bbp = model_bbp.to(args.device)
    
    # optimiser
    optimizer = torch.optim.Adam(model_bbp.parameters(), lr=args.lr_bbp)
    
    # scheduler
    scheduler = scheduler_const([args.lr_bbp])
    
    print("----------BBP training----------")
    
    # training loop
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch, n_iter = 0, 0
    for epoch in range(args.epochs_bbp):
        print(f'BBP epoch {epoch}')
    
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

        # Save model checkpoint if improved validation score
        if (args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score) and (epoch >= args.presave_bbp):
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model_bbp.pt'), model_bbp, scaler, features_scaler, args)
    
    # load model with best validation score
    template = MoleculeModelBBP(args)
    for layer in template.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
    for layer in template.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
    print(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    model_bbp = load_checkpoint(os.path.join(save_dir, 'model_bbp.pt'), device=args.device, logger=None, template = template)


    return model_bbp

