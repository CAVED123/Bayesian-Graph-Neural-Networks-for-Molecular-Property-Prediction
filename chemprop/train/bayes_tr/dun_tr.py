
import torch
from torch import nn
import copy
import numpy as np
import os
import sys
import wandb

from chemprop.models import MoleculeModelDUN
from chemprop.bayes import BayesLinear, neg_log_likeDUN
from chemprop.bayes_utils import scheduler_const
from chemprop.utils import save_checkpoint, load_checkpoint
from chemprop.data import MoleculeDataLoader
from chemprop.nn_utils import NoamLR

from ..train import train
from ..evaluate import evaluate



def train_dun(
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

    # data loaders for dun
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size_dun,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size_dun,
        num_workers=num_workers,
        cache=cache
    )
    
    # instantiate DUN model with Bayesian linear layers (includes log noise)
    model_dun = MoleculeModelDUN(args)

    # copy over parameters from pretrained to DUN model
    # we take the transpose because the Bayes linear layers have transpose shapes
    for (_, param_dun), (_, param_pre) in zip(model_dun.named_parameters(), model.named_parameters()):
        param_dun.data = copy.deepcopy(param_pre.data.T)
        
    # instantiate rho for each weight
    for layer in model_dun.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_dun, args.rho_max_dun)
    for layer in model_dun.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_dun, args.rho_max_dun)

    # instantiate variational categorical distribution
    model_dun.create_categorical(args)

    # move dun model to cuda
    if args.cuda:
        print('Moving dun model to cuda')
        model_dun = model_dun.to(args.device)
    
    # loss_func
    loss_func = neg_log_likeDUN
    
    # optimiser
    optimizer = torch.optim.Adam(model_dun.parameters(), lr=args.lr_dun_min)
    
    # scheduler
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[2],
        total_epochs=[100],
        steps_per_epoch=args.train_data_size // args.batch_size_dun,
        init_lr=[args.lr_dun_min],
        max_lr=[args.lr_dun_max],
        final_lr=[args.lr_dun_min])
    
    print("----------DUN training----------")
    
    # training loop
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch, n_iter = 0, 0
    bbp_switch = 3
    for epoch in range(args.epochs_dun):
        print(f'DUN epoch {epoch}')

        # switch scheduler after 100 epochs
        if epoch == 100:
            scheduler = scheduler_const([args.lr_dun_min])

        # change bbp_switch after 150 epochs
        if epoch == 150:
            bbp_switch = 4
    
        n_iter = train(
                model=model_dun,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=bbp_switch
            )
        
        val_scores = evaluate(
                model=model_dun,
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
                not args.minimize_score and avg_val_score > best_score) and (epoch >= args.presave_dun):
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model_dun.pt'), model_dun, scaler, features_scaler, args)
    
    # load model with best validation score
    template = MoleculeModelDUN(args)
    for layer in template.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_dun, args.rho_max_dun)
    for layer in template.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_dun, args.rho_max_dun)
    template.create_categorical(args)
    print(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    model_dun = load_checkpoint(os.path.join(save_dir, 'model_dun.pt'), device=args.device, logger=None, template = template)


    return model_dun

