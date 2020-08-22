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


def train_swag(
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
        save_dir):

    # create data loaders for swag (allows different batch size)
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size_swag,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size_swag,
        num_workers=num_workers,
        cache=cache
    )

    # define no_cov_mat from cov_mat
    if args.cov_mat:
        no_cov_mat = False
    else:
        no_cov_mat = True
    
    # instantiate SWAG model (wrapper)
    swag_model = SWAG(
        model,
        args,
        no_cov_mat,
        args.max_num_models,
        var_clamp=1e-30
    )

    ############## DEFINE COSINE OPTIMISER AND SCHEDULER ##############
    
    # define optimiser
    optimizer = torch.optim.SGD([
        {'params': model.encoder.parameters()},
        {'params': model.ffn.parameters()},
        {'params': model.log_noise, 'lr': args.lr_swag/5/25, 'weight_decay': 0}
        ], lr=args.lr_swag/25, weight_decay=args.weight_decay_swag, momentum=args.momentum_swag)

    # define scheduler
    num_param_groups = len(optimizer.param_groups)
    scheduler = OneCycleLR(
        optimizer,
        max_lr = [args.lr_swag, args.lr_swag, args.lr_swag/5],
        epochs=args.epochs_swag, 
        steps_per_epoch=-(-args.train_data_size // args.batch_size_sgld), 
        pct_start=5/args.epochs_swag,
        anneal_strategy='cos', 
        cycle_momentum=False, 
        div_factor=25.0,
        final_div_factor=1/25)

    ###################################################################

    print("----------SWAG training----------")
    
    # training loop
    n_iter = 0
    for epoch in range(args.epochs_swag):

        print(f'SWAG epoch {epoch}')
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter
            )
        
        val_scores = evaluate(
                model=model,
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

        # SWAG update
        if (epoch >= args.burnin_swag) and (avg_val_score < args.val_threshold):
            swag_model.collect_model(model)
            print('***collection***')

    # save final swag model
    save_checkpoint(os.path.join(save_dir, 'model.pt'), swag_model, scaler, features_scaler, args)

    return swag_model

    
