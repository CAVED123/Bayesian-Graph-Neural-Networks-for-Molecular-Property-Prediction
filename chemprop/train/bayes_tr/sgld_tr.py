import numpy as np
import torch
import os
import wandb

from ..train import train
from ..evaluate import evaluate

from chemprop.utils import save_checkpoint
from chemprop.nn_utils import NoamLR
from chemprop.data import MoleculeDataLoader

from chemprop.bayes import loss_sgld
from chemprop.bayes import SGLD
from chemprop.bayes_utils import scheduler_const

from torch.optim.lr_scheduler import CosineAnnealingLR



def train_sgld(
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
    
    # create data loaders for sgld (allows different batch size)
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size_sgld,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size_sgld,
        num_workers=num_workers,
        cache=cache
    )
    

    ##### DEFINE OPTIMISER AND SCHEDULER FOR BURNIN #####

    optimizer = torch.optim.SGD([
        {'params': model.encoder.parameters()},
        {'params': model.ffn.parameters()},
        {'params': model.log_noise, 'lr': 1e-6, 'weight_decay': 0}
        ], lr=1e-6, weight_decay=args.weight_decay_sgld)

    num_param_groups = len(optimizer.param_groups)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[5] * num_param_groups,
        total_epochs=[args.burnin_sgld] * num_param_groups,
        steps_per_epoch=args.train_data_size // args.batch_size_sgld,
        init_lr=[1e-6] * num_param_groups,
        max_lr=[args.lr_base_sgld, args.lr_base_sgld, 1e-5],
        final_lr=[args.lr_base_sgld, args.lr_base_sgld, 1e-5]
    )

    #####################################################
    

    # number of sgld epochs
    epochs_sgld = args.burnin_sgld + args.mix_epochs * args.samples

    print("----------SGLD training----------")
    
    # training loop
    n_iter = 0
    sample_idx = 0
    for epoch in range(epochs_sgld):

        ##### DEFINE OPTIMISER AND SCHEDULER FOR SAMPLING #####

        if (epoch - args.burnin_sgld) % args.mix_epochs == 0 and epoch >= args.burnin_sgld:
            print('\n********** resetting scheduler **********')

            optimizer = SGLD([
                {'params': model.encoder.parameters()},
                {'params': model.ffn.parameters()},
                {'params': model.log_noise, 'lr': 2e-5, 'addnoise': False}
                ], args, lr=args.lr_max_sgld, weight_decay=args.weight_decay_sgld, addnoise=True)

            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max = -(-args.train_data_size // args.batch_size_sgld) * (args.mix_epochs), 
                eta_min=1e-10
                )

        #######################################################
    
        print(f'SGLD epoch {epoch}')

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

        # collect model samples
        if (epoch + 1 - args.burnin_sgld) % args.mix_epochs == 0 and (epoch + 1) > args.burnin_sgld:
            print(f'---------- collecting sgld sample {sample_idx} ----------\n')
            save_checkpoint(os.path.join(save_dir, f'model_{sample_idx}.pt'), model, scaler, features_scaler, args)
            sample_idx += 1
        
    return model
















