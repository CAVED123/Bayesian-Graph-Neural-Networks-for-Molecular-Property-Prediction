import numpy as np
import torch
import os
import wandb

from ..train import train
from ..evaluate import evaluate

from chemprop.utils import save_checkpoint
from chemprop.data import MoleculeDataLoader

from chemprop.bayes import SGLD
from chemprop.bayes_utils import scheduler_const

from torch.optim.lr_scheduler import OneCycleLR



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

    # number of sgld epochs
    epochs_sgld = args.mix_epochs * args.samples

    print("----------SGLD training----------")
    
    # training loop
    n_iter = 0
    sample_idx = 0
    for epoch in range(epochs_sgld):

        ##### DEFINE OPTIMISER AND SCHEDULER ########################

        if epoch % args.mix_epochs == 0:
            print('\n********** resetting scheduler **********')

            optimizer = SGLD([
                {'params': model.encoder.parameters()},
                {'params': model.ffn.parameters()},
                {'params': model.log_noise, 'lr': args.lr_max_sgld/5/25, 'addnoise': False}
                ], args, lr=args.lr_max_sgld/25, weight_decay=args.weight_decay_sgld, addnoise=True)

            num_param_groups = len(optimizer.param_groups)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr = [args.lr_max_sgld, args.lr_max_sgld, args.lr_max_sgld/5], 
                epochs=args.mix_epochs, 
                steps_per_epoch=-(-args.train_data_size // args.batch_size_sgld), 
                pct_start=0.2,
                anneal_strategy='cos', 
                cycle_momentum=False, 
                div_factor=25.0,
                final_div_factor=10000)

        #############################################################
    
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
        if (epoch + 1) % args.mix_epochs == 0:
            print(f'---------- collecting sgld sample {sample_idx} ----------\n')
            save_checkpoint(os.path.join(save_dir, f'model_{sample_idx}.pt'), model, scaler, features_scaler, args)
            sample_idx += 1
        
    return model
















