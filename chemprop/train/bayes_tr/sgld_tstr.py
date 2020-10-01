import numpy as np
import torch
import os
import wandb

from ..train import train
from ..evaluate import evaluate

from chemprop.utils import save_checkpoint, makedirs
from chemprop.data import MoleculeDataLoader

from chemprop.bayes import SGLD
from chemprop.bayes_utils import scheduler_const

from torch.optim.lr_scheduler import OneCycleLR



def train_sgld_pdts(
        model,
        train_data_loader,
        loss_func,
        scaler,
        features_scaler,
        args,
        save_dir,
        batch_no):

    save_dir_sgld = os.path.join(save_dir, f'model_{batch_no}')
    makedirs(save_dir_sgld)

    # number of sgld epochs
    epochs_sgld = int(args.mix_epochs * args.samples)

    # freeze log noise
    for name, parameter in model.named_parameters():
        if name == 'log_noise':
            parameter.requires_grad = False

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
                steps_per_epoch=-(-args.train_data_size // args.batch_size), 
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

        # collect model samples
        if (epoch + 1) % args.mix_epochs == 0:
            print(f'---------- collecting sgld sample {sample_idx} ----------\n')
            save_checkpoint(os.path.join(save_dir_sgld, f'model_{sample_idx}.pt'), model, scaler, features_scaler, args)
            sample_idx += 1
        
    return model
















