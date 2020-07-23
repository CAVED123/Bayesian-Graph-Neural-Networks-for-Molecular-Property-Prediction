import numpy as np
import os

from .train import train
from .evaluate import evaluate

from chemprop.utils import save_checkpoint

from chemprop.data import MoleculeDataLoader

from chemprop.bayes import loss_sgld
from chemprop.bayes import SGLD
from chemprop.bayes_utils import scheduler_const



def train_sgld(
        model,
        train_data,
        val_data,
        num_workers,
        cache,
        metric_func,
        scaler,
        features_scaler,
        args,
        save_dir_sgld):
    
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
    
    # loss function 
    loss_func = loss_sgld
    
    # instantiate SGLD optimiser
    params = [{'params': model.encoder.parameters()},
              {'params': model.ffn.parameters()},
              {'params': model.log_noise, 'addnoise': False}]
    optimizer = SGLD(params, args, lr=args.lr_sgld, weight_decay=args.weight_decay_sgld, addnoise=True)
    
    # instantiate scheduler
    scheduler = scheduler_const(args.lr_sgld)
    
    # number of sgld epochs
    epochs_sgld = args.burnin_epochs + args.mix_epochs * args.samples

    
    
    print("----------SGLD training----------")
    
    # training loop
    n_iter = 0
    sample_idx = 0
    for epoch in range(epochs_sgld):
        print(f'SGLD spoch {epoch}')
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                sgld_switch=True
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
        
        # collect model samples
        if (epoch + 1 - args.burnin_epochs) % args.mix_epochs == 0 and (epoch + 1) > args.burnin_epochs:
            print(f'Collecting sample {sample_idx}')
            save_checkpoint(os.path.join(save_dir_sgld, f'model_{sample_idx}.pt'), model, scaler, features_scaler, args)
            sample_idx += 1
        
    return model
















