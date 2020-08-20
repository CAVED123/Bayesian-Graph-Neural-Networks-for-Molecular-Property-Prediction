
import torch
from torch import nn
import copy
import numpy as np
import os

from chemprop.models import MoleculeModelBBP
from chemprop.bayes import data_loss_bbp, BayesLinear
from chemprop.bayes_utils import scheduler_const
from chemprop.utils import save_checkpoint, load_checkpoint

from ..train import train
from ..evaluate import evaluate




def train_bbp(
        model_pretrain,
        train_data_loader,
        val_data_loader,
        metric_func,
        scaler,
        features_scaler,
        args,
        save_dir_bbp,
        logger=None
        ):

    
    # instantiate BBP model with Bayesian linear layers
    model = MoleculeModelBBP(args)
    
    
    # copy over parameters from pretraining to BBP model
    # we take the transpose because the Bayes linear layers have transpose shapes
    for (_, param_bbp), (_, param_pre) in zip(model.named_parameters(), model_pretrain.named_parameters()):
        param_bbp.data = copy.deepcopy(param_pre.data.T)    
    
    
    
    
    ##########################################################################
    # PHASE 1 = INSTANTIATE LOG NOISE AND TRAIN LOG NOISE
    ##########################################################################
    
    
    model.create_log_noise(args)

    for name, parameter in model.named_parameters():
        if name == 'log_noise':
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False    

    
    # loss, opt and scheduler
    loss_func = data_loss_bbp
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_phase1_bbp)
    scheduler = scheduler_const(args.lr_phase1_bbp)
    
    
    print("----------BBP training PHASE 1: learning log noise----------")
    
    
    # training loop
    n_iter = 0
    
    for epoch in range(args.epochs_phase1_bbp):
        break
        print(f'BBP epoch {epoch}')
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=1
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
        
    #print('saving log noise trained model')
    #save_checkpoint(os.path.join(save_dir_bbp, 'model_phase1.pt'), model, scaler, features_scaler, args)        
    
    print('loading log noise trained model')
    template = MoleculeModelBBP(args)
    template.create_log_noise(args)
    model = load_checkpoint(os.path.join(save_dir_bbp, 'model_phase1.pt'), device=args.device, logger=logger, template = template)
    
    
    
    
    ##########################################################################
    # PHASE 2 = TRAIN WITH PRIOR SIG
    ##########################################################################
    # purpose of this section is to train the means with prior sig regularisaion
    
    
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
  
    
    
    # loss, opt and scheduler
    loss_func = data_loss_bbp
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_phase2_bbp)
    scheduler = scheduler_const(args.lr_phase2_bbp)
    
    
    print("----------BBP training PHASE 2: learning means with regularisation----------")
    
    
    # training loop
    best_score = float('inf') if args.minimize_score else -float('inf')
    n_iter = 0
    for epoch in range(args.epochs_phase2_bbp):
        break
        print(f'BBP epoch {epoch}')
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=1
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
        
        # Save model checkpoint if improved validation score
        if args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score:
            best_score = avg_val_score
            save_checkpoint(os.path.join(save_dir_bbp, 'model_phase2.pt'), model, scaler, features_scaler, args)    
            
    
    print('loading BEST reg trained model')
    template = MoleculeModelBBP(args)
    template.create_log_noise(args)
    model = load_checkpoint(os.path.join(save_dir_bbp, 'model_phase2.pt'), device=args.device, logger=logger, template = template)
    


        
    ##########################################################################
    # PHASE 3 = TRAIN BANDWIDTHS
    ##########################################################################
    # purpose of this section is to train the rho for each weight
    
    
    # instantiate rho for each weight
    for layer in model.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
    for layer in model.encoder.encoder.children():
        if isinstance(layer, BayesLinear):
            layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
            
            
    # turn grad on for rho params only        
    for name, parameter in model.named_parameters():
        if 'W_p' in name or 'b_p' in name:
        #if 'encoder.encoder.W_o.W_p' in name:
            parameter.requires_grad = True
            #print(name)
            #print(np.all(np.isfinite(parameter.detach().numpy())))
        else:
            parameter.requires_grad = True  
    

    # loss, opt and scheduler
    loss_func = data_loss_bbp
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_phase3_bbp)
    scheduler = scheduler_const(args.lr_phase3_bbp)
    
    
    print("----------BBP training PHASE 3: learning a bandwidth for each weight----------")
    
    
    # training loop
    best_score = float('inf') if args.minimize_score else -float('inf')
    n_iter = 0
    for epoch in range(args.epochs_phase3_bbp):
        print(f'BBP epoch {epoch}')
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch=2
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
        
        # Save model checkpoint if improved validation score
        if args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score:
            best_score = avg_val_score
            save_checkpoint(os.path.join(save_dir_bbp, 'model_phase3.pt'), model, scaler, features_scaler, args)    
            
    
    print('loading reg trained model with best val acc')
    template = MoleculeModelBBP(args)
    template.create_log_noise(args)
    model = load_checkpoint(os.path.join(save_dir_bbp, 'model_phase3.pt'), device=args.device, logger=logger, template = template)











