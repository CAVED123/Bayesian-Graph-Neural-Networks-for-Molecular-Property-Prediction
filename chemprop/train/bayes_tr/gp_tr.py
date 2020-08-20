import os
import torch
import gpytorch
import numpy as np
import copy

from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataLoader
from chemprop.bayes import GPLayer, DKLMoleculeModel, initial_inducing_points
from chemprop.utils import save_checkpoint, load_checkpoint
from chemprop.nn_utils import NoamLR

from ..train import train
from ..evaluate import evaluate



def train_gp(
        model,
        train_data,
        val_data,
        num_workers,
        cache,
        metric_func,
        scaler,
        features_scaler,
        args,
        save_dir_gp,
        logger=None):
    
    
    # create data loaders for gp (allows different batch size)
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size_gp,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size_gp,
        num_workers=num_workers,
        cache=cache
    )
    
    
    
    
    
    # feature_extractor
    model.featurizer = True
    feature_extractor = model
    
    # inducing points
    inducing_points = initial_inducing_points(
        train_data_loader,
        feature_extractor,
        args
        )
    
    # GP layer
    gp_layer = GPLayer(inducing_points, args.num_tasks)
    
    # full DKL model
    model = copy.deepcopy(DKLMoleculeModel(feature_extractor, gp_layer))
    
    # likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args.num_tasks)
    
    # loss object
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=args.train_data_size)
    
    # optimizer
    params_list = [
        {'params': model.feature_extractor.parameters()},
        {'params': model.gp_layer.hyperparameters()},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ]    
    optimizer = torch.optim.Adam(params_list, lr = args.init_lr_gp)    
    
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs_gp]*len(params_list),
        total_epochs=[args.epochs_gp]*len(params_list),
        steps_per_epoch=args.train_data_size // args.batch_size_gp,
        init_lr=[args.init_lr_gp]*len(params_list),
        max_lr=[args.max_lr_gp]*len(params_list),
        final_lr=[args.final_lr_gp]*len(params_list))
    
    # freeze weights
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False

    
    
    print("----------GP training----------")
    
    
    
    # training loop
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch, n_iter = 0, 0
    for epoch in range(args.epochs_gp):
        
        # unfreeze weights
        if epoch == args.unfreeze_epoch_gp:
            print('----------Unfreezing featurizer weights----------')
            for parameter in model.feature_extractor.parameters():
                parameter.requires_grad = True
                
        print(f'GP epoch {epoch}')
        
    
        n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=mll,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                gp_switch=True,
                likelihood = likelihood
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
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir_gp, 'DKN_model.pt'), model, scaler, features_scaler, args)
            
            
    # load model with best validation score
    # NOTE: TEMPLATE MUST BE NEWLY INSTANTIATED MODEL
    print(f'Loading model with best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    model = load_checkpoint(os.path.join(save_dir_gp, 'DKN_model.pt'), device=args.device, logger=logger,
                            template = DKLMoleculeModel(MoleculeModel(args, featurizer=True), gp_layer))

    
    
    return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        







