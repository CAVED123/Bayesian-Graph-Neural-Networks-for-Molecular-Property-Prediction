import csv
from logging import Logger
import os
import sys
from typing import List

import numpy as np
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, SGD
import wandb

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits
from chemprop.bayes_utils import neg_log_like, scheduler_const

from .bayes_tr.swag_tr import train_swag
from .bayes_tr.sgld_tr import train_sgld
from .bayes_tr.gp_tr import train_gp
from .bayes_tr.bbp_tr import train_bbp
from .bayes_tr.dun_tr import train_dun
from chemprop.bayes import predict_std_gp, predict_MCdepth



def new_noise(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """

    debug = info = print

    # Get data
    args.task_names = args.target_columns or get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = neg_log_like
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        cache = True
        num_workers = 0
    else:
        cache = False
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )



    ###########################################
    ########## Outer loop over ensemble members
    ###########################################
    
    for model_idx in range(args.ensemble_start_idx, args.ensemble_start_idx + args.ensemble_size):
        


        # load the model
        model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model.pt', device=args.device, logger=logger)
        
        

        
        
        ##################################
        ########## Inner loop over samples
        ##################################
        
        for sample_idx in range(args.samples):
            
            
            # make predictions
            train_preds = predict(
                model=model,
                data_loader=train_data_loader,
                args=args,
                scaler=scaler,
                test_data=True,
                bbp_sample=True)

            print(train_preds.shape)

            
            #######################################################################
            #######################################################################
            #####        SAVING STUFF DOWN
            
            
            #noise = np.exp(log_noise.detach().cpu().numpy()) * np.array(scaler.stds)
            #np.savez(os.path.join(results_dir, f'preds_{sample_idx}'), np.array(test_preds))
            #np.savez(os.path.join(results_dir, f'noise_{sample_idx}'), noise)


            #######################################################################
            #######################################################################
    
 





