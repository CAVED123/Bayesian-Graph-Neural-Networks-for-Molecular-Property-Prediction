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
import copy

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel, MoleculeModelBBP, MoleculeModelDUN
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits
from chemprop.bayes_utils import neg_log_like, scheduler_const

from .bayes_tr.swag_tr import train_swag
from .bayes_tr.sgld_tr import train_sgld
from .bayes_tr.gp_tr import train_gp
from .bayes_tr.bbp_tr import train_bbp
from .bayes_tr.dun_tr import train_dun
from chemprop.bayes import predict_std_gp, predict_MCdepth, GPLayer, DKLMoleculeModel, initial_inducing_points, BayesLinear, SWAG



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
        cache=cache
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
        if (args.method == 'map') or (args.method == 'swag') or (args.method == 'sgld'):
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model.pt', device=args.device, logger=logger)

        if args.method == 'gp':
            args.num_inducing_points = 1200
            fake_model = MoleculeModel(args)
            fake_model.featurizer = True
            feature_extractor = fake_model
            inducing_points = initial_inducing_points(
                train_data_loader,
                feature_extractor,
                args
                )
            gp_layer = GPLayer(inducing_points, args.num_tasks)
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/DKN_model.pt', device=args.device, logger=None,
                            template = DKLMoleculeModel(MoleculeModel(args, featurizer=True), gp_layer))

        if args.method == 'dropR' or args.method == 'dropA':
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model.pt', device=args.device, logger=logger)


        if args.method == 'bbp':
            template = MoleculeModelBBP(args)
            for layer in template.children():
                if isinstance(layer, BayesLinear):
                    layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
            for layer in template.encoder.encoder.children():
                if isinstance(layer, BayesLinear):
                    layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model_bbp.pt', device=args.device, logger=None, template = template)

        if args.method == 'dun':
            args.prior_sig_dun = 0.05
            args.depth_min = 1
            args.depth_max = 5
            args.rho_min_dun = -5.5
            args.rho_max_dun = -5
            args.log_cat_init = 0
            template = MoleculeModelDUN(args)
            for layer in template.children():
                if isinstance(layer, BayesLinear):
                    layer.init_rho(args.rho_min_dun, args.rho_max_dun)
            for layer in template.encoder.encoder.children():
                if isinstance(layer, BayesLinear):
                    layer.init_rho(args.rho_min_dun, args.rho_max_dun)
            template.create_log_cat(args)
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model_dun.pt', device=args.device, logger=None, template = template)




        # make results_dir
        results_dir = os.path.join(args.results_dir, f'model_{model_idx}')
        makedirs(results_dir)

        # train_preds, train_targets
        train_preds = predict(
            model=model,
            data_loader=train_data_loader,
            args=args,
            scaler=scaler,
            test_data=False,
            bbp_sample=False)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)

        # compute noise

        props = [0.035, 0.03, 0.02, 0.025, 0.025, 0.05, 0.02, 0.035, 0.04, 0.04, 0.04, 0.04]
        noise = np.ones(12)
        for task in range(12):
            abs_errors = np.abs(train_preds[:,task] - train_targets[:,task])
            prop = props[task]
            trimmed = np.sort(abs_errors)[:round((1-prop)*len(abs_errors))]
            noise[task] = np.sqrt(np.mean((trimmed**2)))

        # BROOKS METHOD: robust stdev estimate
        #noise_list = np.ones((30,12))
        #for estimate in range(30):

            # sample 100 points from test data
        #    idx = np.random.choice(len(train_preds), size=100, replace=False)

            # compute stdev and add to list
        #    abs_errors = np.abs(train_preds[idx] - train_targets[idx])
        #    noise_list[estimate] = np.sqrt(np.mean((abs_errors**2),0))


        # compute median
        #noise_list = np.sort(noise_list, 0)
        #noise = noise_list[14]

        ##################################
        ########## Inner loop over samples
        ##################################

        for sample_idx in range(args.samples):

            # save nosie down
            np.savez(os.path.join(results_dir, f'noise_{sample_idx}'), noise)

        print('done one')
            
            
            


