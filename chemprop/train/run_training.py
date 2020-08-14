import csv
from logging import Logger
import os
import sys
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .swag_tr import train_swag
from .sgld_tr import train_sgld
from .gp_tr import train_gp
from .bbp_tr import train_bbp
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits
from chemprop.bayes import data_loss_bbp
from chemprop.bayes_utils import neg_log_like



def run_training(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    args.save(os.path.join(args.save_dir, 'args.json'))

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Get data
    debug('Loading data')
    args.task_names = args.target_columns or get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')



    # Split data
    debug(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.save_smiles_splits:
        save_smiles_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            data_path=args.data_path,
            save_dir=args.save_dir
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

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
    
    for model_idx in range(args.ensemble_size):
        
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)
        

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = Adam([
            {'params': model.encoder.parameters()},
            {'params': model.ffn.parameters()},
            {'params': model.log_noise, 'weight_decay': 0}
            ], lr=args.init_lr, weight_decay=args.weight_decay)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                args=args,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)        

        
        # load model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)
        
        
        # SWAG training loop, returns swag_model
        if args.swag:
            model = train_swag(
                model,
                train_data_loader,
                val_data_loader,
                loss_func,
                metric_func,
                args,
                scaler)
        
        # SGLD loop, which saves nets
        if args.sgld:
            save_dir_sgld = os.path.join(save_dir, 'SGLD_models')
            makedirs(save_dir_sgld)
            model = train_sgld(
                model,
                train_data,
                val_data,
                num_workers,
                cache,
                metric_func,
                scaler,
                features_scaler,
                args,
                save_dir_sgld)
        
        # GP loop
        if args.gp:
            save_dir_gp = os.path.join(save_dir, 'GP_model')
            makedirs(save_dir_gp)
            model = train_gp(
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
                logger)
        
        # BBP
        if args.bbp:
            save_dir_bbp = os.path.join(save_dir, 'bbp_models')
            makedirs(save_dir_bbp)
            model = train_bbp(
                model,
                train_data_loader,
                val_data_loader,
                metric_func,
                scaler,
                features_scaler,
                args,
                save_dir_bbp)



        
        
        ##################################
        ########## Inner loop over samples
        ##################################
        
        for sample_idx in range(args.samples):
            
            # draw model from SWAG posterior
            if args.swag:
                model.sample(scale=1.0, cov=args.cov_mat, block=args.block)
            
            # draw model from collected SGLD models
            if args.sgld:
                model = load_checkpoint(os.path.join(save_dir_sgld, f'model_{sample_idx}.pt'), device=args.device, logger=logger)
            
            
            # make predictions
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                args=args,
                scaler=scaler,
                test_data=True,
                bbp_sample=True
            )
            
            # evaluate predictions using metric function
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )   
            
            
            
            # add predictions to sum_test_preds
            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)
    
            # compute average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx}, sample {sample_idx} test {args.metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)
    
            # show individual test scores
            if args.show_individual_scores:
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx}, sample {sample_idx} test {task_name} {args.metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
            writer.close()



    #################################
    ########## Bayesian Model Average
    #################################
    
    # note: this may be an average of Bayesian samples and/or components in an ensemble
            
    # compute number of prediction iterations
    pred_iterations = args.ensemble_size * args.samples
    
    # average predictions across iterations
    avg_test_preds = (sum_test_preds / pred_iterations).tolist()

    # evaluate
    BMA_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # average scores across tasks
    avg_BMA_test_score = np.nanmean(BMA_scores)
    info(f'BMA test {args.metric} = {avg_BMA_test_score:.6f}')

    # individual scores
    if args.show_individual_scores:
        for task_name, BMA_score in zip(args.task_names, BMA_scores):
            info(f'BMA test {task_name} {args.metric} = {BMA_score:.6f}')

    return BMA_scores








