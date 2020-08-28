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
from chemprop.bayes import predict_std_gp



def run_training(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """

    debug = info = print

    # Print command line and args
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')
    debug('Args')
    debug(args)

    # Save args
    args.save(os.path.join(args.save_dir, 'args.json'))

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
    
    for model_idx in range(args.ensemble_start_idx, args.ensemble_start_idx + args.ensemble_size):

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seeds[model_idx])
        

        ######## set up all logging ########
        # make save_dir
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # make results_dir
        results_dir = os.path.join(args.results_dir, f'model_{model_idx}')
        makedirs(results_dir)

        # initialise wandb
        os.environ['WANDB_MODE'] = 'dryrun'
        wandb.init(
            name=args.wandb_name+'_'+str(model_idx),
            project=args.wandb_proj,
            reinit=True)
        print('WANDB directory is:')
        print(wandb.run.dir)
        ####################################


        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_path}')
            model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model.pt', device=args.device, logger=logger)
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

        # Optimizer
        optimizer = Adam([
            {'params': model.encoder.parameters()},
            {'params': model.ffn.parameters()},
            {'params': model.log_noise, 'weight_decay': 0}
            ], lr=args.init_lr, weight_decay=args.weight_decay)

        # Learning rate scheduler
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
                logger=logger
            )
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
            wandb.log({"Validation MAE": avg_val_score})

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch == args.noam_epochs - 1:
                optimizer = Adam([
                    {'params': model.encoder.parameters()},
                    {'params': model.ffn.parameters()},
                    {'params': model.log_noise, 'weight_decay': 0}
                    ], lr=args.final_lr, weight_decay=args.weight_decay)

                scheduler = scheduler_const([args.final_lr])
        
        # load model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), device=args.device, logger=logger)
        
        
        # SWAG training loop, returns swag_model
        if args.swag:
            model = train_swag(
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
                save_dir)

        # SGLD loop, which saves nets
        if args.sgld:
            model = train_sgld(
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
                save_dir)
        
        # GP loop
        if args.gp:
            model, likelihood = train_gp(
                model,
                train_data,
                val_data,
                num_workers,
                cache,
                metric_func,
                scaler,
                features_scaler,
                args,
                save_dir)
        
        # BBP
        if args.bbp:
            model = train_bbp(
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
                save_dir)

        # DUN
        if args.dun:
            model = train_dun(
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
                save_dir)

        
        
        ##################################
        ########## Inner loop over samples
        ##################################
        
        for sample_idx in range(args.samples):
            
            # draw model from SWAG posterior
            if args.swag:
                model.sample(scale=1.0, cov=args.cov_mat, block=args.block)
            
            # draw model from collected SGLD models
            if args.sgld:
                model = load_checkpoint(os.path.join(save_dir, f'model_{sample_idx}.pt'), device=args.device, logger=logger)
            
            # make predictions
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                args=args,
                scaler=scaler,
                test_data=True,
                bbp_sample=True)

            
            #######################################################################
            #######################################################################
            #####        SAVING STUFF DOWN
            
            
            if args.gp:

                # get test_preds_std (scaled back to original data)
                test_preds_std = predict_std_gp(
                    model=model,
                    data_loader=test_data_loader,
                    args=args,
                    scaler=scaler,
                    likelihood = likelihood)

                # 1 - MEANS
                np.savez(os.path.join(results_dir, f'preds_{sample_idx}'), np.array(test_preds))

                # 2 - STD, combined aleatoric and epistemic (we save down the stds, always)
                np.savez(os.path.join(results_dir, f'predsSTDEV_{sample_idx}'), np.array(test_preds_std))


            else:

                # save test_preds and aleatoric uncertainties
                if args.dun:
                    log_cat = model.log_cat.detach().cpu().numpy()
                    cat = np.exp(log_cat) / np.sum(np.exp(log_cat))    
                    np.savez(os.path.join(results_dir, f'cat_{sample_idx}'), cat)    
                if args.swag:
                    log_noise = model.base.log_noise
                else:
                    log_noise = model.log_noise
                noise = np.exp(log_noise.detach().cpu().numpy()) * np.array(scaler.stds)
                np.savez(os.path.join(results_dir, f'preds_{sample_idx}'), np.array(test_preds))
                np.savez(os.path.join(results_dir, f'noise_{sample_idx}'), noise)


            #######################################################################
            #######################################################################






            # add predictions to sum_test_preds
            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)
            
            # evaluate predictions using metric function
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )
    
            # compute average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx}, sample {sample_idx} test {args.metric} = {avg_test_score:.6f}')



    #################################
    ########## Bayesian Model Average
    #################################
    # note: this is an average over Bayesian samples AND components in an ensemble
            
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

    return BMA_scores








