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
import gpytorch

from chemprop.data.data import MoleculeDatapoint, MoleculeDataset

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

from .bayes_tr.swag_tstr import train_swag_pdts
from .bayes_tr.sgld_tstr import train_sgld_pdts
from .bayes_tr.gp_tr import train_gp
from .bayes_tr.bbp_tr import train_bbp
from .bayes_tr.dun_tr import train_dun
from chemprop.bayes import predict_std_gp, predict_MCdepth
from chemprop.models import MoleculeModelBBP
from chemprop.bayes import BayesLinear
from chemprop.bayes import GPLayer, DKLMoleculeModel, initial_inducing_points


def pdts(args: TrainArgs, model_idx):
    """
    preliminary experiment with PDTS (approximate BO)
    we use a data set size of 50k and run until we have trained with 15k data points
    our batch size is 50
    we initialise with 1000 data points
    """

    

    ######## set up all logging ########
    logger = None
    
    # make save_dir
    save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
    makedirs(save_dir)

    # make results_dir
    results_dir = args.results_dir
    makedirs(results_dir)

    # initialise wandb
    #os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(
        name=args.wandb_name+'_'+str(model_idx),
        project=args.wandb_proj,
        reinit=True)
    #print('WANDB directory is:')
    #print(wandb.run.dir)
    ####################################



    ########## get data
    args.task_names = args.target_columns or get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()



    ########## SMILES of top 1%
    top1p = np.array(MoleculeDataset(data).targets())
    top1p_idx = np.argsort(-top1p[:,0])[:int(args.max_data_size*0.01)]
    SMILES = np.array(MoleculeDataset(data).smiles())[top1p_idx]



    ########## initial data splits
    args.seed = args.data_seeds[model_idx]    
    data.shuffle(seed=args.seed)
    sizes = args.split_sizes
    train_size = int(sizes[0] * len(data))
    train_orig = data[:train_size]
    test_orig = data[train_size:]
    train_data, test_data = copy.deepcopy(MoleculeDataset(train_orig)), copy.deepcopy(MoleculeDataset(test_orig))
    args.train_data_size = len(train_data)



    ########## standardising
    # features (train and test)
    features_scaler = train_data.normalize_features(replace_nan_token=0)
    test_data.normalize_features(features_scaler)
    # targets (train)
    train_targets = train_data.targets()
    test_targets = test_data.targets()
    scaler = StandardScaler().fit(train_targets)
    scaled_targets = scaler.transform(train_targets).tolist()
    train_data.set_targets(scaled_targets)

    

    ########## loss, metric functions
    loss_func = neg_log_like
    metric_func = get_metric_func(metric=args.metric)



    ########## data loaders
    if len(data) <= args.cache_cutoff:
        cache = True
        num_workers = 0
    else:
        cache = False
        num_workers = args.num_workers
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )



    ########## instantiating model, optimiser, scheduler (MAP)
    # set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seeds[model_idx])
    # build model
    print(f'Building model {model_idx}')
    model = MoleculeModel(args)
    print(model)
    print(f'Number of parameters = {param_count(model):,}')
    if args.cuda:
        print('Moving model to cuda')
    model = model.to(args.device)
    # optimizer
    optimizer = Adam([
        {'params': model.encoder.parameters()},
        {'params': model.ffn.parameters()},
        {'params': model.log_noise, 'weight_decay': 0}
        ], lr=args.lr, weight_decay=args.weight_decay)
    # learning rate scheduler
    scheduler = scheduler_const([args.lr])









    ####################################################################
    ####################################################################
    # FIRST THOMPSON ITERATION

    ### scores array
    ptds_scores = np.ones(args.pdts_batches + 1)
    batch_no = 0
    
    ### fill for batch 0
    SMILES_train = np.array(train_data.smiles())
    SMILES_stack = np.hstack((SMILES, SMILES_train))
    overlap = len(SMILES_stack) - len(np.unique(SMILES_stack))
    prop = overlap / len(SMILES)
    ptds_scores[batch_no] = prop
    wandb.log({"Proportion of top 1%": prop, "batch_no": batch_no}, commit=False)

    ### train MAP posterior
    gp_switch = False
    likelihood = None
    bbp_switch = None
    n_iter = 0
    for epoch in range(args.epochs_init_map):
        n_iter = train(
            model=model,
            data_loader=train_data_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
            bbp_switch = bbp_switch
        )
        # save to save_dir
        #if epoch == args.epochs_init_map - 1:
            #save_checkpoint(os.path.join(save_dir, f'model_{batch_no}.pt'), model, scaler, features_scaler, args)
    # if X load from checkpoint path
    if args.bbp or args.gp or args.swag or args.sgld:
        model = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model_{batch_no}.pt', device=args.device, logger=None)



    ########## BBP
    if args.bbp:
        model_bbp = MoleculeModelBBP(args) # instantiate with bayesian linear layers
        for (_, param_bbp), (_, param_pre) in zip(model_bbp.named_parameters(), model.named_parameters()):
            param_bbp.data = copy.deepcopy(param_pre.data.T) # copy over parameters
        # instantiate rhos
        for layer in model_bbp.children():
            if isinstance(layer, BayesLinear):
                layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
        for layer in model_bbp.encoder.encoder.children():
            if isinstance(layer, BayesLinear):
                layer.init_rho(args.rho_min_bbp, args.rho_max_bbp)
        model = model_bbp # name back
        # move to cuda
        if args.cuda:
            print('Moving bbp model to cuda')
            model = model.to(args.device)
        # optimiser and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = scheduler_const([args.lr])
        
        bbp_switch = 2
        n_iter = 0
        for epoch in range(args.epochs_init):
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                bbp_switch = bbp_switch
            )
    


    ########## GP
    if args.gp:
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
        # likelihood (rank 0 restricts to diagonal matrix)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=12, rank=0)
        # model and likelihood to CUDA
        if args.cuda:
            model.cuda()
            likelihood.cuda()
        # loss object
        loss_func = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=args.train_data_size)
        # optimiser and scheduler
        params_list = [
            {'params': model.feature_extractor.parameters(), 'weight_decay': args.weight_decay_gp},
            {'params': model.gp_layer.hyperparameters()},
            {'params': model.gp_layer.variational_parameters()},
            {'params': likelihood.parameters()},
        ]    
        optimizer = torch.optim.Adam(params_list, lr = args.lr)
        scheduler = scheduler_const([args.lr])

        gp_switch = True
        n_iter = 0
        for epoch in range(args.epochs_init):
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                gp_switch=gp_switch,
                likelihood=likelihood
            )
        


    ########## SWAG
    if args.swag:
        model_core = copy.deepcopy(model)
        model = train_swag_pdts(
            model_core,
            train_data_loader,
            loss_func,
            scaler,
            features_scaler,
            args,
            save_dir,
            batch_no
        )



    ########## SGLD
    if args.sgld:
        model = train_sgld_pdts(
            model,
            train_data_loader,
            loss_func,
            scaler,
            features_scaler,
            args,
            save_dir,
            batch_no
        )



    ### find top_idx
    top_idx = [] # need for thom
    sum_test_preds = np.zeros((len(test_orig), args.num_tasks)) # need for greedy
    for sample in range(args.samples):
        
        # draw model from SWAG posterior
        if args.swag:
            model.sample(scale=1.0, cov=args.cov_mat, block=args.block)

        # retrieve sgld sample
        if args.sgld:
            model = load_checkpoint(args.save_dir + f'/model_{model_idx}/model_{batch_no}/model_{sample}.pt', device=args.device, logger=logger)
        
        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            args=args,
            scaler=scaler,
            test_data=True,
            gp_sample = args.thompson,
            bbp_sample=True)
        test_preds = np.array(test_preds)
        # thompson bit
        rank = 0
        
        # base length
        if args.sgld:
            base_length = 5 * sample + 4
        else:
            base_length = sample
            
        while args.thompson and (len(top_idx) <= base_length):
            top_unique_molecule = np.argsort(-test_preds[:,0])[rank]
            rank += 1
            if top_unique_molecule not in top_idx:
                top_idx.append(top_unique_molecule)
        # add to sum_test_preds
        sum_test_preds += test_preds
        # print
        print('done sample ' + str(sample))
    # final top_idx
    if args.thompson:
        top_idx = np.array(top_idx)
    else:
        sum_test_preds /= args.samples
        top_idx = np.argsort(-sum_test_preds[:,0])[:50]

    ### transfer from test to train
    top_idx = -np.sort(-top_idx)
    for idx in top_idx:
        train_orig.append(test_orig.pop(idx))
    train_data, test_data = copy.deepcopy(MoleculeDataset(train_orig)), copy.deepcopy(MoleculeDataset(test_orig))
    args.train_data_size = len(train_data)
    if args.gp:
        loss_func = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=args.train_data_size)
    print(args.train_data_size)

    ### standardise features (train and test; using original features_scaler)
    train_data.normalize_features(features_scaler)
    test_data.normalize_features(features_scaler)

    ### standardise targets (train only; using original scaler)
    train_targets = train_data.targets()
    scaled_targets_tr = scaler.transform(train_targets).tolist()
    train_data.set_targets(scaled_targets_tr)

    ### create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )

    ####################################################################
    ####################################################################






    ##################################
    ########## thompson sampling loop
    ##################################
    
    for batch_no in range(1, args.pdts_batches+1):
        
        ### fill in ptds_scores
        SMILES_train = np.array(train_data.smiles())
        SMILES_stack = np.hstack((SMILES, SMILES_train))
        overlap = len(SMILES_stack) - len(np.unique(SMILES_stack))
        prop = overlap / len(SMILES)
        ptds_scores[batch_no] = prop
        wandb.log({"Proportion of top 1%": prop, "batch_no": batch_no}, commit=False)

        ### train posterior
        n_iter = 0
        for epoch in range(args.epochs):
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                gp_switch=gp_switch,
                likelihood=likelihood,
                bbp_switch = bbp_switch
            )
            # save to save_dir
            #if epoch == args.epochs - 1:
                #save_checkpoint(os.path.join(save_dir, f'model_{batch_no}.pt'), model, scaler, features_scaler, args)
        # if swag, load checkpoint
        if args.swag:
            model_core = load_checkpoint(args.checkpoint_path + f'/model_{model_idx}/model_{batch_no}.pt', device=args.device, logger=None)



        ########## SWAG
        if args.swag:
            model = train_swag_pdts(
                model_core,
                train_data_loader,
                loss_func,
                scaler,
                features_scaler,
                args,
                save_dir,
                batch_no
            )
        


        ########## SGLD
        if args.sgld:
            model = train_sgld_pdts(
                model,
                train_data_loader,
                loss_func,
                scaler,
                features_scaler,
                args,
                save_dir,
                batch_no
            )



        ### find top_idx
        top_idx = [] # need for thom
        sum_test_preds = np.zeros((len(test_orig), args.num_tasks)) # need for greedy
        for sample in range(args.samples):

            # draw model from SWAG posterior
            if args.swag:
                model.sample(scale=1.0, cov=args.cov_mat, block=args.block)

            # retrieve sgld sample
            if args.sgld:
                model = load_checkpoint(args.save_dir + f'/model_{model_idx}/model_{batch_no}/model_{sample}.pt', device=args.device, logger=logger)

            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                args=args,
                scaler=scaler,
                test_data=True,
                gp_sample = args.thompson,
                bbp_sample=True)
            test_preds = np.array(test_preds)
            # thompson bit
            rank = 0
            
            # base length
            if args.sgld:
                base_length = 5 * sample + 4
            else:
                base_length = sample
            
            while args.thompson and (len(top_idx) <= base_length):
                top_unique_molecule = np.argsort(-test_preds[:,0])[rank]
                rank += 1
                if top_unique_molecule not in top_idx:
                    top_idx.append(top_unique_molecule)
            # add to sum_test_preds
            sum_test_preds += test_preds
            # print
            print('done sample ' + str(sample))
        # final top_idx
        if args.thompson:
            top_idx = np.array(top_idx)
        else:
            sum_test_preds /= args.samples
            top_idx = np.argsort(-sum_test_preds[:,0])[:50]

        ### transfer from test to train
        top_idx = -np.sort(-top_idx)
        for idx in top_idx:
            train_orig.append(test_orig.pop(idx))
        train_data, test_data = copy.deepcopy(MoleculeDataset(train_orig)), copy.deepcopy(MoleculeDataset(test_orig))
        args.train_data_size = len(train_data)
        if args.gp:
            loss_func = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=args.train_data_size)
        print(args.train_data_size)

        ### standardise features (train and test; using original features_scaler)
        train_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)

        ### standardise targets (train only; using original scaler)
        train_targets = train_data.targets()
        scaled_targets_tr = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets_tr)

        ### create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            cache=cache,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            cache=cache
        )

    

    # save scores
    np.savez(os.path.join(results_dir, f'ptds_{model_idx}'), ptds_scores)
