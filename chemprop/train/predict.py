from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.bayes_utils import enable_dropout
from chemprop.bayes import BayesLinear


def predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            test_data: bool = False,
            bbp_sample: bool = False) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param args: Arguments.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :param test_data: Flag indicating whether data is test data.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    
    ########## detection of gp or bayeslinear layer or DUN
    
    try:
        model.gp_layer
    except:
        gp = False
    else:
        gp = True
        
    bbp = False
    for layer in model.children():
        if isinstance(layer, BayesLinear):
            bbp = True
            break

    try:
        model.log_cat
    except:
        dun = False
    else:
        dun = True
    


    
    
    # set model to eval mode
    model.eval()
    
    # enable dropout layers with test data, if args.test_dropout == True
    if args.test_dropout and test_data:
        model.apply(enable_dropout)

    preds = []

    #for batch in tqdm(data_loader, disable=disable_progress_bar):
    for batch in data_loader:
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        with torch.no_grad():
            if gp:
                batch_preds = model(mol_batch, features_batch).mean
            elif bbp:
                if dun:
                    batch_preds, _, _, _ = model(mol_batch, features_batch, sample=bbp_sample)
                else:
                    batch_preds, _ = model(mol_batch, features_batch, sample=bbp_sample)
            else:
                batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
