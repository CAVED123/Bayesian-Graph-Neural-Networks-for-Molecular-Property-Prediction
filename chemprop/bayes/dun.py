from typing import List

import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler

import numpy as np



def neg_log_likeDUN(output, target, sigma, cat):
    """
    function to compute expected loss across different network depths
    inputs:
    - preds_list, list of predictions for different depths
    - targets, single set of targets
    - noise, aleatoric noise (length 12)
    - cat, variational categorical distribution
    """
    
    target_reshape = target.reshape(1,len(target),-1)
    sigma_reshape = sigma.reshape(1, 1, len(sigma))
    
    exponent = -0.5*torch.sum((target_reshape - output)**2/sigma_reshape**2, 2)
    log_coeff = -torch.sum(torch.log(sigma)) - len(sigma) * torch.log(torch.sqrt(torch.tensor(2*np.pi)))
    
    scale = 1 / (exponent.size()[1])
    pre_expectation = - scale * torch.sum(log_coeff + exponent, 1)
    expectation = (pre_expectation * cat).sum()
    
    return expectation




def predict_MCdepth(
            model: nn.Module,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            scaler: StandardScaler,
            d) -> List[List[float]]:

    """
    makes a random prediction given a certain depth, d
    """
    
    # set model to eval mode
    model.eval()
    
    preds = []

    for batch in data_loader:
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        with torch.no_grad():
            _, batch_preds_list, _, _ = model(mol_batch, features_batch, sample=True)
    
        batch_preds = batch_preds_list[d]
        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds