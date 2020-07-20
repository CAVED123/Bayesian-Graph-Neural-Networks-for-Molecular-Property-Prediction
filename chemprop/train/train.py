import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          swag_model: nn.Module = None,
          sgld_switch: bool = False) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :param swag_model: SWAG model containing stored moments and deviations
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    loss_sum, iter_count = 0, 0

    #for batch in tqdm(data_loader, total=len(data_loader)):
    for batch in data_loader:
        # Prepare batch
        batch: MoleculeDataset
        
        # .batch_graph() returns BatchMolGraph
        # .features() returns None if no additional features
        # .targets() returns list of lists of floats containing the targets
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
        
        # mask is 1 where targets are not None
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        # where targets are None, replace with 0
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # zero gradients
        model.zero_grad()
        
        # forward pass
        preds = model(mol_batch, features_batch)

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)


        ### compute loss
        if sgld_switch:
            loss = loss_func(preds, targets, torch.exp(model.log_noise))
        else:
            if args.dataset_type == 'multiclass':
                targets = targets.long()
                loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum() # average per molecule per task   
        

        # backward pass; update weights
        loss.backward()
        optimizer.step()

        # add to loss_sum and iter_count
        loss_sum += loss.item()
        iter_count += len(batch)

        # update learning rate by taking a step
        if isinstance(scheduler, NoamLR):
            scheduler.step()

        # increment n_iter (total number of examples across epochs)
        n_iter += len(batch)
        
        # determine reporting frequency
        log_frequency = args.log_frequency_sgld if sgld_switch else args.log_frequency

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            
            ### they seem to report something funny here... check it out?
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)
        
        # SWAG update
        if (swag_model is not None) and ((n_iter // args.batch_size) % args.c_swag == 0):
            swag_model.collect_model(model)

    return n_iter















