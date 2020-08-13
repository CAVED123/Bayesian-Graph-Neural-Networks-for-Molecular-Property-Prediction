import csv
import os
import sys
import numpy as np
import torch
import pickle
import copy

from logging import Logger
from typing import List
from tqdm import trange

from torch.optim.lr_scheduler import ExponentialLR

# cd to chempropBayes
#%cd /home/willlamb/chempropBayes
#os.chdir('/home/willlamb/chempropBayes/')
print('working directory is:')
print(os.getcwd())

print('is CUDA available?')
print(torch.cuda.is_available())


# imports
from chemprop.train.run_training import run_training
from chemprop.args import TrainArgs
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data





# instantiate args class and load from dict
args = TrainArgs()
args.from_dict({
    'dataset_type': 'regression',
    'data_path': '/home/willlamb/chempropBayes/data/qm9.csv'
})

# location for model checkpoints to be saved
args.save_dir = '/home/willlamb/chempropBayes/log'
#args.checkpoint_paths = ['/home/willlamb/chempropBayes/log/model_0/model.pt']

### args (non-model)

# seed for splitting and loading data
args.seed = 0

# data
args.max_data_size = 150000
args.features_path = None
args.features_generator = None

# splitting data
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)

# evaluation metric
args.metric = 'mae'

# epochs and logging
args.epochs = 200
args.log_frequency = 2000

### args (model)

# seed for random initial weights
args.pytorch_seed = 0

# message passing
args.atom_messages = False
args.undirected = False
args.bias = False
args.hidden_size = 500
args.depth = 5

# FFN
args.ffn_hidden_size = args.hidden_size
args.ffn_num_layers = 3

# shared
args.activation = 'ReLU'





args.ensemble_size = 1
args.samples = 1
args.init_log_noise = -2
args.init_lr = 1e-5  # Initial learning rate
args.max_lr= 1e-5  # Maximum learning rate
args.final_lr= 1e-5  # Final learning rate




args.prior_sig_bbp = 1
args.bbp = False

results_MAP = run_training(args)
#np.savez(args.save_dir+'/results_MAP', results_MAP)