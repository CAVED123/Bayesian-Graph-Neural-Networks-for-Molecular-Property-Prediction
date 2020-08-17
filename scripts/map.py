# script to generate map results
# NOTE: MUST CHANGE QM9 TO qm9 WHEN RUNNING ON CLUSTER
# NOTE: checkpoint folder (save_dir) must be created before running

import os
import torch

# checks
print('working directory is:')
print(os.getcwd())
print('is CUDA available?')
print(torch.cuda.is_available())

# imports
from chemprop.args import TrainArgs
from chemprop.train.run_training import run_training

# instantiate args class and load from dict
args = TrainArgs()
args.from_dict({
    'dataset_type': 'regression',
    'data_path': '/home/willlamb/chempropBayes/data/qm9.csv'
})



##################### ARGS #####################

# architecture
args.hidden_size = 500
args.depth = 5
args.ffn_num_layers = 3
args.activation = 'ReLU'
args.ffn_hidden_size = args.hidden_size
args.features_path = None
args.features_generator = None
args.atom_messages = False
args.undirected = False
args.bias = False

# data
args.max_data_size = 150000 # full data set
args.seed = 0 # seed for data splits
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)

# metric
args.metric = 'mae'

################################################

# names and directories
args.save_dir = '/home/willlamb/checkpoints/map'
args.results_dir = '/home/willlamb/results/map'
args.wandb_proj = 'rocket'
args.wandb_name = 'map'

# ensembling
args.ensemble_size = 5
args.pytorch_seeds = [0,1,2,3,4]

# samples
args.samples = 1

### map ###
args.warmup_epochs = 2.0
args.noam_epochs = 100
args.epochs = 200

args.init_lr = 1e-4
args.max_lr = 1e-3
args.final_lr = 1e-4

args.init_log_noise = -2
args.weight_decay = 0.01

################################################

# run
results = run_training(args)