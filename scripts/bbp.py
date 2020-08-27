# script to generate bbp results
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
args.max_data_size = 150000
args.seed = 0 # seed for data splits
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)

# metric
args.metric = 'mae'

################################################

# names and directories
args.save_dir = '/home/willlamb/checkpoints/bbp'
args.results_dir = '/home/willlamb/results/bbp'
args.wandb_proj = 'official3'
args.wandb_name = 'bbp'
args.checkpoint_path = '/home/willlamb/checkpoints/map'

# ensembling and samples
args.ensemble_size = 3
args.ensemble_start_idx = 0
args.pytorch_seeds = [0,1,2,3,4]
args.samples = 30


### bbp ###
args.bbp = True
args.epochs = 0
args.epochs_bbp = 100

args.batch_size_bbp = 50
args.lr_bbp = 1e-4
args.prior_sig_bbp = 0.05

args.rho_min_bbp = -5.5
args.rho_max_bbp = -5
args.samples_bbp = 5

args.presave_bbp = 50


################################################

# run
results = run_training(args)















