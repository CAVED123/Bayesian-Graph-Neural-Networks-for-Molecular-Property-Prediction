# script to generate dun results
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
#args.depth = 5
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
args.save_dir = '/home/willlamb/checkpoints/dun'
args.results_dir = '/home/willlamb/results/dun'
args.wandb_proj = 'dun_tune'
args.wandb_name = 'dun_practice19'
args.checkpoint_path = None

# ensembling and samples
args.ensemble_size = 1
args.ensemble_start_idx = 0
args.pytorch_seeds = [0,1,2,3,4]
args.samples = 30


### dun ###

args.dun = True
args.depth_min = 3
args.depth_max = 7

args.epochs = 0
args.epochs_dun = 300

args.batch_size_dun = 50
args.lr_dun_min = 1e-4
args.lr_dun_max = 1e-3
args.prior_sig_dun = 0.05

args.rho_min_dun = -5.5
args.rho_max_dun = -5
args.samples_dun = 5

args.presave_dun = 200

args.log_cat_init = 0


################################################

# run
results = run_training(args)







