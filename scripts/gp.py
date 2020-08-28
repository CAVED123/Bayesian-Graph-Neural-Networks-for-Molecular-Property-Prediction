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
    'data_path': '/Users/georgelamb/Documents/GitHub/chempropBayes/data/qm9.csv'
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
args.max_data_size = 500
args.seed = 0 # seed for data splits
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)

# metric
args.metric = 'mae'

################################################

# names and directories
args.save_dir = '/Users/georgelamb/Documents/checkpoints/gp'
args.results_dir = '/Users/georgelamb/Documents/results/gp'
args.wandb_proj = 'gp_tune'
args.wandb_name = 'gp_practice'
args.checkpoint_path = '/Users/georgelamb/Documents/checkpoints/map'

# ensembling and samples
args.ensemble_size = 1
args.ensemble_start_idx = 0
args.pytorch_seeds = [0,1,2,3,4]
args.samples = 1


### gp ###

args.gp = True
args.epochs = 0

args.batch_size_gp = 50

args.warmup_epochs_gp = 2
args.noam_epochs_gp = 4
args.epochs_gp = 20

args.init_lr_gp = 1e-4
args.max_lr_gp = 1e-3
args.final_lr_gp = 1e-4

args.num_inducing_points = 100



################################################

# run
results = run_training(args)















