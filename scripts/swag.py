# script to generate swag results
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
args.save_dir = '/home/willlamb/checkpoints/swag'
args.results_dir = '/home/willlamb/results/swag'
args.wandb_proj = 'official2b'
args.wandb_name = 'swag'
args.checkpoint_path = '/home/willlamb/checkpoints/map'

# ensembling and samples
args.ensemble_size = 5
args.pytorch_seeds = [0,1,2,3,4]
args.samples = 30

### swag ###
args.swag = True
args.epochs = 0

args.batch_size_swag = 50
args.lr_swag = 2e-5
args.weight_decay_swag = 0.01
args.momentum_swag = 0

args.burnin_swag = 20
args.epochs_swag = 100
args.val_threshold = 2.8

args.cov_mat = True
args.max_num_models = 20
args.block = False

################################################

# run
results = run_training(args)
