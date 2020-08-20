# script to generate sgld results
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
    'data_path': '/Users/georgelamb/Documents/GitHub/chempropBayes/data/QM9.csv'
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
args.max_data_size = 50000
args.seed = 0 # seed for data splits
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)

# metric
args.metric = 'mae'

################################################

# names and directories
args.save_dir = '/Users/georgelamb/Documents/checkpoints/sgld'
args.results_dir = '/Users/georgelamb/Documents/results/sgld'
args.wandb_proj = 'swaggerB'
args.wandb_name = 'sgld'
args.checkpoint_path = '/Users/georgelamb/Documents/checkpoints/map'

# ensembling
args.ensemble_size = 1
args.pytorch_seeds = [0,1,2,3,4,5,6,7,8,9]

### sgld ###
args.sgld = True
args.epochs = 0

args.batch_size_sgld = 50
args.lr_base_sgld = 1e-4
args.lr_max_sgld = 2e-4
args.weight_decay_sgld = 0.01

args.burnin_sgld = 20
args.mix_epochs = 25
args.samples = 20

################################################

# run
results = run_training(args)
