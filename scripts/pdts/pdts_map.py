import os
import torch

# checks
print('working directory is:')
print(os.getcwd())
print('is CUDA available?')
print(torch.cuda.is_available())

# imports
from chemprop.args import TrainArgs
from chemprop.train.pdts import pdts

# instantiate args class and load from dict
args = TrainArgs()
args.from_dict({
    'dataset_type': 'regression',
    'data_path': '/Users/georgelamb/Documents/GitHub/chempropBayes/data/qm9.csv'
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
args.max_data_size = 100000
args.data_seeds = [0,1,2,3,4]
args.split_type = 'random'
args.split_sizes = (0.0005, 0.9995)

# metric
args.metric = 'mae'

# run seeds
args.pytorch_seeds = [0,1,2,3,4]

################################################

# names and directories
args.results_dir = '/Users/georgelamb/Documents/results_pdts/map_greedy'
args.wandb_proj = 'pdts_practice'
args.wandb_name = 'map_greedy'


### map ###
args.samples = 1

args.pdts = True
args.pdts_batches = 30

args.epochs_init_map = 100
args.epochs = 30

args.lr = 5e-4

args.init_log_noise = -2
args.weight_decay = 0.01


################################################


# run
results = pdts(args, model_idx = 0)
