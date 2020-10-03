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
args.max_data_size = 100000
args.data_seeds = [0,1,2,3,4]
args.split_type = 'random'
args.split_sizes = (0.05, 0.95)

# metric
args.metric = 'mae'

# run seeds
args.pytorch_seeds = [0,1,2,3,4]


################################################

# names and directories
args.results_dir = '/home/willlamb/results_pdts/sgld_greedy'
args.save_dir = '/home/willlamb/checkpoints_pdts/sgld_greedy'
args.checkpoint_path = '/home/willlamb/checkpoints_pdts/map_greedy'
args.wandb_proj = 'lanterne_sgld'
args.wandb_name = 'sgld_greedy'
args.thompson = False

### sgld ###
args.sgld = True
args.samples = 10

args.pdts = True
args.pdts_batches = 30

args.epochs_init_map = 0
args.epochs = 0
args.mix_epochs = 30

args.lr_max_sgld = 5e-5
args.lr = 1e-4
args.weight_decay_sgld = 0.01

################################################


# run
results = pdts(args, model_idx = 0)
