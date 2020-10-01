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
args.results_dir = '/home/willlamb/results_pdts/swag_thom2'
args.save_dir = '/home/willlamb/checkpoints_pdts/swag_thom2'
args.checkpoint_path = '/home/willlamb/checkpoints_pdts/map_greedy'
args.wandb_proj = 'lanterne_swag2'
args.wandb_name = 'swag_thom'
args.thompson = True

### swag ###
args.swag = True
args.samples = 50

args.pdts = True
args.pdts_batches = 30

args.epochs_init_map = 0
args.epochs = 0

args.lr_swag = 1e-5
args.lr = 1e-4
args.weight_decay_swag = 0.01
args.momentum_swag = 0

args.burnin_swag = 300
args.epochs_swag = 400
args.loss_threshold = -5

args.cov_mat = True
args.max_num_models = 20
args.block = False

################################################


# run
results = pdts(args, model_idx = 3)
