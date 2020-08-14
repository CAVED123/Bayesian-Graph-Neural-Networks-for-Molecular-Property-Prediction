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

# args (general)
args.seed = 0
args.max_data_size = 75000
args.features_path = None
args.features_generator = None
args.split_type = 'scaffold_balanced'
args.split_sizes = (0.64, 0.16, 0.2)
args.metric = 'mae'
args.pytorch_seed = 0
args.atom_messages = False
args.undirected = False
args.bias = False
args.hidden_size = 500
args.depth = 5
args.ffn_hidden_size = args.hidden_size
args.ffn_num_layers = 3
args.activation = 'ReLU'

# args (MAP)
args.wandb_name = 'MAP'
args.save_dir = '/home/willlamb/chempropBayes/log'
args.ensemble_size = 1
args.samples = 1
args.epochs = 10
args.log_frequency = 960
args.init_log_noise = -2
args.init_lr = 1e-4 # Initial learning rate
args.max_lr= 1e-3 # Maximum learning rate
args.final_lr= 1e-4 # Final learning rate

# run
results_MAP = run_training(args)