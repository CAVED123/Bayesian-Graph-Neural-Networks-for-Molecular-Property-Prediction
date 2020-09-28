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
args.results_dir = '/home/willlamb/results_pdts/dropA_thom'
args.save_dir = '/home/willlamb/checkpoints_pdts/dropA_thom'
args.checkpoint_path = '/home/willlamb/checkpoints_pdts/dropA_thom'
args.wandb_proj = 'lanterne_dropA'
args.wandb_name = 'dropA_thom'
args.thompson = True

### dropR ###
args.samples = 50

args.pdts = True
args.pdts_batches = 30

args.epochs_init_map = 500
args.epochs = 200

args.lr = 1e-4

args.init_log_noise = -2
args.weight_decay = 0.01

args.dropout_mpnn = 0.1
args.dropout_ffn = 0.1
args.test_dropout = True  


################################################


# run
results = pdts(args, model_idx = 0)
