import torch
import torch.nn as nn
import numpy as np

from .mpn_dun import MPNDUN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function
from chemprop.bayes import BayesLinear


class MoleculeModelDUN(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MoleculeModelDUN, self).__init__()
        
        self.ffn_num_layers = args.ffn_num_layers
        self.output_size = args.num_tasks
        self.prior_sig = args.prior_sig_dun
        
        ######### ENCODER
        self.encoder = MPNDUN(args)
                        
        ######### ACTIVATION LAYER
        self.act_func = get_activation_function(args.activation)


        ######### LINEAR LAYERS (handles up to 4 layers)
            
        # set first linear dimension
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size
                


        # if single layer
        if args.ffn_num_layers == 1:
            self.layer_single = BayesLinear(first_linear_dim, self.output_size, self.prior_sig)
        
        # if multiple layers
        else:
            self.layer_in = BayesLinear(first_linear_dim, args.ffn_hidden_size, self.prior_sig)
            if args.ffn_num_layers > 2:
                self.layer_hid_1 = BayesLinear(args.ffn_hidden_size, args.ffn_hidden_size, self.prior_sig)
            if args.ffn_num_layers > 3:
                self.layer_hid_2 = BayesLinear(args.ffn_hidden_size, args.ffn_hidden_size, self.prior_sig)
            self.layer_out = BayesLinear(args.ffn_hidden_size, self.output_size, self.prior_sig)

        # create log noise parameter
        self.create_log_noise(args)
            
            
    def create_log_noise(self, args: TrainArgs):
        self.log_noise = nn.Parameter(torch.ones(args.num_tasks))



    def create_log_cat(self, args: TrainArgs):
        """
        creates variational categorical distribution over depths
        """
        depth_range = (args.depth_max - args.depth_min) + 1
        self.log_cat = nn.Parameter(torch.ones(depth_range)*args.log_cat_init)
    
    
    def forward(self, *input, sample = False):
        
        # pass input through encoder
        X_list, tkl_encoder = self.encoder(*input, sample)

        # loop through X_list
        output_list = []
        for X_in in X_list:
            tkl = tkl_encoder

            # if single layer
            if self.ffn_num_layers == 1:
                output, kl = self.layer_single(X_in, sample)
                tkl += kl
            
            # if multiple layers
            else:    
                
                # initial layer
                X, kl = self.layer_in(X_in, sample)
                tkl += kl
                
                # hidden layers
                if self.ffn_num_layers > 2:
                    X = self.act_func(X)
                    X, kl = self.layer_hid_1(X, sample)
                    tkl += kl
                    
                if self.ffn_num_layers > 3:
                    X = self.act_func(X)
                    X, kl = self.layer_hid_2(X, sample)
                    tkl += kl
                    
                # output layer
                X = self.act_func(X)
                output, kl = self.layer_out(X, sample)
                tkl += kl

            output_list.append(output)
        
        # output_list to tensor
        output_list = torch.stack(output_list)

        # normalise categorical
        cat = torch.exp(self.log_cat) / torch.sum(torch.exp(self.log_cat))

        # compute single output
        cat_reshape = cat.reshape(len(cat),1,1)
        output = torch.sum(cat_reshape*output_list,0)

        # categorical kl
        dkl = (torch.sum(cat*torch.log(cat*len(cat)))).item()

        return output, output_list, tkl, dkl
        



