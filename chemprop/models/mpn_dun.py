from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.bayes import BayesLinear


class MPNEncoderDUN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(MPNEncoderDUN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.dropout_mpnn = args.dropout_mpnn

        # dun args
        self.prior_sig = args.prior_sig_dun
        self.depth_min = args.depth_min
        self.depth_max = args.depth_max

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout_mpnn)
        # Activation
        self.act_func = get_activation_function(args.activation)
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim

        w_h_input_size = self.hidden_size
        
        # Bayes linear layers
        self.W_i = BayesLinear(input_dim, self.hidden_size, self.prior_sig, bias=self.bias)
        self.W_h = BayesLinear(w_h_input_size, self.hidden_size, self.prior_sig, bias=self.bias)
        self.W_o = BayesLinear(self.atom_fdim + self.hidden_size, self.hidden_size, self.prior_sig)



    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                sample = False) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        f_atoms_or_bonds = f_atoms if self.atom_messages else f_bonds
        
        
        
        
        ##### LAYER FOR HIDDEN STATE INITIALISATION #####
        input, kl = self.W_i(f_atoms_or_bonds, sample)
        tkl = kl
        message = self.act_func(input)  # num_bonds x hidden_size
        #################################################
        
        
        
        mol_vecs_list = []        
        # Message passing
        for depth in range(self.depth_max):
            
            if depth != 0:
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
                
                ##### LAYER FOR HIDDEN STATE UPDATES #####
                message, kl = self.W_h(message, sample)
                if depth == self.depth_max - 1:
                    tkl += kl # ONLY ADD ON KL LOSS ONCE
                
                message = self.act_func(input + message)  # num_bonds x hidden_size
                message = self.dropout_layer(message)  # num_bonds x hidden
                ##########################################
        
        

            # save outputs for final 4 depths
            if depth >= self.depth_min - 1:
            
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
                
                ##### LAYER FOR ATOM REPRESENTATION #####
                atom_hiddens, kl = self.W_o(a_input, sample)
                if depth == self.depth_max - 1:
                    tkl += kl # ONLY ADD ON KL LOSS ONCE       
                atom_hiddens = self.act_func(atom_hiddens)  # num_atoms x hidden
                atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
                #########################################
                
                # Readout
                mol_vecs = []
                for i, (a_start, a_size) in enumerate(a_scope):
                    if a_size == 0:
                        mol_vecs.append(self.cached_zero_vector)
                    else:
                        cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                        mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                        mol_vec = mol_vec.sum(dim=0) / a_size
                        mol_vecs.append(mol_vec)
                mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
                mol_vecs_list.append(mol_vecs)
        



        return mol_vecs_list, tkl
        
        


class MPNDUN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNDUN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)
        self.encoder = MPNEncoderDUN(args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                sample = False) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings, a list of RDKit molecules, or a BatchMolGraph.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        
        if type(batch) != BatchMolGraph:
            batch = mol2graph(batch)

        
        return self.encoder.forward(batch, features_batch, sample)

    
    
    
    
    
    
    
    
    
    
    
    
    
