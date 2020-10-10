
# This file contains a consolidated list of the hyperparameter settings used for the original chempropBayes benchmarking experiments
# Two experiments:
### (i) Core experiment, used to assess predictive accuracy and calibration
### (ii) Molecular search experiment, added for submission to NeurIPS ml4molecules 2020 workshop





########################################################
# Hyperparameters shared across experiments (i) and (ii)
########################################################

# architecture
args.hidden_size = 500 # hidden size for message passing phase
args.ffn_hidden_size = args.hidden_size # hidden size for FFN
args.depth = 5 # number of message passing iterations
args.ffn_num_layers = 3 # number of FFN layers
args.activation = 'ReLU' # activation function
args.atom_messages = False # centres messages on atoms instead of on bonds
args.undirected = False # undirected edges (always sum the two relevant bond vectors)
args.bias = False # bias term

# features
args.features_only = False # use only 'additional' features for FFN, no learned representations
args.features_generator = None # method(s) of generating additional features
args.features_path = None # path(s) to additional features to use in FNN (instead of features_generator)

# other
args.batch_size = 50 # batch size
args.metric = 'mae' # evaluation metric
args.ensemble_size = 5 # number of models in ensemble
args.pytorch_seeds = [0,1,2,3,4] # seeds for random weight initialisation





###########################################
# (i) Core experiment
###########################################

# shared across experiment (i)
args.max_data_size = 150000 # number of molecules to use from QM9 (150000 > full dataset)
args.seed = 0 # seed for data splits
args.split_type = 'scaffold_balanced' # split type
args.split_sizes = (0.64, 0.16, 0.2) # [tr, val, test] proportions

# MAP
args.samples = 1 # number of posterior samples
args.warmup_epochs = 2.0 # epochs of linear lr increase
args.noam_epochs = 100 # epochs of noam scheduler
args.epochs = 200 # total number of epochs
args.init_lr = 1e-4
args.max_lr = 1e-3
args.final_lr = 1e-4
args.init_log_noise = -2 # initial value of log noise parameters
args.weight_decay = 0.01

# GP
args.gp = True
args.samples = 1 # number of posterior samples
args.batch_size_gp = 50 # batch size (GP)
args.epochs = 0 # MAP pre-training epochs (we load checkpoint rather than pre-training)
args.warmup_epochs_gp = 2 # epochs of linear lr increase (GP)
args.noam_epochs_gp = 100 # epochs of noam scheduler (GP)
args.epochs_gp = 200 # total number of epochs (GP)
args.init_lr_gp = 1e-4
args.max_lr_gp = 1e-3
args.final_lr_gp = 1e-4
args.num_inducing_points = 1200 # number of inducing points
args.weight_decay_gp = 0.01 # weight decay over featuriser parameters

# DropR, DropA
args.samples = 30 # number of posterior samples
args.warmup_epochs = 2.0 # epochs of linear lr increase
args.noam_epochs = 100 # epochs of noam scheduler
args.epochs = 300 # total number of epochs
args.init_lr = 1e-4
args.max_lr = 1e-3
args.final_lr = 1e-4
args.init_log_noise = -2 # initial value of log noise parameters
args.weight_decay = 0.01
args.dropout_mpnn = 0 # dropout probability for message passing phase (WE SET THIS TO 0.1 FOR DROPA)
args.dropout_ffn = 0.1 # dropout probability for FFN
args.test_dropout = True # switch for test time dropout

# SWAG
args.swag = True
args.samples = 30 # number of posterior samples
args.batch_size_swag = 50 # batch size (SWAG)
args.epochs = 0 # MAP pre-training epochs (we load checkpoint rather than pre-training)
args.burnin_swag = 20 # number of epochs before model collection begins
args.epochs_swag = 100 # total number of swag epochs
args.lr_swag = 2e-5 # constant SWAG lr
args.weight_decay_swag = 0.01
args.momentum_swag = 0
args.val_threshold = 2.8 # val threshold (if val acc above this, we do not collect a model)
args.cov_mat = True # switch for whether we model posterior covariance
args.max_num_models = 20 # maximum number of columns of deviation matrix
args.block = False # if True, we only model covariance within layers

# SGLD
args.sgld = True
args.samples = 20 # number of posterior samples
args.epochs = 0 # MAP pre-training epochs (we load checkpoint rather than pre-training)
args.mix_epochs = 50 # epochs between saving down samples
args.batch_size_sgld = 50 # batch size (SGLD)
args.lr_max_sgld = 1e-4
args.weight_decay_sgld = 0.01

# BBP
args.bbp = True
args.samples = 30 # number of posterior samples
args.batch_size_bbp = 50 # batch size (BBP)
args.epochs = 0 # MAP pre-training epochs (we load checkpoint rather than pre-training)
args.epochs_bbp = 100 # bbp epochs
args.lr_bbp = 1e-4
args.prior_sig_bbp = 0.05 # sigma for prior Gaussian
args.rho_min_bbp = -5.5 # we initialise rho uniformly at random between rho_min_bbp and rho_max_bbp
args.rho_max_bbp = -5 # we initialise rho uniformly at random between rho_min_bbp and rho_max_bbp
args.samples_bbp = 5 # number of forward passes per backward pass
args.presave_bbp = 50 # number of epochs before we begin saving the best model

# DUN
args.dun = True
args.samples = 100 # number of posterior samples
args.depth_min = 1 # categorical distributions over depth are from depth_min to depth_max 
args.depth_max = 5 # categorical distributions over depth are from depth_min to depth_max 
args.batch_size_dun = 50 # batch size (DUN)
args.epochs = 0 # MAP pre-training epochs (we load checkpoint rather than pre-training)
args.epochs_dun = 350 # dun epochs
args.lr_dun_min = 1e-4
args.lr_dun_max = 1e-3
args.prior_sig_dun = 0.05 # sigma for prior Gaussian
args.rho_min_dun = -5.5 # we initialise rho uniformly at random between rho_min_bbp and rho_max_bbp
args.rho_max_dun = -5 # we initialise rho uniformly at random between rho_min_bbp and rho_max_bbp
args.samples_dun = 5 # number of forward passes per backward pass
args.presave_dun = 150 # number of epochs before we begin saving the best model
args.log_cat_init = 0 # initialisation for log of variational categorical distribution





###########################################
# (ii) Molecular search experiment
###########################################

# in this section we highlight differences vs. experiment (i)
# args.thompson is set to True below; for greedy trials it is set to False

# shared across experiment (ii)
args.max_data_size = 100000 # subset of molecules to use from QM9
args.data_seeds = [0,1,2,3,4] # seeds for data splits
args.split_type = 'random' # split type
args.split_sizes = (0.05, 0.95) # [tr, test] proportions
args.pdts = True # switch for molecular search experiment
args.thompson = True # switch for Thompson sampling
args.pdts_batches = 30 # number of batch additions

# MAP
args.epochs_init_map = 500 # epochs before first batch addition (MAP/dropout)
args.epochs = 100 # epochs before each subsequent batch addition
args.lr = 1e-4 # constant lr

# GP
args.samples = 50 # posterior samples per iteration
args.epochs_init_map = 0 # epochs before first batch addition (MAP/dropout)
args.epochs_init = 1500 # epochs before first batch addition (GP)
args.epochs = 100 # epochs before each subsequent batch addition
args.lr = 1e-4 # constant lr
args.num_inducing_points = 500 # number of inducing points

# DropR, DropA
args.samples = 50 # posterior samples per iteration
args.epochs_init_map = 500 # epochs before first batch addition (MAP/dropout)
args.epochs = 200 # epochs before each subsequent batch addition
args.lr = 1e-4 # constant lr

# SWAG
args.samples = 50 # posterior samples per iteration
args.epochs_init_map = 0 # epochs before first batch addition (MAP/dropout)
args.epochs = 0 # epochs before each subsequent batch addition (non-SWAG/SGLD)
args.epochs_swag = 150 # total number of epochs before each batch addition (SWAG)
args.burnin_swag = 75 # epochs before model collection
args.lr_swag = 2e-5 # SWAG lr
args.lr = 1e-4 # constant lr (not used for SWAG)
args.loss_threshold = -5 # loss threshold (if loss above this, we do not collect a model)

# SGLD
args.samples = 10 # posterior samples per iteration
args.epochs_init_map = 0 # epochs before first batch addition (MAP/dropout)
args.epochs = 0 # epochs before each subsequent batch addition (non-SWAG/SGLD)
args.mix_epochs = 30 # epochs between saving down samples
args.lr_max_sgld = 5e-5 # max SGLD lr
args.lr = 1e-4 # constant lr (not used for SGLD)

# BBP
args.samples = 50 # posterior samples per iteration
args.epochs_init_map = 0 # epochs before first batch addition (MAP/dropout)
args.epochs_init = 1000 # epochs before first batch addition (BBP)
args.epochs = 100 # epochs before each subsequent batch addition
args.lr = 1e-4 # constant lr
args.prior_sig_bbp = 0.15 # sigma for prior Gaussian