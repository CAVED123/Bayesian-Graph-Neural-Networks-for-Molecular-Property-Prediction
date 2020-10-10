# Bayesian Molecular Property Prediction

This repo is a fork of [Chemprop](https://github.com/chemprop/chemprop). We apply a set of Bayesian methods to the Chemprop directed message passing neural network (D-MPNN). The code can be used to assess predictive accuracy, calibration and performance on a downstream molecular search task.

## Methods

The code contains implementations of eight methods, abbreviated as follows:
* **MAP**: classical *maximum a posteriori* training; we find the regularised maximum likelihood solution.
* **GP**: the final layer of the readout FFN is replaced with a GPyTorch variational GP (https://docs.gpytorch.ai/en/v1.2.0/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html). We train the resulting model end-to-end (deep kernel learning).
* **DropR**: MC dropout across readout FFN layers (https://arxiv.org/abs/1506.02142).
* **DropA**: MC dropout over the full D-MPNN.
* **SWAG**: Stochastic Weight Averaging - Gaussian (https://arxiv.org/abs/1902.02476).
* **SGLD**: Stochastic Gradient Langevin Dynamics (https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf).
* **BBP**: Bayes by Backprop (https://arxiv.org/abs/1505.05424). We use 'local reparameterisation' as a variance reduction technique (https://arxiv.org/abs/1506.02557).
* **DUN**: A novel depth uncertainty network which permits inference over both weights and the number of message passing iterations. Our DUN combines Bayes by Backprop with the 'vanilla' DUN proposed by Antoran et al. (https://arxiv.org/abs/2006.08437).

If you're new to Bayesian learning, these are excellent resources (they helped me a lot!):
1. 'The Case for Bayesian Deep Learning' by Andrew Gordon Wilson (https://arxiv.org/abs/2001.10995)
2. The first two chapters of Yarin Gal's PhD thesis (http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)
3. The first two chapters of the GP book (http://www.gaussianprocess.org/gpml/chapters/RW.pdf)

## A guide to the code

If you're reading the code for the first time, the best place to start is `chemprop/train/run_training.py`. The `run_training` function inside this file executes a run of the core experiment, used to assess predictive accuracy and calibration. `run_training` contains an outer loop over ensemble members and an inner loop over samples. For each sample, the function returns predictive means and learned aleatoric uncertainty.

For certain Bayesian methods, `run_training` calls other training loop functions. These are housed within the folder `chemprop/train/bayes_tr`. Important classes and functions for Bayesan implementations are housed within the folder `chemprop/bayes`.

The secondary experiment is molecular search. The main training loop for this experiment is found in the file `chemprop/train/pdts.py` (which contains the `pdts` function).

We run experiments via scripts inside the `scripts` folder. These scripts set hyperparameter values and then call either `run_training` or `pdts`. Hyperparameter settings for all our experiments are listed in the file `scripts/bayesHyp.py`.

## Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:
```
smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
...
```

By default, it is assumed that the SMILES are in the first column and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the `--smiles_column <column>` and `--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from [MoleculeNet](http://moleculenet.ai/) and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Installation

The easiest way to install the `chemprop` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop`
3. `conda env create -f environment.yml`
4. `conda activate chemprop` (or `source activate chemprop` for older versions of conda)

If you would like to use functions or classes from `chemprop` in your own code, you can install `chemprop` as a pip package as follows:

1. `cd /path/to/chemprop`
2. `pip install -e .`

Then you can use `import chemprop` or `from chemprop import ...` in your other code.

## Logging

## Results

We compared our model against MolNet by Wu et al. on all of the MolNet datasets for which we could reproduce their splits (all but Bace, Toxcast, and qm7). When there was only one fold provided (scaffold split for BBBP and HIV), we ran our model multiple times and reported average performance. In each case we optimize hyperparameters on separate folds, use rdkit_2d_normalized features when useful, and compare to the best-performing model in MolNet as reported by Wu et al. We did not ensemble our model in these results.

Results on regression datasets (lower is better)

Dataset | Size | Metric | Ours | MolNet Best Model |
| :---: | :---: | :---: | :---: | :---: |
QM8 | 21,786 | MAE | 0.011 ± 0.000 | 0.0143 ± 0.0011 |
QM9 | 133,885 | MAE | 2.666 ± 0.006 | 2.4 ± 1.1 |
ESOL | 1,128 | RMSE | 0.555 ± 0.047 | 0.58 ± 0.03 |
FreeSolv | 642 | RMSE | 1.075 ± 0.054 | 1.15 ± 0.12 |
Lipophilicity | 4,200 | RMSE | 0.555 ± 0.023 | 0.655 ± 0.036 |
PDBbind (full) | 9,880 | RMSE | 1.391 ± 0.012 | 1.25 ± 0 | 
PDBbind (core) | 168 | RMSE | 2.173 ± 0.090 | 1.92 ± 0.07 | 
PDBbind (refined) | 3,040 | RMSE | 1.486 ± 0.026 | 1.38 ± 0 | 

Results on classification datasets (higher is better)

| Dataset | Size | Metric | Ours | MolNet Best Model |
| :---: | :---: | :---: | :---: | :---: |
| PCBA | 437,928 | PRC-AUC | 0.335 ± 0.001 |  0.136 ± 0.004 |
| MUV | 93,087 | PRC-AUC | 0.041 ± 0.007 | 0.184 ± 0.02 |
| HIV | 41,127 | ROC-AUC | 0.776 ± 0.007 | 0.792 ± 0 |
| BBBP | 2,039 | ROC-AUC | 0.737 ± 0.001 | 0.729 ± 0 |
| Tox21 | 7,831 | ROC-AUC | 0.851 ± 0.002 | 0.829 ± 0.006 |
| SIDER | 1,427 | ROC-AUC | 0.676 ± 0.014 | 0.648 ± 0.009 |
| ClinTox | 1,478 | ROC-AUC | 0.864 ± 0.017 | 0.832 ± 0.037 |
