# Bayesian Molecular Property Prediction

This repo is a fork of [chemprop](https://github.com/chemprop/chemprop). We apply a set of Bayesian methods to the chemprop directed message passing neural network (D-MPNN). We assess predictive accuracy, calibration and performance on a downstream molecular search task.

## Methods

The code contains implementations of eight methods, abbreviated as follows:
* **MAP**: classical *maximum a posteriori* training; we find the regularised maximum likelihood solution.
* **GP**: the final layer of the readout FFN is replaced with a GPyTorch stochastic variational GP (https://docs.gpytorch.ai/en/v1.2.0/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html). We train the resulting model end-to-end (deep kernel learning).
* **DropR**: MC dropout across readout FFN layers (https://arxiv.org/abs/1506.02142).
* **DropA**: MC dropout over the full D-MPNN.
* **SWAG**: Stochastic Weight Averaging - Gaussian (https://arxiv.org/abs/1902.02476).
* **SGLD**: Stochastic Gradient Langevin Dynamics (https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf).
* **BBP**: Bayes by Backprop (https://arxiv.org/abs/1505.05424). We use 'local reparameterisation' as a variance reduction technique (https://arxiv.org/abs/1506.02557).
* **DUN**: A novel depth uncertainty network which permits inference over both weights and the number of message passing iterations (depth). Our DUN combines Bayes by Backprop with the 'vanilla' DUN proposed by Antoran et al. (https://arxiv.org/abs/2006.08437).

If you're new to Bayesian learning, these are excellent resources (they helped me a lot!):
1. 'The Case for Bayesian Deep Learning' by Andrew Gordon Wilson (https://arxiv.org/abs/2001.10995).
2. The first two chapters of Yarin Gal's PhD thesis (http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf).
3. The first two chapters of the GP book (http://www.gaussianprocess.org/gpml/chapters/RW.pdf).

## A guide to the code

If you're reading the code for the first time, the best place to start is `/chemprop/train/run_training.py`. The `run_training()` function inside this file executes a run of our core experiment, used to assess predictive accuracy and calibration. `run_training()` contains an outer loop over ensemble members and an inner loop over samples. For each posterior sample, the function saves down predictive means and learned aleatoric uncertainty.

`run_training()` calls Bayesian training loop functions. These are housed within the folder `/chemprop/train/bayes_tr/`. Important classes and functions for Bayesian implementations are housed within the folder `/chemprop/bayes/`.

The second experiment is molecular search. The main code for this experiment is found in the file `/chemprop/train/pdts.py` (containing the `pdts()` function).

We run experiments via scripts inside the `/scripts/` folder. These scripts set hyperparameter values and then call either `run_training()` or `pdts()`. Hyperparameter settings for all our experiments are listed in the file `/scripts/bayesHyp.py`.

## Data

We perform all experiments using the QM9 regression dataset, comprising 12 tasks. The chempropBayes code could be adapted to run with any [MoleculeNet](http://moleculenet.ai/) dataset or with ChEMBL. The original chemprop code has this functionality.

Datasets from MoleculeNet and a 450K subset of ChEMBL from [http://www.bioinf.jku.at/research/lsc/index.html](http://www.bioinf.jku.at/research/lsc/index.html) have been preprocessed and are available in `data.tar.gz`. To uncompress them, run `tar xvzf data.tar.gz`.

## Installation

The easiest way to install the `chempropBayes` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chempropBayes`
3. `conda env create -f environment.yml`
4. `conda activate chempropBayes`

If you would like to use functions or classes from `chempropBayes` in your own code, you can install `chempropBayes` as a pip package as follows:

1. `cd /path/to/chempropBayes`
2. `pip install -e .`

Then you can use `import chempropBayes` or `from chempropBayes import ...` in your other code.

## Logging

`chempropBayes` is setup for logging with [wandb](https://www.wandb.com/). When running on a GPU offline, set `os.environ['WANDB_MODE'] = 'dryrun'`. For most methods, the code logs negative log likelihood, validation accuracy and learning rate.

## Results

Results for single models (as opposed to model ensembles) are as follows. We report Accuracy (measured by Mean Rank across QM9 tasks; lower is better), Miscalibration Area (lower is better) and Search Scores (higher is better). We present the mean and standard deviation across 5 runs. MAs are computed with post-hoc *t*-distribution likelihoods and presented X 10<sup>2</sup>. Search Scores equate to the % of the top 1% of molecules discovered after 30 batch additions, for Thompson sampling trials.


Method | Accuracy (Mean Rank) | Miscalibration Area | Search Score (Thompson) |
| :---: | :---: | :---: | :---: |
MAP   | 4.08 ± 0.16 |  4.20 ± 0.42 |      n/a     |
GP    | 3.87 ± 0.42 |  9.12 ± 0.98 | 75.86 ± 0.85 |
DropR | 7.05 ± 0.15 | 14.59 ± 0.37 | 76.02 ± 1.09 |
DropA | 7.87 ± 0.19 | 16.58 ± 0.47 | 77.34 ± 0.88 |
SWAG  | 3.55 ± 0.12 |  9.29 ± 1.78 | 73.14 ± 0.59 |
SGLD  | 3.23 ± 0.51 |  1.79 ± 1.03 | 69.70 ± 1.31 |
BBP   | 1.95 ± 0.40 |  4.22 ± 0.57 | 61.63 ± 3.87 |
DUN   | 4.40 ± 0.25 |  4.36 ± 0.39 |       -      |
