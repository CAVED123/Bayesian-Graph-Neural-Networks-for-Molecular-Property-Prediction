"""
    implementation of SGLD
    based on Javier Antoran's implementation of SGLD for homoscedastic regression
    source: https://github.com/JavierAntoran/Bayesian-Neural-Networks
"""

### we need to:
# 1) produce a new model with log noise as a parameter
# 2) produce a new optimiser which adds noise
# 3) produce a new loss function

