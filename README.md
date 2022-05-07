# Hierarchical Bayesian Modeling of Septic System Failure
This project aims to apply a hierarchical bayesian modeling to infer the status of septic systems in Georgia. The hierarchy is on the basin level, and the input variables include the following:
- [x] Precipitation annual maxima
- [x] Distance to water bodies
- [x] Soil hydraulic conductivity
- [x] Median housing value
- [x] Analysis (Confusion, HDI-ROPE statistical significance, covariations)
- [x] Topography (flow acc, elevation)
- [x] Comparison with optimized baseline models (pooled bayesian, SVC, RF, XGB, GBDT)
- Goodness-of-fit tests:
    - [x] Implement WAIC, LOO, posterior variance checks
    - [x] Applied for 1-layer hierarchical bayesian models
    - [x] Applied for pooled models
    - [x] Diagnose relative SE (dSE) - measure of relative uncertainty across models 
    - [x] Applied for multi-level models
    - [ ] Check spatial independence/dependence

The directory of this repository is divided as follows:
```
bayesian_septic
│   README.md    
└───data
└───notebooks
│   │   utils.py (a collection of utility functions)
    |   params.py (a set of constants)
    |   models.py (a collection of model definitions and implementations)
```
- `data` folder contains all processed data necessary to run the notebooks
- `notebooks` folder contains processes and analyses relevant to the project