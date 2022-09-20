# Hierarchical Bayesian Modeling of Septic System Failure
This project aims to apply a hierarchical bayesian modeling to infer the status of septic systems in Georgia. The hierarchy is either on (1) a single level (ie. _sub_-basin) or (2) two-level (ie. basin --> _sub_-basin). Baselines include pooled Bayesian and several ML models (eg. RF, boosting trees, SVC). 

## Quickstart
1. Create a new environment using Anaconda: `conda create -n septic pymc3`
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks in sequence, although each notebook is self-contained

## Checklists
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
- [x] Implement multi-run logic for accuracy checks and reproducibility
- Analysis:
    - [ ] Risk map

The directory of this repository is divided as follows:
```
bayesian_septic
│   README.md    
└───data
└───docs
└───notebooks
└───src
    └───utils.py (utility functions)
    └───params.py (constants)
    └───models.py (model definitions and implementations)
```
- `data` folder contains all processed data necessary to run the notebooks
- `notebooks` folder contains processes and analyses relevant to the research project