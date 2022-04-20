# Hierarchical Bayesian Modeling of Septic System Failure
This project aims to apply a hierarchical bayesian modeling to infer the status of septic systems in Georgia. The hierarchy is on the basin level, and the input variables include the following:
- [x] Precipitation annual maxima
- [x] Distance to water bodies
- [x] Soil hydraulic conductivity
- [x] Median housing value
- [x] Analysis (Confusion, HDI-ROPE statistical significance, covariations)
- [x] Topography (flow acc, elevation)
- [x] Comparison with baseline model (with hyperparameter optimization)
- [ ] Basin-level generalization

The directory of this repository is divided as follows:
```
bayesian_septic
│   README.md    
└───data
└───notebooks
│   │   utils.py (utility functions)
    |   params.py (constants)
```
- `data` folder contains all processed data necessary to run the notebooks
- `notebooks` folder contains processes and analyses relevant to the project