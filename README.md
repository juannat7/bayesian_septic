# Inferring failure risk of on-site wastewater systems from physical and social factors

__Authors__: Juan Nathaniel (jn2808@columbia.edu), Sara Schwetschenau, Upmanu Lall

__Abstract__: Aging infrastructure and climate change present emerging challenges for clean water supply and reliable wastewater services for communities in the United States (US). In Georgia, for example, the failure rates of on-site wastewater systems (OWTS) have increased from 10% to 35% in the last two decades as the systems age. In this work, we develop a hierarchical Bayesian model to understand the different contributions of physical and social factors driving OWTS failures using a long-term collection of 201,000 Georgia’s OWTS inspection records. The out-of-sample validation accuracy of our hierarchical Bayesian model is 70% within Georgia, outperforming other machine learning models that do not consider the multiscale nature of the problem. Overall, we find counties that experience more extreme precipitation and are situated in steeper-sloped regions are significantly associated with increased failure risks. Uncertainties, meanwhile, are largely associated with counties experiencing more precipitation and have lower median housing value.

## Quickstart
The directory of this repository is divided as follows:
```
bayesian_septic
│   README.md    
└───data
└───docs
└───notebooks
└───src
    └───utils.py (utility functions)
    └───params.py (constants file)
    └───models.py (model definitions and implementations)
```
- `data` folder contains all processed data necessary to run the notebooks
- `notebooks` folder contains processes and analyses relevant to the research project
