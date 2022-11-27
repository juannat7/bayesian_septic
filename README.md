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

## Test Env
The below function will be added in test env
- [x] Bayesian Hierarchical Model
- [ ] Multilevel Bayesian Hierarchical Model
- [ ] Baseline Model
- [ ] Goodness of fit
- [ ] Inference

### Quick Start
1. **Set up the model parameter**: Create a json file in your configs/ folder that contains the parameter needed, we have an example for you to copy-paste
2. **Run in test environment with MakeFile**: If you are using the C-Compiler (if you are in mac / linux / Windows Git Bash you can do this) and have poetry, run: 
```
$ make train-bayes-water-model

====== Command Line Output ======
No dependencies to install or update
Enter Model Name (No need to include file extension): {model_example} # user input, should match your json file name
```
Check GNUMakeFile for more available options!
3. **Run in test environment without MakeFile**: If you do not want to use make command, simply do:
```
poetry run python main.py -m train_bayes_model -o $${model_name} -mo water_model
# or 
python main.py -m train_bayes_model -o $${model_name} -mo water_model
```
4. **Get your results**: The accuracy, VIF, and other results (including goodness of fit but not yet implemented) should be appear in "tmp/{your_json_file}_output.json". Check it out!

## Poetry Venv
Some of the packages depend on older version of python, numpy, and scipy. To avoid conflict with your local package, we suggest you use [poetry](https://python-poetry.org/docs/), which will create virtual environment to manage these conflicts. If pip works just fine, you can ignore this as well.
Here is the set-up tutorial:

1. **Install poetry**: 

For Linux, macOS, Windows (WSL)
```
curl -sSL https://install.python-poetry.org | python3 -
```

For Windows (Powershell)

```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

2. **Add poetry to your path**
The installer creates a poetry wrapper in a well-known, platform-specific directory:
- $HOME/.local/bin on Unix.
- %APPDATA%\Python\Scripts on Windows.
- $POETRY_HOME/bin if $POETRY_HOME is set.
If this directory is not present in your $PATH, you can add it in order to invoke Poetry as poetry.

3. **Run the program**
You can just run make ... program that is prefixed with "poetry run", it will install the package written in "poetry.lock" into a venv

4. **Add package**
If a package is used during the development, use: 
```
poetry add {package}
# or
poetry add {package}=={version number}
```

More [tutorial](https://python-poetry.org/docs/basic-usage/)