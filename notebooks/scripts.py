import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from tqdm import tqdm
import theano.tensor as tt
import scipy.stats as stats 
import geopandas as gpd

def evaluate_bayes(trace, model, y_actual, samples=500):
    """
    Evaluates the accuracy of hierarchical bayesian model
    
    parameters:
    -----------
    trace: xarray
        collection of diagnostic variables after fitting
    model: object
        the model definition
    y_actual: numpy array
        the actual septic status
    samples: int
        the number of sampling done per each data point
    
    returns:
    --------
    acc: float
        the accuracy of the model
    """
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=samples)
    y_pred = ppc['failures'].mean(axis=0)
    y_pred = [1 if i >0.5 else 0 for i in y_pred] # if p > 0.5, return 1, else return 0
    corr = np.sum(np.array(y_actual) == np.array(y_pred))
    
    acc = (corr / len(y_pred)) * 100
    return acc