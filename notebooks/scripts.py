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

def read_data(file_dir, cols, is_balanced=True):
    """
    Read the initial CSV file used for modeling
    
    parameters:
    -----------
    file_dir: str
        directory of the CSV file
    cols: list of str
        the columns of interest where null values will be dropped
    is_balanced: boolean
        balances class-of-interest; to avoid class imbalance
    
    returns:
    --------
    df: dataframe
        the loaded dataframe
    basin_idx: list of index
        the indices of basins used for hierarchical modeling
    basins: list of str
        the names of basins used for hierarchical modeling
    coords: dict
        the dictionary mapping between basin_idx and basins
    """
    df = pd.read_csv(file_dir)
    
    # keep only relevant columns
    df = df.dropna(subset=cols).reset_index(drop=True)
    
    # normalize
    for i, col in enumerate(cols):
        try:
            new_var = col + '_norm'
            df[new_var] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        except:
            continue

    # encode categorical sewage system
    enc, _ = pd.factorize(df['sewageSystem'])
    df['sewageSystem_enc'] = enc
    df.loc[df['sewageSystem_enc'] == 0, 'sewageSystem_enc'] = 0 # need repair
    df.loc[(df['sewageSystem_enc'] == 1) | (df['sewageSystem_enc'] == 2), 'sewageSystem_enc'] = 1 # new + addition

    # get balanced class (septics needing repair are not as many)
    if is_balanced:
        num_repair = len(df[df['sewageSystem_enc'] == 0].values)
        print(f'balancing...\nrepairs: {num_repair/len(df)*100}%, non-repairs: {(len(df) - num_repair)/len(df)*100}%')

        # split equally
        df = pd.concat((df[df['sewageSystem_enc'] == 0], df[df['sewageSystem_enc'] == 1][:num_repair]))

    # separate septics based on their locations within a basin
    basin_idx, basins = pd.factorize(df['HU_12_NAME'])
    coords = {'basin': basins, 'septic': np.arange(len(df))}

    return df, basin_idx, basins, coords