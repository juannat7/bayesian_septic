
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_context("paper")


def evaluate_bayes(trace, model, y_actual):
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
    samples: int (deprecated)
        the number of sampling done per each data point
    
    returns:
    --------
    acc: float
        the accuracy of the model
    """
    ppc = pm.sample_posterior_predictive(trace, model=model)
    y_pred = ppc['posterior_predictive']['failures'].mean(axis=0).mean(axis=0) # averaging across chains and draws
    # if p > 0.5, return 1, else return 0
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    corr = np.sum(np.array(y_actual) == np.array(y_pred))

    acc = (corr / len(y_pred)) * 100
    return acc, y_pred

def read_data(file_dir, cols, is_balanced=True, train_frac=0.9, norm_scale='z', hierarchy_type='basin', is_multilevel=False):
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
    train_frac: float
        the fraction for training split
    norm_scale: str
        the scaling for normalization, one of ['z', 'min_max']
    hierarchy_type: str
        the hierarchy type, one of ['basin', 'county']
    is_multilevel: boolean
        to indicate whether the coordinates returned include both basin and sub-basin (if True)
    
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
    assert hierarchy_type in ['basin', 'county']
    df = pd.read_csv(file_dir)

    # encode categorical sewage system
    enc, _ = pd.factorize(df['sewageSystem'])
    df['sewageSystem_enc'] = enc
    df.loc[df['sewageSystem_enc'] == 1, 'sewageSystem_enc'] = 1  # need repair
    df.loc[(df['sewageSystem_enc'] == 0) | (df['sewageSystem_enc'] == 2), 'sewageSystem_enc'] = 0  # new + addition

    # get balanced class (septics needing repair are not as many)
    if is_balanced:
        num = len(df[df['sewageSystem_enc'] == 1].values)
        print(
            f'balancing...\nrepairs: {num/len(df)*100}%, non-repairs: {(len(df) - num)/len(df)*100}%')

        # split equally
        idx = df[df['sewageSystem_enc'] == 0]
        df = pd.concat((df.sample(n=len(idx), random_state=42),
                       df[df['sewageSystem_enc'] == 1]))

    # keep only relevant columns
    if hierarchy_type == 'basin':
        all_cols = cols + ['HU_10_NAME', 'sewageSystem_enc']
        if is_multilevel: all_cols += ['HU_12_NAME']
    else:
        all_cols = cols + ['tblSGA_Property.county_property', 'sewageSystem_enc']
        
    df = df.dropna(subset=all_cols).reset_index(drop=True)

    # normalize
    for i, col in enumerate(cols):
        new_var = col + '_norm'
        df = normalize(df, col, new_var, scale=norm_scale)
        #df[new_var] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
 

    # separate septics based on their locations within a basin OR county
    coords = dict()
    if hierarchy_type == 'basin':
        basin_idx, basins = pd.factorize(df['HU_10_NAME'])
    else:
        basin_idx, basins = pd.factorize(df['tblSGA_Property.county_property'])
    coords.update({'basin': basins, 'septic': np.arange(len(df))})

    if is_multilevel:
        # For now only valid for hierarchy_type == 'basin'
        catchment_idx, catchments = pd.factorize(df['HU_12_NAME'])
        coords.update({'catchment': catchments})
        return df, basin_idx, catchment_idx, coords

    # legacy code to ensure other notebooks work
    return df, basin_idx, basins, coords


def match_acs_features(septic_df, acs_df, pri_key, for_key, var_name):
    # extract county name with columns: ['index', 'County']
    for i, row in acs_df.iterrows():
        county = (str(row['index'])
                  .split(',')[0]
                  .split(for_key)[0]
                  .strip()
                  .upper())

        acs_df.loc[i, for_key] = county

    # match with our septic systems with columns ['tblSGA_Property.county_property']
    for i, row in septic_df.iterrows():
        septic_cty = str(row[pri_key])
        match_idx = acs_df.index[acs_df[for_key] == septic_cty][0]
        median_hse = acs_df[var_name][match_idx]
        septic_df.loc[i, var_name] = median_hse

    return septic_df


def normalize(df, var, var_norm, scale='z'):
    """
    Normalize variables
    
    parameters:
    -----------
    df: DataFrame
        DataFrame where the variables are found
    var: str
        the name of the variable where normalization is performed
    var_norm: str
        the new normalized variable name
    scale: str
        the scale of normalization ['z', 'min_max']
    
    returns:
    --------
    df: DataFrame
        the updated DataFrame with the normalized variable
    """

    if scale == 'min_max':
        df[var_norm] = (df[var] - df[var].min()) / \
            (df[var].max() - df[var].min())
    else:
        df[var_norm] = (df[var] - df[var].mean()) / df[var].std()

    return df


def plot_confusion(y, y_pred, title, savedir=None):
    """
    Plot confusion matrix given the true y and the predicted y
    
    parameters:
    -----------
    y: ndarray
        numpy array for the actual y
    y_pred: ndarray
        numpy array for the predicted y
    
    returns:
    --------
    None
    """
    cf = confusion_matrix(y, y_pred, labels=None,
                          sample_weight=None, normalize='true')
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cf, annot=True, cmap='Blues',
                annot_kws={'fontsize': 24}, ax=ax)

    ax.set_title(title)
    ax.set_xlabel('\nPredicted Values', fontsize=20)
    ax.set_ylabel('Actual Values ', fontsize=20)
    ax.xaxis.set_ticklabels(['Failing', 'Non-failing'])
    ax.yaxis.set_ticklabels(['Failing', 'Non-failing'])
    ax.tick_params(labelsize=18)

    if savedir is not None:
        f.savefig(savedir, dpi=300, bbox_inches='tight')


def build_kdtree(xa):
    """
    Build fast spatial lookup tree
    
    parameters:
    -----------
    xa: xarray
        xarray data for the variable of interest containing spatial information (ie. x,y coordinates)
    
    returns:
    --------
    KDTree: object
        spatial lookup object
    coords: 2D array
        corresponding x,y coordinates for each of the data point
    """
    coords = []
    for x in xa.x.values:
        for y in xa.y.values:
            coords.append([x, y])

    return scipy.spatial.KDTree(coords), coords
