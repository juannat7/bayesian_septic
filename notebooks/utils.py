import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from tqdm import tqdm
import theano.tensor as tt
import scipy.stats as stats 
import geopandas as gpd
import seaborn as sns
import scipy
import rioxarray
import xarray as xr
import censusdata
from sklearn.metrics import confusion_matrix
from params import *

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
    return acc, y_pred

def read_data(file_dir, cols, is_balanced=True, train_frac=0.9, norm_scale='z', is_multilevel=False):
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
    df = pd.read_csv(file_dir)

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
    
    # get hydraulic information
    print('processing soil hydraulic data...')
    df['hydraulic_c'] = df['gSSURGO_GA'].str[0] # get the first value if a system has more than one soil type
    df['hydraulic_c'] = df['hydraulic_c'].replace(soil_dict)
    
    # get housing information
    print('acquiring housing information...')
    ga_acs = (censusdata.download(ga_acs_dict['type'], 
                                 ga_acs_dict['year'],
                                 censusdata.censusgeo([('state', ga_acs_dict['state']), ('county', '*')]),
                                 [ga_acs_dict['code']])
              .reset_index()
              .rename(columns={ga_acs_dict['code']: 'median_hse'}))
    
    df = match_acs_features(
            df, 
            ga_acs, 
            pri_key='tblSGA_Property.county_property',
            for_key='County',
            var_name='median_hse'
        )
    
    # get DEM and flow
    print('processing DEM and flow information...')
    ga_bdry = gpd.read_file('../../Data/ga_bdry')
    
    dem = rioxarray.open_rasterio('../../Data/hydrosheds/hyd_na_dem_15s/hyd_na_dem_15s.tif')
    flow = rioxarray.open_rasterio('../../Data/hydrosheds/na_acc_15s')
    
    ga_dem = dem.rio.clip(ga_bdry.geometry.values, ga_bdry.crs)
    ga_dem = xr.where(ga_dem > 30000, 0, ga_dem)
    ga_flow = flow.rio.clip(ga_bdry.geometry.values, ga_bdry.crs)
    ga_flow = xr.where(ga_flow < 0, 0, ga_flow)
    
    ## build spatial nearest neighbour lookup
    kdtree_dem, dem_coords = build_kdtree(ga_dem)
    kdtree_flow, flow_coords = build_kdtree(ga_flow)
    
    def get_spatial_nearest(row, xa, kdtree, coords):
        pt = [row["tblSGA_Property.gis_long_property"], row["tblSGA_Property.gis_lat_property"]]
        dist, index = kdtree.query(pt)
        coor = coords[index]
        vals = xa.sel(x=coor[0], y=coor[1])
        return vals.item()
    
    df['dem'] = df.apply(lambda x: get_spatial_nearest(x, xa=ga_dem, kdtree=kdtree_dem, coords=dem_coords), axis=1)
    df['flow'] = df.apply(lambda x: get_spatial_nearest(x, xa=ga_flow, kdtree=kdtree_flow, coords=flow_coords), axis=1)
    
    # keep only relevant columns
    all_cols = cols + ['HU_12_NAME', 'sewageSystem_enc']
    if is_multilevel:
        all_cols += ['HU_10_NAME']
        
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    
    # normalize
    for i, col in enumerate(cols):
        new_var = col + '_norm'
        df = normalize(df, col, new_var, scale=norm_scale)
        #df[new_var] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
 

    # separate septics based on their locations within a basin
    coords = dict()
    basin_idx, basins = pd.factorize(df['HU_12_NAME'])
    coords.update({'basin': basins, 'septic': np.arange(len(df))})
    
    if is_multilevel:
        catchment_idx, catchments = pd.factorize(df['HU_10_NAME'])
        coords.update({'catchment': catchments})
        return df, basin_idx, catchment_idx, coords
    
    return df, basin_idx, basins, coords # legacy code to ensure other notebooks work

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
        match_idx = acs_df.index[acs_df[for_key]==septic_cty][0]
        median_hse = acs_df[var_name][match_idx]
        septic_df.loc[i,var_name] = median_hse
        
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
        df[var_norm] = (df[var] - df[var].min()) / (df[var].max() - df[var].min())
    else:
        df[var_norm] = (df[var] - df[var].mean()) / df[var].std()
    
    return df

def plot_confusion(y, y_pred):
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
    cf = confusion_matrix(y, y_pred, labels=None, sample_weight=None, normalize='true')
    ax = sns.heatmap(cf, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix of Septic System Status Forecast\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['Repair','No-Repair'])
    ax.yaxis.set_ticklabels(['Repair','No-Repair'])

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
            coords.append([x,y])
            
    return scipy.spatial.KDTree(coords), coords