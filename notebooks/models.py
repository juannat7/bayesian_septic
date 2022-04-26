"""
Contains all the definition and implementation for the following bayesian hierarchical models:
(each successive model includes the ones prior)

    1. Water model (distance to water bodies and precipitation)
    2. Soil model (hydraulic conductivity)
    3. Socio-economic model (housing value)
    4. Topography model (elevation and flow accumulation)
"""

import pymc3 as pm
from utils import *
from params import *

df, basin_idx, basins, coords = read_data(file_dir='../data/hierarchical_septics_v2.csv',
        cols=['ppt_2013', 'water_dist', 'hydraulic_c','median_hse', 'dem', 'flow'], is_balanced=True)

# water model
with pm.Model(coords=coords) as water_model:
    print('fitting water model...')
    # constant data: basin information and variables
    basin = pm.Data("basin", basin_idx, dims="septic")
    water_d = pm.Data("water_d", df.water_dist_norm.values, dims="septic")
    ppt_d = pm.Data("ppt_d", df.ppt_2013_norm.values, dims="septic")

    # global model parameters
    wtr_alpha = pm.HalfNormal("wtr_alpha", sigma=1.)
    wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    ppt_mu = pm.HalfNormal("ppt_mu", sigma=0.5)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    mu_c = pm.Normal("mu_c", mu=0, sigma=10)
    sigma_c = pm.HalfNormal("sigma_c", 10)

    # septic-specific model parameters
    wtr_dist = pm.Gamma("wtr_dist", alpha=wtr_alpha, beta=wtr_beta, dims="basin")
    ppt = pm.HalfNormal("ppt", sigma=ppt_sig, dims="basin")
    c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
    # fitting using NUTS sampler
    water_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)

# soil model
with pm.Model(coords=coords) as soil_model:
    print('fitting soil model...')
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')

    # global model parameters
    wtr_alpha = pm.HalfNormal('wtr_alpha', sigma=1.)
    wtr_beta = pm.HalfNormal('wtr_beta', sigma=5)
    ppt_mu = pm.HalfNormal('ppt_mu', sigma=10)
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Gamma('wtr_dist', alpha=wtr_alpha, beta=wtr_beta, dims='basin')
    ppt = pm.HalfNormal('ppt', sigma=ppt_sig, dims='basin')
    hydr = pm.Uniform('hydr', lower=0, upper=hydr_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                    + hydr[basin_idx] * hydr_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    soil_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)
    
# socio-economic model
with pm.Model(coords=coords) as socio_model:
    print('fitting socio model...')
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')

    # global model parameters
    wtr_alpha = pm.HalfNormal('wtr_alpha', sigma=1.)
    wtr_beta = pm.HalfNormal('wtr_beta', sigma=5)
    ppt_mu = pm.HalfNormal('ppt_mu', sigma=10)
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=5)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Gamma('wtr_dist', alpha=wtr_alpha, beta=wtr_beta, dims='basin')
    ppt = pm.HalfNormal('ppt', sigma=ppt_sig, dims='basin')
    hydr = pm.Uniform('hydr', lower=0, upper=hydr_sig, dims='basin')
    hse = pm.HalfNormal('hse', sigma=hse_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                    + hydr[basin_idx] * hydr_d
                                    + hse[basin_idx] * hse_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    socio_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)
    
# topography model
with pm.Model(coords=coords) as topo_model:
    print('fitting topo model...')
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')
    flow_d = pm.Data('flow_d', df.flow_norm.values, dims='septic')
    dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

    # global model parameters
    wtr_alpha = pm.HalfNormal('wtr_alpha', sigma=1.)
    wtr_beta = pm.HalfNormal('wtr_beta', sigma=5)
    ppt_mu = pm.HalfNormal('ppt_mu', sigma=10)
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=5)
    flow_alpha = pm.HalfNormal('flow_alpha', sigma=1.)
    flow_beta = pm.HalfNormal('flow_beta', sigma=5)
    dem_alpha = pm.HalfNormal('dem_alpha', sigma=1.)
    dem_beta = pm.HalfNormal('dem_beta', sigma=5)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Gamma('wtr_dist', alpha=wtr_alpha, beta=wtr_beta, dims='basin')
    ppt = pm.HalfNormal('ppt', sigma=ppt_sig, dims='basin')
    hydr = pm.Uniform('hydr', lower=0, upper=hydr_sig, dims='basin')
    hse = pm.HalfNormal('hse', sigma=hse_sig, dims='basin')
    flow = pm.Gamma('flow', alpha=flow_alpha, beta=flow_beta, dims='basin')
    dem = pm.Gamma('dem', alpha=dem_alpha, beta=dem_beta, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + wtr_dist[basin] * water_d 
                                    + ppt[basin] * ppt_d
                                    + hydr[basin] * hydr_d
                                    + hse[basin] * hse_d
                                    + flow[basin] * flow_d
                                    + dem[basin] * dem_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    topo_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)
    
traces_dict = dict()
traces_dict.update({water_model: water_trace, 
                    soil_model: soil_trace, 
                    socio_model: socio_trace, 
                    topo_model: topo_trace})