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
        cols=['ppt_2013', 'water_dist', 'hydraulic_c','median_hse', 'dem', 'flow'], is_balanced=True, norm_scale='z')

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
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    mu_c = pm.Normal("mu_c", mu=0, sigma=10)
    sigma_c = pm.HalfNormal("sigma_c", 10)

    # septic-specific model parameters
    wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
    c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                   )

    # likelihood of observed data
    water_priors = pm.sample_prior_predictive(samples=500)
    failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
    # fitting using NUTS sampler
    water_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)

# distance to water bodies
with pm.Model(coords=coords) as dist_model:
    print('fitting water distance model...')
    # constant data: basin information and variables
    basin = pm.Data("basin", basin_idx, dims="septic")
    water_d = pm.Data("water_d", df.water_dist_norm.values, dims="septic")

    # global model parameters
    wtr_alpha = pm.HalfNormal("wtr_alpha", sigma=1.)
    wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    mu_c = pm.Normal("mu_c", mu=0, sigma=10)
    sigma_c = pm.HalfNormal("sigma_c", 10)

    # septic-specific model parameters
    wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
    c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                   )

    # likelihood of observed data
    dist_priors = pm.sample_prior_predictive(samples=500)
    failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
    # fitting using NUTS sampler
    dist_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)

# precipitation
with pm.Model(coords=coords) as ppt_model:
    print('fitting precipitation model...')
    # constant data: basin information and variables
    basin = pm.Data("basin", basin_idx, dims="septic")
    ppt_d = pm.Data("ppt_d", df.ppt_2013_norm.values, dims="septic")

    # global model parameters
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    mu_c = pm.Normal("mu_c", mu=0, sigma=10)
    sigma_c = pm.HalfNormal("sigma_c", 10)

    # septic-specific model parameters
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
    c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + ppt[basin_idx] * ppt_d
                                   )

    # likelihood of observed data
    ppt_priors = pm.sample_prior_predictive(samples=500)
    failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
    # fitting using NUTS sampler
    ppt_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)
    
# soil model
with pm.Model(coords=coords) as soil_model:
    print('fitting soil model...')
    # constant data: basin information and variables
    basin = pm.Data("basin", basin_idx, dims="septic")
    water_d = pm.Data("water_d", df.water_dist_norm.values, dims="septic")
    ppt_d = pm.Data("ppt_d", df.ppt_2013_norm.values, dims="septic")
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')

    # global model parameters
    wtr_alpha = pm.HalfNormal("wtr_alpha", sigma=1.)
    wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
    hydr = pm.Uniform('hydr', lower=-2, upper=hydr_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                    + hydr[basin_idx] * hydr_d
                                   )

    # likelihood of observed data
    soil_priors = pm.sample_prior_predictive(samples=500)
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
    wtr_alpha = pm.HalfNormal("wtr_alpha", sigma=1.)
    wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=5)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
    hydr = pm.Uniform('hydr', lower=-2, upper=hydr_sig, dims='basin')
    hse = pm.Normal('hse', mu=0, sigma=hse_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin_idx] 
                                    + wtr_dist[basin_idx] * water_d 
                                    + ppt[basin_idx] * ppt_d
                                    + hydr[basin_idx] * hydr_d
                                    + hse[basin_idx] * hse_d
                                   )

    # likelihood of observed data
    socio_priors = pm.sample_prior_predictive(samples=500)
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
    dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

    # global model parameters
    wtr_alpha = pm.HalfNormal("wtr_alpha", sigma=1.)
    wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=5)
    dem_alpha = pm.HalfNormal('dem_alpha', sigma=1.)
    dem_beta = pm.HalfNormal('dem_beta', sigma=5)
    mu_c = pm.HalfNormal('mu_c', sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
    hydr = pm.Uniform('hydr', lower=-2, upper=hydr_sig, dims='basin')
    hse = pm.Normal('hse', mu=0, sigma=hse_sig, dims='basin')
    dem = pm.Exponential('dem', lam=dem_beta, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + wtr_dist[basin] * water_d 
                                    + ppt[basin] * ppt_d
                                    + hydr[basin] * hydr_d
                                    + hse[basin] * hse_d
                                    #+ flow[basin] * flow_d
                                    + dem[basin] * dem_d
                                   )

    # likelihood of observed data
    topo_priors = pm.sample_prior_predictive(samples=500)
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    topo_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99)
    
traces_dict = dict()
traces_dict.update({'Water': water_trace, 
                    'Soil': soil_trace, 
                    'Socio': socio_trace, 
                    'Topo': topo_trace})

opt_traces_dict = dict()
opt_traces_dict.update({'Distance': dist_trace, 
                        'Precip': ppt_trace,
                        'Water': water_trace, 
                        'Soil': soil_trace, 
                        'Socio': socio_trace, 
                        'Topo': topo_trace})