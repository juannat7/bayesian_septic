"""
Contains all the definition and implementation for the following bayesian hierarchical models:
(each successive model includes the ones prior)

    1. Water model (distance to water bodies and precipitation)
    2. Topography model (elevation and hydraulic conductivity)
    3. Socio-economic model (housing value)
    4. Full model (all variables)
"""

import pymc3 as pm
from src.utils import *
from src.params import *

df, basin_idx, catchment_idx, coords = read_data(file_dir='../data/hierarchical_septics_v4.csv',
                                                 cols=['ppt_2021', 'hydraulic_c','median_hse', 'dem'],
                                                 is_balanced=True, norm_scale='z',
                                                 is_multilevel=True)

tune = 1000
rs = 100

###########################################
############ 1-level Bayesian #############
###########################################
print('Fitting 1-layer hierarchical Bayesian models...')
# # water model
# with pm.Model(coords=coords) as water_model:
#     # constant data: basin information and variables
#     basin = pm.Data("basin", basin_idx, dims="septic")
#     water_d = pm.Data("water_d", df.water_dist_norm.values, dims="septic")
#     ppt_d = pm.Data("ppt_d", df.ppt_2013_norm.values, dims="septic")

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
#     ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
#     mu_c = pm.Normal("mu_c", mu=0, sigma=10)
#     sigma_c = pm.HalfNormal("sigma_c", 10)

#     # septic-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
#     ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig, dims="basin")
#     c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin_idx] 
#                                     + wtr_dist[basin_idx] * water_d 
#                                     + ppt[basin_idx] * ppt_d
#                                    )

#     # likelihood of observed data
#     water_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
#     # fitting using NUTS sampler
#     water_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# # distance to water bodies
# with pm.Model(coords=coords) as dist_model:
#     print('fitting water distance model...')
#     # constant data: basin information and variables
#     basin = pm.Data("basin", basin_idx, dims="septic")
#     water_d = pm.Data("water_d", df.water_dist_norm.values, dims="septic")

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     mu_c = pm.Normal("mu_c", mu=0, sigma=10)
#     sigma_c = pm.HalfNormal("sigma_c", 10)

#     # septic-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta, dims="basin")
#     c = pm.Normal("c", mu=mu_c, sigma=sigma_c, dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin_idx] 
#                                     + wtr_dist[basin_idx] * water_d 
#                                    )

#     # likelihood of observed data
#     dist_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
#     # fitting using NUTS sampler
#     dist_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# precipitation
with pm.Model(coords=coords) as ppt_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    # water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values, dims='septic')

    # global model parameters
    # wtr_beta = pm.HalfNormal('wtr_beta', sigma=10)
    ppt_mu = pm.Normal('ppt_mu', mu=0, sigma=10)
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    # wtr_dist = pm.Exponential('wtr_dist', lam=wtr_beta, dims='basin')
    ppt = pm.Normal('ppt', mu=ppt_mu, sigma=ppt_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    # + wtr_dist[basin] * water_d 
                                    + ppt[basin] * ppt_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    ppt_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# soil model
with pm.Model(coords=coords) as soil_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')

    # global model parameters
    hydr_mu = pm.Normal('hydr_mu', mu=0, sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + hydr[basin] * hydr_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    soil_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# topo model
with pm.Model(coords=coords) as topo_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

    # global model parameters
    dem_beta = pm.HalfNormal('dem_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    dem = pm.Exponential('dem', lam=dem_beta, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + dem[basin] * dem_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    topo_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# socio-economic model
with pm.Model(coords=coords) as socio_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')

    # global model parameters
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    hse = pm.Normal('hse', mu=0, sigma=hse_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + hse[basin] * hse_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    socio_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)
    
# full model
with pm.Model(coords=coords) as full_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    # water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')
    dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

    # global model parameters
    # wtr_beta = pm.HalfNormal('wtr_beta', sigma=10)
    ppt_mu = pm.Normal('ppt_mu', mu=0, sigma=10)
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    hydr_mu = pm.Normal('hydr_mu', mu=0, sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    dem_beta = pm.HalfNormal('dem_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    # wtr_dist = pm.Exponential('wtr_dist', lam=wtr_beta, dims='basin')
    ppt = pm.Normal('ppt', mu=ppt_mu, sigma=ppt_sig, dims='basin')
    hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig, dims='basin')
    hse = pm.Normal('hse', mu=0, sigma=hse_sig, dims='basin')
    dem = pm.Exponential('dem', lam=dem_beta, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    # + wtr_dist[basin] * water_d 
                                    + ppt[basin] * ppt_d
                                    + hydr[basin] * hydr_d
                                    + hse[basin] * hse_d
                                    + dem[basin] * dem_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    full_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

traces_dict = dict()
traces_dict.update({'L1_Precip': ppt_trace,
                    'L1_Soil': soil_trace,
                    'L1_Topo': topo_trace, 
                    'L1_Socio': socio_trace, 
                    'L1_Full': full_trace
                   })

# opt_traces_dict = dict()
# opt_traces_dict.update({'L1_Distance': dist_trace, 
#                         'L1_Precip': ppt_trace,
#                         'L1_Water': water_trace, 
#                         'L1_Soil': soil_trace, 
#                         'L1_Socio': socio_trace, 
#                         'L1_Topo': topo_trace
#                        })

###########################################
############# Pooled Bayesian #############
###########################################
print('Fitting pooled Bayesian models...')
# # water model
# with pm.Model() as water_pooled_model:
#     # constant data: basin information and variables
#     basin = pm.Data("basin", basin_idx)
#     water_d = pm.Data("water_d", df.water_dist_norm.values)
#     ppt_d = pm.Data("ppt_d", df.ppt_2013_norm.values)

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
#     ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
#     mu_c = pm.Normal("mu_c", mu=0, sigma=10)
#     sigma_c = pm.HalfNormal("sigma_c", 10)

#     # septic-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta)
#     ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig)
#     c = pm.Normal("c", mu=mu_c, sigma=sigma_c)
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c
#                                     + wtr_dist * water_d 
#                                     + ppt * ppt_d
#                                    )

#     # likelihood of observed data
#     water_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
#     # fitting using NUTS sampler
#     water_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# # distance to water bodies
# with pm.Model(coords=coords) as dist_model:
#     print('fitting water distance model...')
#     # constant data: basin information and variables
#     basin = pm.Data("basin", basin_idx)
#     water_d = pm.Data("water_d", df.water_dist_norm.values)

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     mu_c = pm.Normal("mu_c", mu=0, sigma=10)
#     sigma_c = pm.HalfNormal("sigma_c", 10)

#     # septic-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta)
#     c = pm.Normal("c", mu=mu_c, sigma=sigma_c)
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c
#                                     + wtr_dist * water_d 
#                                    )

#     # likelihood of observed data
#     dist_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli("failures", failure_theta, observed=df["sewageSystem_enc"])
    
#     # fitting using NUTS sampler
#     dist_trace = pm.sample(500, tune=200, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# precipitation
with pm.Model(coords=coords) as ppt_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values)

    # global model parameters
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    # wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta)
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c
                                    + ppt * ppt_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    ppt_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# soil model
with pm.Model(coords=coords) as soil_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values)

    # global model parameters
    hydr_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c
                                    + hydr * hydr_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    soil_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# topo model
with pm.Model(coords=coords) as topo_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    dem_d = pm.Data('dem_d', df.dem_norm.values)

    # global model parameters
    dem_beta = pm.HalfNormal('dem_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    dem = pm.Exponential('dem', lam=dem_beta)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c
                                    + dem * dem_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    topo_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# socio-economic model
with pm.Model(coords=coords) as socio_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    hse_d = pm.Data('hse_d', df.median_hse_norm.values)

    # global model parameters
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    hse = pm.Normal('hse', mu=0, sigma=hse_sig)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c
                                    + hse * hse_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    socio_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)
    
# full model
with pm.Model(coords=coords) as full_model:
    print('fitting full pooled Bayesian model...')
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    # water_d = pm.Data('water_d', df.water_dist_norm.values)
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values)
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values)
    hse_d = pm.Data('hse_d', df.median_hse_norm.values)
    dem_d = pm.Data('dem_d', df.dem_norm.values)

    # global model parameters
    # wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    hydr_mu = pm.Normal("hydr_mu", mu=0, sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    dem_beta = pm.HalfNormal('dem_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    # wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta)
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig)
    hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig)
    hse = pm.Normal('hse', mu=0, sigma=hse_sig)
    dem = pm.Exponential('dem', lam=dem_beta)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c
                                    # + wtr_dist * water_d 
                                    + ppt * ppt_d
                                    + hydr * hydr_d
                                    + hse * hse_d
                                    + dem * dem_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    full_trace = pm.sample(100, tune=tune, cores=3, return_inferencedata=True, target_accept=0.99, random_seed=rs)

traces_dict.update({'L0_Precip': ppt_trace,
                    'L0_Soil': soil_trace,
                    'L0_Topo': topo_trace, 
                    'L0_Socio': socio_trace, 
                    'L0_Full': full_trace})

# opt_traces_dict.update({'L0_Distance': dist_trace, 
#                         'L0_Precip': ppt_trace,
#                         'L0_Water': water_trace, 
#                         'L0_Soil': soil_trace, 
#                         'L0_Socio': socio_trace, 
#                         'L0_Topo': topo_trace})

###########################################
############ 2-level Bayesian #############
###########################################
# print('Fitting 2-level Bayesian models...')
# with pm.Model(coords=coords) as water_model:
#     # constant data: basin information and variables
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
#     ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
#     ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     wtr_beta_c = pm.HalfNormal("wtr_beta_c", sigma=wtr_beta, dims='catchment')
#     ppt_mu_c = pm.Normal("ppt_mu_c", mu=ppt_mu, sigma=10, dims='catchment')
#     ppt_sig_c = pm.HalfNormal("ppt_sig_c", sigma=ppt_sig, dims='catchment')
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # basin-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta_c.mean(), dims="basin")
#     ppt = pm.Normal("ppt", mu=ppt_mu_c.mean(), sigma=ppt_sig_c.mean(), dims="basin")
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + wtr_dist[basin] * water_d 
#                                     + ppt[basin] * ppt_d
#                                    )

#     # likelihood of observed data
#     water_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     water_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)
    
# #distance to water bodies
# with pm.Model(coords=coords) as dist_model:
#     print('fitting water distance model...')
#     # constant data: basin information and variables
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     wtr_beta_c = pm.HalfNormal("wtr_beta_c", sigma=wtr_beta, dims='catchment')
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # basin-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta_c.mean(), dims="basin")
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + wtr_dist[basin] * water_d 
#                                    )

#     # likelihood of observed data
#     dist_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     dist_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# # precipitation
# with pm.Model(coords=coords) as ppt_model:
#     print('fitting precipitation model...')
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')

#     # global model parameters
#     ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
#     ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     ppt_mu_c = pm.Normal("ppt_mu_c", mu=ppt_mu, sigma=10, dims='catchment')
#     ppt_sig_c = pm.HalfNormal("ppt_sig_c", sigma=ppt_sig, dims='catchment')
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # basin-specific model parameters
#     ppt = pm.Normal("ppt", mu=ppt_mu_c.mean(), sigma=ppt_sig_c.mean(), dims="basin")
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + ppt[basin] * ppt_d
#                                    )

#     # likelihood of observed data
#     ppt_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     ppt_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)
    
# # topo model
# with pm.Model(coords=coords) as topo_model:
#     # constant data: basin information and variables
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
#     dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

#     # global model parameters
#     hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
#     dem_beta = pm.HalfNormal('dem_beta', sigma=5)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     hydr_sig_c = pm.HalfNormal('hydr_sig_c', sigma=hydr_sig, dims='catchment')
#     dem_beta_c = pm.HalfNormal('dem_beta_c', sigma=dem_beta)
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # septic-specific model parameters
#     hydr = pm.Uniform('hydr', lower=-2, upper=hydr_sig_c.mean(), dims='basin')
#     dem = pm.Exponential('dem', lam=dem_beta_c.mean(), dims='basin')
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + hydr[basin] * hydr_d
#                                     + dem[basin] * dem_d
#                                    )

#     # likelihood of observed data
#     priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     topo_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)
    
# # socio model
# with pm.Model(coords=coords) as socio_model:
#     # constant data: basin information and variables
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')

#     # global model parameters
#     hse_sig = pm.HalfNormal('hse_sig', sigma=5)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     hse_sig_c = pm.HalfNormal('hse_sig_c', sigma=hse_sig, dims='catchment')
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # septic-specific model parameters
#     hse = pm.Normal('hse', mu=0, sigma=hse_sig_c.mean(), dims='basin')
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + hse[basin] * hse_d
#                                    )

#     # likelihood of observed data
#     socio_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     socio_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.95, random_seed=rs)
    
# # full model
# with pm.Model(coords=coords) as full_model:
#     # constant data: basin information and variables
#     basin = pm.Data('basin', basin_idx, dims='septic')
#     catchment = pm.Data('catchment', catchment_idx, dims='septic')
#     water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
#     ppt_d = pm.Data('ppt_d', df.ppt_2013_norm.values, dims='septic')
#     hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
#     hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')
#     dem_d = pm.Data('dem_d', df.dem_norm.values, dims='septic')

#     # global model parameters
#     wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
#     ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
#     ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
#     hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
#     hse_sig = pm.HalfNormal('hse_sig', sigma=5)
#     dem_beta = pm.HalfNormal('dem_beta', sigma=5)
#     mu_inter = pm.Normal('mu_inter', mu=0, sigma=10)
#     sigma_inter = pm.HalfNormal('sigma_inter', sigma=10)
    
#     # catchment parameters
#     wtr_beta_c = pm.HalfNormal("wtr_beta_c", sigma=wtr_beta, dims='catchment')
#     ppt_mu_c = pm.Normal("ppt_mu_c", mu=ppt_mu, sigma=10, dims='catchment')
#     ppt_sig_c = pm.HalfNormal("ppt_sig_c", sigma=ppt_sig, dims='catchment')
#     hydr_sig_c = pm.HalfNormal('hydr_sig_c', sigma=hydr_sig, dims='catchment')
#     hse_sig_c = pm.HalfNormal('hse_sig_c', sigma=hse_sig, dims='catchment')
#     dem_beta_c = pm.HalfNormal('dem_beta_c', sigma=dem_beta)
#     mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=10, dims='catchment')
#     sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

#     # septic-specific model parameters
#     wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta_c.mean(), dims="basin")
#     ppt = pm.Normal("ppt", mu=ppt_mu_c.mean(), sigma=ppt_sig_c.mean(), dims="basin")
#     hydr = pm.Uniform('hydr', lower=-2, upper=hydr_sig_c.mean(), dims='basin')
#     hse = pm.Normal('hse', mu=0, sigma=hse_sig_c.mean(), dims='basin')
#     dem = pm.Exponential('dem', lam=dem_beta_c.mean(), dims='basin')
#     c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
    
#     # hierarchical bayesian formula
#     failure_theta = pm.math.sigmoid(c[basin] 
#                                     + wtr_dist[basin] * water_d 
#                                     + ppt[basin] * ppt_d
#                                     + hydr[basin] * hydr_d
#                                     + hse[basin] * hse_d
#                                     + dem[basin] * dem_d
#                                    )

#     # likelihood of observed data
#     full_priors = pm.sample_prior_predictive(samples=500)
#     failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
#     # fitting using NUTS sampler
#     full_trace = pm.sample(500, tune=tune, cores=4, return_inferencedata=True, target_accept=0.99, random_seed=rs)

# traces_dict.update({'L2_Water': water_trace, 
#                     'L2_Topo': topo_trace, 
#                     'L2_Socio': socio_trace, 
#                     'L2_Full': full_trace
#                    })

# opt_traces_dict.update({'L2_Distance': dist_trace, 
#                         'L2_Precip': ppt_trace,
#                         'L2_Water': water_trace, 
#                         'L2_Soil': soil_trace, 
#                         'L2_Socio': socio_trace, 
#                         'L2_Topo': topo_trace
#                        })