"""
Contains all the definition and implementation for the following bayesian hierarchical models:
(each successive model includes the ones prior)

    1. Water model (distance to water bodies and precipitation)
    2. Topography model (elevation and hydraulic conductivity)
    3. Socio-economic model (housing value)
    4. Full model (all variables)
"""

import pymc as pm
from src.utils import read_data


df, basin_idx, basins, coords = read_data(
        file_dir='../data/hierarchical_septics_v7.csv',
        cols=['ppt_2021', 'hydraulic_c','median_hse', 'slope', 'age_median', 'income_median'], 
        is_balanced=True, hierarchy_type='county')

tune = 100
rs = 100

###########################################
############ 1-level Bayesian #############
###########################################
print('Fitting 1-layer hierarchical Bayesian models...')

# precipitation
with pm.Model(coords=coords) as ppt_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values, dims='septic')
    income_d = pm.Data('income_d', df.income_median_norm.values, dims='septic')

    # global model parameters
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # similarity parameters
    similarity_c = pm.Normal('similarity_c', mu=0, sigma=10)
    income_mu = pm.Normal('income_mu', mu=0, sigma=10)
    ppt_mu = similarity_c + income_mu * income_d

    # septic-specific model parameters
    ppt = pm.Normal('ppt', mu=ppt_mu.mean(), sigma=ppt_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + ppt[basin] * ppt_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    ppt_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
    return_inferencedata=True, target_accept=0.99, random_seed=rs)

# soil model
with pm.Model(coords=coords) as soil_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
    age_d = pm.Data('age_d', df.age_median_norm.values, dims='septic')
    income_d = pm.Data('income_d', df.income_median_norm.values, dims='septic')

    # global model parameters
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # similarity parameters
    similarity_c = pm.Normal('similarity_c', mu=0, sigma=10)
    age_mu = pm.Normal('age_mu', mu=0, sigma=10)
    income_mu = pm.Normal('income_mu', mu=0, sigma=10)
    hydr_mu = similarity_c + age_mu * age_d + income_mu * income_d

    # septic-specific model parameters
    hydr = pm.Normal('hydr', mu=hydr_mu.mean(), sigma=hydr_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + hydr[basin] * hydr_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    soil_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
    return_inferencedata=True, target_accept=0.99, random_seed=rs)

# topo model
with pm.Model(coords=coords) as topo_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    slope_d = pm.Data('slope_d', df.slope_norm.values, dims='septic')

    # global model parameters
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # similarity parameters
    similarity_c = pm.Normal('similarity_c', mu=0, sigma=10)
    slope_beta = similarity_c

    # septic-specific model parameters
    slope = pm.Exponential('slope', lam=slope_beta.mean(), dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + slope[basin] * slope_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    topo_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
    return_inferencedata=True, target_accept=0.99, random_seed=rs)

# socio-economic model
with pm.Model(coords=coords) as socio_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')
    age_d = pm.Data('age_d', df.age_median_norm.values, dims='septic')
    income_d = pm.Data('income_d', df.income_median_norm.values, dims='septic')

    # global model parameters
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # similarity parameters
    similarity_c = pm.Normal('similarity_c', mu=0, sigma=10)
    age_mu = pm.Normal('age_mu', mu=0, sigma=10)
    income_mu = pm.Normal('income_mu', mu=0, sigma=10)
    hse_mu = similarity_c + age_mu * age_d + income_mu * income_d

    # septic-specific model parameters
    hse = pm.Normal('hse', mu=hse_mu.mean(), sigma=hse_sig, dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    + hse[basin] * hse_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    socio_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
    return_inferencedata=True, target_accept=0.99, random_seed=rs)

# full model
# Modeling (full)
with pm.Model(coords=coords) as full_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx, dims='septic')
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values, dims='septic')
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values, dims='septic')
    hse_d = pm.Data('hse_d', df.median_hse_norm.values, dims='septic')
    slope_d = pm.Data('slope_d', df.slope_norm.values, dims='septic')
    age_d = pm.Data('age_d', df.age_median_norm.values, dims='septic')
    income_d = pm.Data('income_d', df.income_median_norm.values, dims='septic')

    # global model parameters
    ppt_sig = pm.HalfNormal('ppt_sig', sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # similarity parameters
    ppt_similarity_c = pm.Normal('ppt_similarity_c', mu=0, sigma=10)
    hydr_similarity_c = pm.Normal('hydr_similarity_c', mu=0, sigma=10)
    slope_similarity_c = pm.Normal('slope_similarity_c', mu=0, sigma=10)
    hse_similarity_c = pm.Normal('hse_similarity_c', mu=0, sigma=10)

    ppt_income_mu = pm.Normal('ppt_income_mu', mu=0, sigma=10)
    ppt_mu = ppt_similarity_c + ppt_income_mu * income_d

    hydr_age_mu = pm.Normal('hydr_age_mu', mu=0, sigma=10)
    hydr_income_mu = pm.Normal('hydr_income_mu', mu=0, sigma=10)
    hydr_mu = hydr_similarity_c + hydr_age_mu * age_d + hydr_income_mu * income_d

    slope_beta = slope_similarity_c

    hse_age_mu = pm.Normal('hse_age_mu', mu=0, sigma=10)
    hse_income_mu = pm.Normal('hse_income_mu', mu=0, sigma=10)
    hse_mu = hse_similarity_c + hse_age_mu * age_d + hse_income_mu * income_d

    # septic-specific model parameters
    ppt = pm.Normal('ppt', mu=ppt_mu.mean(), sigma=ppt_sig, dims='basin')
    hydr = pm.Normal('hydr', mu=hydr_mu.mean(), sigma=hydr_sig, dims='basin')
    hse = pm.Normal('hse', mu=hse_mu.mean(), sigma=hse_sig, dims='basin')
    slope = pm.Exponential('slope', lam=slope_beta.mean(), dims='basin')
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
    
    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c[basin] 
                                    # + wtr_dist[basin] * water_d 
                                    + ppt[basin] * ppt_d
                                    + hydr[basin] * hydr_d
                                    + hse[basin] * hse_d
                                    + slope[basin] * slope_d
                                   )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta, observed=df['sewageSystem_enc'])
    
    # fitting using NUTS sampler
    full_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
    return_inferencedata=True, target_accept=0.99, random_seed=rs)

traces_dict = dict()
traces_dict.update({'L1_Precip': ppt_trace,
                    'L1_Soil': soil_trace,
                    'L1_Topo': topo_trace,
                    'L1_Socio': socio_trace,
                    'L1_Full': full_trace
                    })

###########################################
############# Pooled Bayesian #############
###########################################
print('Fitting pooled Bayesian models...')
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
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)

    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c + ppt * ppt_d)

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta,
                            observed=df['sewageSystem_enc'])

    # fitting using NUTS sampler
    ppt_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
                          return_inferencedata=True, target_accept=0.99, random_seed=rs)

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
    failure_theta = pm.math.sigmoid(c + hydr * hydr_d)

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta,
                            observed=df['sewageSystem_enc'])

    # fitting using NUTS sampler
    soil_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
                           return_inferencedata=True, target_accept=0.99, random_seed=rs)

# topo model
with pm.Model(coords=coords) as topo_model:
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    slope_d = pm.Data('slope_d', df.slope_norm.values)

    # global model parameters
    slope_beta = pm.HalfNormal('slope_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    slope = pm.Exponential('slope', lam=slope_beta)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)

    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c + slope * slope_d)

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta,
                            observed=df['sewageSystem_enc'])

    # fitting using NUTS sampler
    topo_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
                           return_inferencedata=True, target_accept=0.99, random_seed=rs)

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
    failure_theta = pm.math.sigmoid(c + hse * hse_d)

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta,
                            observed=df['sewageSystem_enc'])

    # fitting using NUTS sampler
    socio_trace = pm.sample(
        100, tune=tune, cores=3, return_inferencedata=True, idata_kwargs={"log_likelihood": True}, 
        target_accept=0.99, random_seed=rs)

# full model
with pm.Model(coords=coords) as full_model:
    print('fitting full pooled Bayesian model...')
    # constant data: basin information and variables
    basin = pm.Data('basin', basin_idx)
    ppt_d = pm.Data('ppt_d', df.ppt_2021_norm.values)
    hydr_d = pm.Data('hydr_d', df.hydraulic_c_norm.values)
    hse_d = pm.Data('hse_d', df.median_hse_norm.values)
    slope_d = pm.Data('slope_d', df.slope_norm.values)

    # global model parameters
    ppt_mu = pm.Normal("ppt_mu", mu=0, sigma=10)
    ppt_sig = pm.HalfNormal("ppt_sig", sigma=10)
    hydr_mu = pm.Normal("hydr_mu", mu=0, sigma=10)
    hydr_sig = pm.HalfNormal('hydr_sig', sigma=10)
    hse_sig = pm.HalfNormal('hse_sig', sigma=10)
    slope_beta = pm.HalfNormal('slope_beta', sigma=10)
    mu_c = pm.Normal('mu_c', mu=0, sigma=10)
    sigma_c = pm.HalfNormal('sigma_c', sigma=10)

    # septic-specific model parameters
    ppt = pm.Normal("ppt", mu=ppt_mu, sigma=ppt_sig)
    hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig)
    hse = pm.Normal('hse', mu=0, sigma=hse_sig)
    slope = pm.Exponential('slope', lam=slope_beta)
    c = pm.Normal('c', mu=mu_c, sigma=sigma_c)

    # hierarchical bayesian formula
    failure_theta = pm.math.sigmoid(c +
                                    ppt * ppt_d +
                                    hydr * hydr_d +
                                    hse * hse_d +
                                    slope * slope_d
                                    )

    # likelihood of observed data
    failures = pm.Bernoulli('failures', failure_theta,
                            observed=df['sewageSystem_enc'])

    # fitting using NUTS sampler
    full_trace = pm.sample(100, tune=tune, cores=3, idata_kwargs={"log_likelihood": True},
                           return_inferencedata=True, target_accept=0.99, random_seed=rs)

traces_dict.update({'L0_Precip': ppt_trace,
                    'L0_Soil': soil_trace,
                    'L0_Topo': topo_trace,
                    'L0_Socio': socio_trace,
                    'L0_Full': full_trace})