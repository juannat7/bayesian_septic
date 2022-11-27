import os
import json
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pymc as pm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .utils import read_data, evaluate_bayes


class _baseModelMethod(ABC):
    def __init__(self, pars):
        self.pars = pars
        self.model_name = pars['name']
        
        self.if_run_reports = eval(self.pars['run_reports'])
        self.report_pars = self.pars['report_script']
        self.pars = {x: self.pars['params'][x] for x in self.pars['params']}
        
        self.df, self.basin_idx, self.catchment_idx, self.coords = read_data(file_dir=f'data/{self.pars["main_data"]}',
                                                                             cols=eval(self.pars["default_columns"]),
                                                                             is_balanced=eval(self.pars["is_balanced"]),
                                                                             is_multilevel=eval(self.pars["is_multilevel"]))
        
        self.report = {}
        
        return

    def _output_report(self):
        if not os.path.exists("tmp/"):
            os.mkdir("tmp/")
        
        with open("tmp/" + self.model_name + '_output.json', 'w') as out:
            json.dump(self.report, out)
        
        print("Report generated!")
        return
    
    @abstractmethod
    def run_scripts(self):
        # Implement this method as your modeling script
        return
    
    def _report_VIF(self):
        vif_data = pd.DataFrame()
        cluster_var = [x + '_norm' for x in eval(self.pars["default_columns"])]
        X = self.df[cluster_var]
        vif_data['feature'] = cluster_var
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(cluster_var))]
        
        self.report['VIF'] = {k: v for k, v in zip(vif_data['feature'], vif_data['VIF'])}
        return

    def _report_model_acc(self, trace, model, name):
        y = self.df[self.pars["observe_column"]].to_list()
        accs = []
        for _ in range(self.pars["evaluate_times"]):
            acc, y_pred = evaluate_bayes(trace, model, y)
            accs.append(acc)
        
        self.report[name] = {"accuracy_mean": np.array(accs).mean(), "accuracy_std": np.array(accs).std()}
        return


class HirarchicalBayesian(_baseModelMethod):
    def __init__(self, pars):
        super().__init__(pars)
        return

    def _water_model(self):
        with pm.Model(coords=self.coords) as model:
            # constant data: basin information and variables
            basin = pm.Data('basin', self.basin_idx, dims='septic')
            # catchment = pm.Data('catchment', self.catchment_idx, dims='septic')
            # water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
            ppt_d = pm.Data('ppt_d', self.df.ppt_2021_norm.values, dims='septic')

            # global model parameters
            # wtr_beta = pm.HalfNormal("wtr_beta", sigma=10)
            ppt_mu = pm.Normal("ppt_mu", mu=self.pars['water_model']['ppt_mu']['mu'], sigma=self.pars['water_model']['ppt_mu']['sigma'])
            ppt_sig = pm.HalfNormal("ppt_sig", sigma=self.pars['water_model']['ppt_sig']['sigma'])
            mu_inter = pm.Normal('mu_inter', mu=self.pars['water_model']['mu_inter']['mu'], sigma=self.pars['water_model']['mu_inter']['sigma'])
            sigma_inter = pm.HalfNormal('sigma_inter', sigma=self.pars['water_model']['sigma_inter']['sigma'])
            
            # catchment parameters
            # wtr_beta_c = pm.HalfNormal("wtr_beta_c", sigma=wtr_beta, dims='catchment')
            ppt_mu_c = pm.Normal("ppt_mu_c", mu=ppt_mu, sigma=self.pars['water_model']['ppt_mu_c']['sigma'], dims='catchment')
            ppt_sig_c = pm.HalfNormal("ppt_sig_c", sigma=ppt_sig, dims='catchment')
            
            mu_inter_c = pm.Normal('mu_inter_c', mu=mu_inter, sigma=self.pars['water_model']['mu_inter_c']['sigma'], dims='catchment')
            sigma_inter_c = pm.HalfNormal('sigma_inter_c', sigma=sigma_inter, dims='catchment')

            # septic-specific model parameters
            # wtr_dist = pm.Exponential("wtr_dist", lam=wtr_beta_c.mean(), dims="basin")
            ppt = pm.Normal("ppt", mu=ppt_mu_c.mean(), sigma=ppt_sig_c.mean(), dims="basin")
            c = pm.Normal('c', mu=mu_inter_c.mean(), sigma=sigma_inter_c.mean(), dims="basin")
            
            # hierarchical bayesian formula
            failure_theta = pm.math.sigmoid(c[basin] +
                                            # + wtr_dist[basin] * water_d
                                            ppt[basin] * ppt_d)

            # likelihood of observed data
            failures = pm.Bernoulli('failures', failure_theta, observed=self.df[self.pars['observe_column']])
            
            # fitting using NUTS sampler
            trace = pm.sample(100, tune=self.pars['tune'], cores=3, return_inferencedata=True, target_accept=0.99, random_seed=self.pars['rs'])
            self._report_model_acc(trace, model, name='water_model')
            
        return
    
    def _soil_model(self):
        with pm.Model(coords=self.coords) as model:
            # constant data: basin information and variables
            basin = pm.Data('basin', self.basin_idx, dims='septic')
            hydr_d = pm.Data('hydr_d', self.df.hydraulic_c_norm.values, dims='septic')

            # global model parameters
            hydr_mu = pm.Normal('hydr_mu', mu=self.pars['soil_model']['hydr_mu']['mu'], sigma=self.pars['soil_model']['hydr_mu']['sigma'])
            hydr_sig = pm.HalfNormal('hydr_sig', sigma=self.pars['soil_model']['hydr_sig']['sigma'])
            mu_c = pm.Normal('mu_c', mu=self.pars['soil_model']['mu_c']['mu'], sigma=self.pars['soil_model']['mu_c']['sigma'])
            sigma_c = pm.HalfNormal('sigma_c', sigma=self.pars['soil_model']['sigma_c']['sigma'])

            # septic-specific model parameters
            hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig, dims='basin')
            c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
            
            # hierarchical bayesian formula
            failure_theta = pm.math.sigmoid(c[basin] + hydr[basin] * hydr_d)

            # likelihood of observed data
            failures = pm.Bernoulli('failures', failure_theta, observed=self.df['sewageSystem_enc'])
            
            # fitting using NUTS sampler
            trace = pm.sample(100, tune=self.pars['tune'], cores=3, return_inferencedata=True, target_accept=0.99, random_seed=self.pars['rs'])
            self._report_model_acc(trace, model, name='soil_model')

    def _topo_model(self):
        with pm.Model(coords=self.coords) as model:
            # constant data: basin information and variables
            basin = pm.Data('basin', self.basin_idx, dims='septic')
            dem_d = pm.Data('dem_d', self.df.dem_norm.values, dims='septic')

            # global model parameters
            dem_beta = pm.HalfNormal('dem_beta', sigma=self.pars['topo_model']['dem_beta']['sigma'])
            mu_c = pm.Normal('mu_c', mu=self.pars['topo_model']['mu_c']['mu'], sigma=self.pars['topo_model']['mu_c']['sigma'])
            sigma_c = pm.HalfNormal('sigma_c', sigma=self.pars['topo_model']['sigma_c']['sigma'])
            
            # septic-specific model parameters
            dem = pm.Exponential('dem', lam=dem_beta, dims='basin')
            c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
            
            # hierarchical bayesian formula
            failure_theta = pm.math.sigmoid(c[basin] + dem[basin] * dem_d)

            # likelihood of observed data
            failures = pm.Bernoulli('failures', failure_theta, observed=self.df['sewageSystem_enc'])
            
            # fitting using NUTS sampler
            trace = pm.sample(100, tune=self.pars['tune'], cores=3, return_inferencedata=True, target_accept=0.99, random_seed=self.pars['rs'])
            self._report_model_acc(trace, model, name='topo_model')
    
    def _socio_model(self):
        with pm.Model(coords=self.coords) as model:
            # constant data: basin information and variables
            basin = pm.Data('basin', self.basin_idx, dims='septic')
            hse_d = pm.Data('hse_d', self.df.median_hse_norm.values, dims='septic')

            # global model parameters
            hse_sig = pm.HalfNormal('hse_sig', sigma=self.pars['socio_model']['hse_sig']['sigma'])
            mu_c = pm.Normal('mu_c', mu=self.pars['socio_model']['mu_c']['mu'], sigma=self.pars['socio_model']['mu_c']['sigma'])
            sigma_c = pm.HalfNormal('sigma_c', sigma=self.pars['socio_model']['sigma_c']['sigma'])

            # septic-specific model parameters
            hse = pm.Normal('hse', mu=0, sigma=hse_sig, dims='basin')
            c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
            
            # hierarchical bayesian formula
            failure_theta = pm.math.sigmoid(c[basin] + hse[basin] * hse_d)

            # likelihood of observed data
            failures = pm.Bernoulli('failures', failure_theta, observed=self.df['sewageSystem_enc'])
            
            # fitting using NUTS sampler
            trace = pm.sample(100, tune=self.pars["tune"], cores=3, return_inferencedata=True, target_accept=0.99, random_seed=self.pars["rs"])
            self._report_model_acc(trace, model, name='socio_model')
    
    def _full_model(self):
        with pm.Model(coords=self.coords) as model:
            # constant data: basin information and variables
            basin = pm.Data('basin', self.basin_idx, dims='septic')
            # water_d = pm.Data('water_d', df.water_dist_norm.values, dims='septic')
            ppt_d = pm.Data('ppt_d', self.df.ppt_2021_norm.values, dims='septic')
            hydr_d = pm.Data('hydr_d', self.df.hydraulic_c_norm.values, dims='septic')
            hse_d = pm.Data('hse_d', self.df.median_hse_norm.values, dims='septic')
            dem_d = pm.Data('dem_d', self.df.dem_norm.values, dims='septic')

            # global model parameters
            # wtr_beta = pm.HalfNormal('wtr_beta', sigma=10)
            ppt_mu = pm.Normal('ppt_mu', mu=self.pars['full_model']['ppt_mu']['mu'], sigma=self.pars['full_model']['ppt_mu']['sigma'])
            ppt_sig = pm.HalfNormal('ppt_sig', sigma=self.pars['full_model']['ppt_sig']['sigma'])
            hydr_mu = pm.Normal('hydr_mu', mu=self.pars['full_model']['hydr_mu']['mu'], sigma=self.pars['full_model']['hydr_mu']['sigma'])
            hydr_sig = pm.HalfNormal('hydr_sig', sigma=self.pars['full_model']['hydr_sig']['sigma'])
            hse_sig = pm.HalfNormal('hse_sig', sigma=self.pars['full_model']['hse_sig']['sigma'])
            dem_beta = pm.HalfNormal('dem_beta', sigma=self.pars['full_model']['dem_beta']['sigma'])
            mu_c = pm.Normal('mu_c', mu=self.pars['full_model']['mu_c']['mu'], sigma=self.pars['full_model']['mu_c']['sigma'])
            sigma_c = pm.HalfNormal('sigma_c', sigma=self.pars['full_model']['sigma_c']['sigma'])

            # septic-specific model parameters
            # wtr_dist = pm.Exponential('wtr_dist', lam=wtr_beta, dims='basin')
            ppt = pm.Normal('ppt', mu=ppt_mu, sigma=ppt_sig, dims='basin')
            hydr = pm.Normal('hydr', mu=hydr_mu, sigma=hydr_sig, dims='basin')
            hse = pm.Normal('hse', mu=self.pars['full_model']['hse']['mu'], sigma=hse_sig, dims='basin')
            dem = pm.Exponential('dem', lam=dem_beta, dims='basin')
            c = pm.Normal('c', mu=mu_c, sigma=sigma_c, dims='basin')
            
            # hierarchical bayesian formula
            failure_theta = pm.math.sigmoid(c[basin] +
                                            # + wtr_dist[basin] * water_d
                                            ppt[basin] * ppt_d +
                                            hydr[basin] * hydr_d +
                                            hse[basin] * hse_d +
                                            dem[basin] * dem_d
                                            )

            # likelihood of observed data
            failures = pm.Bernoulli('failures', failure_theta, observed=self.df['sewageSystem_enc'])
            
            # fitting using NUTS sampler
            trace = pm.sample(100, tune=self.pars["tune"], cores=3, return_inferencedata=True, target_accept=0.99, random_seed=self.pars["rs"])
            self._report_model_acc(trace, model, name='full_model')
    
    def run_scripts(self, mod='full_model'):
        self._report_VIF()
        
        all_methods = ["water_model", "soil_model", "topo_model", "socio_model", "full_model"]
        self.report['run_model'] = []
        
        if mod is None:
            getattr(self, "_full_model")()
            self.report['run_model'].append("full_model")
        
        elif mod in all_methods:
            getattr(self, "_" + mod)()
            self.report['run_model'].append(mod)
        
        else:
            raise ValueError(f"Unknown mod, please use one of the {all_methods}")

        self._output_report()
        return
