# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:44:12 2025

@author: Eric Brandão - Line parameter evaluation with Baysian analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from baysian_sampling import BayesianSampler
#%% Line parameters: y = mx + b (we don't know m and b)
m = -1.5
b = 3.0
#true line - full discretization
x_true = np.linspace(-10, 10, 1000)
y_true = m * x_true + b
# measured line
n_meas = 40
indx = np.arange(0, len(x_true), len(x_true)/n_meas, dtype = int)
x_meas = x_true[indx]
y_meas = y_true[indx] + np.random.normal(0, 1, len(x_meas))
# plot
plt.figure(figsize = (8, 3))
plt.plot(x_true, y_true, '-k', linewidth = 2, label = 'True line')
plt.plot(x_meas, y_meas, 'or', label = 'Measured')
plt.grid(linestyle = '--')
plt.xlabel("x [m]")
plt.ylabel("y [-]")
plt.legend()
plt.tight_layout();

#%% Define your model
def line_model(x_meas, model_par = [1, 0]):
    """ Model for a line. It will be evoked during baysian inference multiple times
        Parameters
    ----------
    x_meas : coordinates of the measurements
    m : flat (inclication of line)
    b : flat (cross of line)
    Returns
    ----------
    y_pred : predicted line a meas coordinates
    """
    y_pred = model_par[0] * x_meas + model_par[1]
    return y_pred

y_pred_test = line_model(x_meas, model_par = [m, b])
#%%
ba = BayesianSampler(measured_coords = x_meas, measured_data = y_meas,
                     num_model_par = 2, seed = 42)
ba.set_model_fun(model_fun = line_model)
ba.set_uniform_prior_limits(lower_bounds = [-5, -5], upper_bounds = [5, 5])
ba.set_convergence_tolerance(convergence_tol = [0.1, 0.2])
#prior_samples = ba.sample_uniform_prior(num_samples = 100)
#prior_samples, weights, _ = ba.brute_force_sampling(num_samples = 100000)

#%%
# prior_samples, weights, logp = ba.nested_sampling(n_live = 100, max_iter=2000)
ba.nested_sampling(n_live = 400, max_iter = 5000, max_up_attempts = 50, seed = 42)
#%%
plt.scatter(ba.live_pts[:,0], ba.live_pts[:,1])
# plt.scatter(ba.dead_pts[:,0], ba.dead_pts[:,1], alpha = 0.5)
plt.scatter(m, b, marker='x')
plt.xlim((ba.lower_bounds[0], ba.upper_bounds[0]))
plt.ylim((ba.lower_bounds[1], ba.upper_bounds[1]))
plt.xlabel("m")
plt.ylabel("b")
#%%
#ba.plot_histograms()
ba.plot_smooth_marginal_posterior()

#%% Statistics and reconstruction
ba.compute_statistics()
y_recon = ba.reconstruct_mean(x_meas = x_true)
y_recon_ci = ba.reconstruct_ci(x_meas = x_true)
print("True values: m = {:.2f}, b = {:.2f}".format(m, b))
print("Inferred values: m = {:.2f}, b = {:.2f}".format(ba.mean_values[0], ba.mean_values[1]))
#%% plot
plt.figure(figsize = (8, 3))
plt.plot(x_true, y_true, '--k', linewidth = 2, label = 'True line')
plt.plot(x_meas, y_meas, 'or', label = 'Measured')
plt.plot(x_true, y_recon, '-', label = 'Reconstructed')
plt.fill_between(x_true, y_recon_ci[0,:], y_recon_ci[1,:], color='tab:blue', alpha=0.2)
plt.grid(linestyle = '--')
plt.xlabel("x [m]")
plt.ylabel("y [-]")
plt.legend()
plt.tight_layout();
