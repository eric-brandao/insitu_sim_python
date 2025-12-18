# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:44:12 2025

@author: Eric Brandão - Line parameter evaluation with Baysian analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from baysian_sampling import BayesianSampler
np.random.seed(0)
#%% Line parameters: y = mx + b (we don't know m and b)
m = 2
b = 3.0
#true line - full discretization
x_true = np.linspace(0, 5, 50)
y_true = m * x_true + b
# measured line
n_meas = 50
indx = np.arange(0, len(x_true), len(x_true)/n_meas, dtype = int)
x_meas = x_true[indx]
y_meas = y_true[indx] + np.random.normal(0, 0.5, len(x_meas))
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
    m : float (inclication of line)
    b : float (cross of line)
    Returns
    ----------
    y_pred : predicted line a meas coordinates
    """
    y_pred = model_par[0] * x_meas + model_par[1]
    return y_pred

y_pred_test = line_model(x_meas, model_par = [m, b])
#%%
ba = BayesianSampler(measured_coords = x_meas, measured_data = y_meas,
                     num_model_par = 2, parameters_names = ["m", "b"],
                     likelihood = "logt", sampling_scheme='single ellipsoid',
                     enlargement_factor = 1.5) #"Gaussian1D"
ba.set_model_fun(model_fun = line_model)
ba.set_uniform_prior_limits(lower_bounds = [-10, 0], upper_bounds = [10, 10])

print("Old bounds: lower: {} / upper: {}".format(ba.lower_bounds, ba.upper_bounds))
# ba.wide_prior_range(widening_factor=0.5)
# print("New bounds: lower: {} / upper: {}".format(ba.lower_bounds, ba.upper_bounds))
#%%
ba.nested_sampling(n_live = 100, max_iter = 1000, max_up_attempts = 100, seed = 0,
                   dlogz = 0.001)
print("\n My Nested log-evidence: {:.4f} +/- {:.4f}".format(ba.logZ, ba.logZ_err))
print("\n My Nested h: {:.4f}".format(ba.info))
#%%
sampler = ba.ultranested_sampling(n_live = 500, max_iter = 6000)
#%%
plt.figure()
plt.scatter(ba.live_cube[:,0], ba.live_cube[:,1])
# plt.scatter(ba.dead_pts[:,0], ba.dead_pts[:,1], alpha = 0.1)
# plt.scatter(ba.live_pts[:,0], ba.live_pts[:,1])
# plt.scatter(m, b, marker='x')
# plt.xlim((ba.lower_bounds[0], ba.upper_bounds[0]))
# plt.ylim((ba.lower_bounds[1], ba.upper_bounds[1]))
plt.grid(linestyle = '--', alpha = 0.7)
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel("m")
plt.ylabel("b")
#%%
#ba.plot_histograms()
ba.plot_smooth_marginal_posterior(figshape = (1,2), figsize = (9, 2.5))

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

#%%
# ba.ultranest_slice_sampler()

#%%
# def line_model2(x_meas, model_par = [[1, 0]]):
#     """ Model for a line. It will be evoked during baysian inference multiple times
#         Parameters
#     ----------
#     x_meas : coordinates of the measurements
#     m : flat (inclication of line)
#     b : flat (cross of line)
#     Returns
#     ----------
#     y_pred : predicted line a meas coordinates
#     """
#     y_pred = model_par[0,0] * x_meas + model_par[0,1]
#     return y_pred
# ba.model_fun = line_model2