# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:17:14 2025

@author: Eric - Baysian sine signal detection with Baysian analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from baysian_sampling import BayesianSampler

#%%
num_meas = 50
x_meas = np.linspace(0, 1, num_meas)
true_amp = 2.0
true_freq = 1.0
y_meas = true_amp * np.sin(2*np.pi*true_freq*x_meas) + 0.1 * np.random.randn(num_meas)

plt.figure(figsize = (5,3))
plt.plot(x_meas, y_meas, 'o', label = "noisy signal");
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('x'); plt.ylabel('y');
plt.tight_layout()

#%% Define your model
def sine_sig(x_meas, model_par = [1, 1]):
    """ Model for sine singal (amp and freq). 
    Parameters
    ----------
    x_meas : coordinates of the measurements
    
    Returns
    ----------
    y_pred : predicted line a meas coordinates
    """
    
    y_pred = model_par[0] * np.sin(2*np.pi*model_par[1]*x_meas)
    return y_pred

#%%
ba = BayesianSampler(measured_coords = x_meas, 
                     measured_data = y_meas,
                     parameters_names = ["Amp", "freq"],
                     num_model_par = 2, seed = 42)
ba.set_model_fun(model_fun = sine_sig)
ba.set_uniform_prior_limits(lower_bounds = [0.0, 0.5], 
                            upper_bounds = [4.0, 2.0])
ba.set_convergence_tolerance(convergence_tol = [0.1, 0.1])

#%%
ba.nested_sampling(n_live = 100, max_iter = 1500, max_up_attempts = 200, seed = 42)
print("\n Log-Evidence value: {:.4f}".format(ba.logZ))
#%%
ba.plot_loglike()
ba.plot_spread()

#%%
plt.figure()
plt.scatter(ba.dead_pts[:,0], ba.dead_pts[:,1], alpha = 0.1, label = "Dead pts")
plt.scatter(ba.init_pop[:,0], ba.init_pop[:,1], alpha = 0.1, label = "Init. pop")
plt.scatter(ba.live_pts[:,0], ba.live_pts[:,1], alpha = 0.3, label = "Live pts")
plt.scatter(true_amp, true_freq, marker='x', s = 200, label = "True Value")
# plt.scatter(a2, T2, marker='x')
plt.legend()
plt.xlim((ba.lower_bounds[0], ba.upper_bounds[0]))
plt.ylim((ba.lower_bounds[1], ba.upper_bounds[1]))
plt.xlabel(ba.parameters_names[0])
plt.ylabel(ba.parameters_names[1])
#%%
ba.plot_smooth_marginal_posterior(figshape = (1,2), figsize = (8,3))
#%%
ba.compute_statistics()
y_recon = ba.reconstruct_mean(x_meas = x_meas)
y_recon_ci = ba.reconstruct_ci(x_meas = x_meas)
print("True values [Amp, freq, T1, T2, A0]: {}".format([true_amp, true_freq]))
print("Mean values [Amp, freq]: {}".format(ba.mean_values))

#%%
plt.figure(figsize = (6,4))
plt.plot(x_meas, y_meas, 'o', label = "noisy signal");
plt.plot(x_meas, y_recon, '-', color = 'k', alpha = 1, label = "Recon")
# plt.fill_between(time, 10*np.log10(edc_recon_ci[0,:]/np.amax(edc_recon_ci[0,:])),
#                   10*np.log10(edc_recon_ci[1,:]/np.amax(edc_recon_ci[1,:])),
#                   color='grey', alpha=0.3)
plt.legend()
plt.grid(linestyle = '--')
plt.xlim([0, x_meas[-1]])
#plt.ylim([-1, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()