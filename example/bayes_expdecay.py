# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:48:12 2025

@author: Win11 - Double exponential decay with Baysian analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from baysian_sampling import BayesianSampler
#%% True decay
a0 = 1e-4
a1 = 10**0
a2 = 10**(-6/10)
T1 = 0.5
T2 = 1
tK = 1

#%% Data generation
fs = 44100 # sample rate
time = np.arange(0, tK, 1/fs)
# Decay curve
decay1 = a1*np.exp(-13.8*time/T1)
decay2 = a2*np.exp(-13.8*time/T2)
noise = a0*(tK-time)
true_edc = decay1 + decay2 + noise
# Noisy h(t)
rng = np.random.default_rng(42)
white_noise = np.random.normal(loc = 0.0, scale = 1.0, size = len(time))
# ht = white_noise * np.sqrt(true_edc)
ht = white_noise * true_edc
ht2 = white_noise**2 * true_edc

skip = 10 # samples
time_meas = time[0::skip]
measured_edc = true_edc[0::skip] + np.random.normal(loc = 0, 
                                                    scale = 0.01*true_edc[0::skip], 
                                                    size = len(time_meas)) 

#%% Plot data
plt.figure(figsize = (6,4))
plt.plot(time, 10*np.log10(true_edc/np.amax(true_edc)), linewidth = 2, label = "True EDC")
plt.plot(time, 10*np.log10(decay1), '--', color = 'grey', label = "Decay 1")
plt.plot(time, 10*np.log10(decay2), '-.', color = 'grey', label = "Decay 2")
plt.plot(time, 10*np.log10(noise), ':', color = 'grey', label = "Noise")
plt.legend()
plt.grid(linestyle = '--')
plt.xlim([0, tK])
plt.ylim([-60, 2])
plt.xlabel("Time [s]")
plt.ylabel("EDC [dB]")
plt.tight_layout()

#%%
plt.figure(figsize = (6,4))
plt.plot(time, 10*np.log10(ht2/np.amax(ht2)), 'tab:blue', linewidth = 0.4,
          label = "h(t)", alpha = 0.4)

plt.plot(time, 10*np.log10(true_edc/np.amax(true_edc)), linewidth = 1, label = "True EDC")

plt.plot(time_meas, 10*np.log10(measured_edc/np.amax(measured_edc)), ':',  
          color = 'm', alpha = 0.6, label = "Measured EDC")
plt.legend()
plt.grid(linestyle = '--')
plt.xlim([-0.1, tK])
plt.ylim([-60, 2])
plt.xlabel("Time [s]")
plt.ylabel("EDC [dB]")
plt.tight_layout()

#%% Define your model
def exp_decay(x_meas, model_par = [1, 0.7, 1.8, 4, 1e-4]):
    """ Model for an exponential decay. 
    Parameters
    ----------
    x_meas : coordinates of the measurements
    Returns
    ----------
    y_pred : predicted line a meas coordinates
    """
    decay1 = model_par[0] * np.exp(-13.8*x_meas/model_par[2])
    decay2 = model_par[1] * np.exp(-13.8*x_meas/model_par[3])
    noise = model_par[4] * (tK - x_meas)
    y_pred = decay1 + decay2 + noise
    return y_pred

# y_pred_test = exp_decay(time_meas, model_par = [a1, a2, T1, T2, a0])

#%%
ba = BayesianSampler(measured_coords = time_meas, 
                     measured_data = measured_edc,
                     num_model_par = 5, seed = 42)
ba.set_model_fun(model_fun = exp_decay)
ba.set_uniform_prior_limits(lower_bounds = [0.5, 0.1, 0.1, 0.5, 5e-5], 
                            upper_bounds = [1.5, 1, 1.0, 2.5, 5e-4])
# prior_samples, weights, _ = ba.brute_force_sampling(num_samples = 1000000)

#%%
prior_samples, weights, Z = ba.nested_sampling(n_live = 800, max_iter = 3000)
posterior_ns = prior_samples[rng.choice(len(prior_samples), size=5000, p=weights)]
plt.scatter(posterior_ns[:,0], posterior_ns[:,2])
plt.xlim((ba.lower_bounds[0], ba.upper_bounds[0]))
plt.ylim((ba.lower_bounds[2], ba.upper_bounds[2]))
#%%
ba.plot_histograms()
ba.plot_smooth_marginal_posterior()

#%%
ba.compute_statistics()
edc_recon = ba.reconstruct_mean(x_meas = time)
edc_recon_ci = ba.reconstruct_ci(x_meas = time)
print("True values [A1, A2, T1, T2, A0]: {}".format([a1, a2, T1, T2, a0]))
print("Mean values [A1, A2, T1, T2, A0]: {}".format(ba.mean_values))
#%%
plt.figure(figsize = (6,4))
# plt.plot(time, 10*np.log10(ht2/np.amax(ht2)), 'tab:blue', linewidth = 0.4,
#           label = "h(t)", alpha = 0.4)

plt.plot(time, 10*np.log10(true_edc/np.amax(true_edc)), linewidth = 1, label = "True EDC")

plt.plot(time, 10*np.log10(edc_recon/np.amax(edc_recon)), '--',  
          color = 'm', alpha = 0.6, label = "Recon EDC")
plt.fill_between(time, 10*np.log10(edc_recon_ci[0,:]/np.amax(edc_recon_ci[0,:])),
                 10*np.log10(edc_recon_ci[1,:]/np.amax(edc_recon_ci[1,:])),
                 color='grey', alpha=0.3)
plt.legend()
plt.grid(linestyle = '--')
plt.xlim([-0.1, tK])
plt.ylim([-60, 2])
plt.xlabel("Time [s]")
plt.ylabel("EDC [dB]")
plt.tight_layout()