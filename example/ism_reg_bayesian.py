# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 2025

@author: Eric Brandão
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
# from controlsair import AlgControls, AirProperties, load_cfg, sph2cart
# from sources import Source
# from receivers import Receiver
# from material import PorousAbsorber  # Material
from field_inf_nlr import NLRInfSph  # Field Inf NLR
# from decomp_quad_v2 import Decomposition_QDT
from decomp2mono import Decomposition_2M
from decomp2mono_bayesian import Decomposition_2M_Bayesian
# import lcurve_functions as lc
import utils_insitu as ut_is

#%% Import sound field for processing with ISM method
field_nlr = NLRInfSph()
field_nlr.load("field_melamine_jcal_r110_array0")
field_nlr.add_noise(snr = 30, uncorr=False)
#%% ISM estimation (Tikhonov)
ism = Decomposition_2M(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
                       receivers = field_nlr.receivers, 
                       source_coord = field_nlr.sources.coord[0],
                       regu_par = 'L-curve')
ism.pk_tikhonov(plot_l=False, method='Tikhonov')
#decomp_mono.least_squares_pk()
_ = ism.zs(Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=[0], avgZs=True);

#%% Limits from Tikhonov
s_min_r, s_max_r = np.real(ism.pk[0,:]).min(), np.real(ism.pk[0,:]).max()
s_min_i, s_max_i = np.imag(ism.pk[0,:]).min(), np.imag(ism.pk[0,:]).max()
is_min_r, is_max_r = np.real(ism.pk[1,:]).min(), np.real(ism.pk[1,:]).max()
is_min_i, is_max_i = np.imag(ism.pk[1,:]).min(), np.imag(ism.pk[1,:]).max()
#%% ISM estimation (Bayesian)
factor = 0.5
ism_b = Decomposition_2M_Bayesian(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
                       receivers = field_nlr.receivers, 
                           source_coord = field_nlr.sources.coord[0])
ism_b.set_prior_limits(lower_bounds = [s_min_r-factor, s_min_i-factor, 
                                       is_min_r-factor, is_min_i-factor], 
                       upper_bounds = [s_max_r+factor, s_max_i+factor, 
                                       is_max_r+factor, is_max_i+factor])

# Sampling
ism_b.pk_bayesian(n_live = 250, max_iter = 3000, max_up_attempts = 50, seed = 0)
#%%
alpha_vp = ism_b.ref_coeff()
#%% Plot absorption
_, ax = ut_is.give_me_an_ax()
ut_is.plot_absorption(field_nlr.controls.freq, field_nlr.material.alpha, ax[0,0], 
                      label = 'JCA', color = 'k', linestyle = '--')
ut_is.plot_absorption(ism.controls.freq, ism.alpha.flatten(), ax[0,0], 
                      label = 'ISM (Tikhonov)', color = 'maroon', linestyle = ':')
ut_is.plot_absorption(ism_b.controls.freq, alpha_vp, ax[0,0], 
                      label = 'ISM (Bayesian)', color = 'dodgerblue', linestyle = '-')
# ax[0,0].fill_between(ism_b.controls.freq, 10*np.log10(edc_recon_ci[0,:]/np.amax(edc_recon_ci[0,:])),
#                   10*np.log10(edc_recon_ci[1,:]/np.amax(edc_recon_ci[1,:])),
#                   color='grey', alpha=0.3)
plt.tight_layout()