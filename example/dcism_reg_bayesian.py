# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:12:36 2025

@author: Win11
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 2025

@author: Eric Brandão
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AirProperties #AlgControls, 
# from sources import Source
# from receivers import Receiver
# from material import PorousAbsorber  # Material
from field_inf_nlr import NLRInfSph  # Field Inf NLR
from decomp_quad_v2 import Decomposition_QDT
from decomp_DCISM_bayesian import DCISM_Bayesian
import lcurve_functions as lc
import utils_insitu as ut_is

#%%
air = AirProperties(c0 = 343.0, rho0 = 1.21)
#%% Import sound field for processing with ISM method
field_nlr = NLRInfSph()
field_nlr.load("field_melamine_jcal_r110_array0")
field_nlr.add_noise(snr = 30, uncorr=False)
#%% ISM estimation (Tikhonov)
dcism = Decomposition_QDT(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
                       receivers = field_nlr.receivers, 
                       source_coord = field_nlr.sources.coord[0], quad_order=51,
                       a = 0, b = 30, retraction = 0, image_source_on = True,
                       regu_par = 'gcv')
dcism.gauss_legendre_sampling()
dcism.pk_tikhonov(plot_l=False, method='Tikhonov')
#decomp_mono.least_squares_pk()
_ = dcism.zs(Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=[0], avgZs=True);


#%% DCISM estimation (Bayesian)
dcism_b = DCISM_Bayesian(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
                         air = air, receivers = field_nlr.receivers, 
                         source = field_nlr.sources)


# dcism_b.show_available_models()
dcism_b.choose_forward_model(chosen_model = 1)
#dcism_b.show_chosen_model_parameters()
# dcism_b.set_prior_limits(lower_bounds = [3.00, -56.00, 1.19, -64.00, 1e-5, -np.pi], 
#                          upper_bounds = [189.00, -1.00, 17.00, -0.15, 1e-3, np.pi])

dcism_b.set_prior_limits(lower_bounds = [20.00, -20.00, 1.211, -3.00, 1e-5, -np.pi], 
                         upper_bounds = [30.00, -10.00, 3.000, -1.00, 1e-3, np.pi])

dcism_b.setup_dDCISM(T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
dcism_b.set_sample_thickness(t_p = field_nlr.material.thickness)
dcism_b.pk_bayesian(n_live = 250, max_iter = 500, max_up_attempts = 50, seed = 0)


#%%
id_f = 10
# dcism_b.current_k0 = field_nlr.controls.k0[id_f]

# vol_vel = 1/(1j*field_nlr.controls.w[id_f]*air.rho0)
kp = field_nlr.material.kp[id_f]

field_nlr.material.jcal(resistivity = field_nlr.material.resistivity, 
                        porosity = field_nlr.material.porosity, 
                        tortuosity = field_nlr.material.tortuosity, 
                        lam = field_nlr.material.lam, lam_l = field_nlr.material.lam_l)
rhop = field_nlr.material.rhop[id_f]

# p_pred = dcism_b.forward_model_1(x_meas = field_nlr.receivers.coord, 
#                         model_par = [np.real(kp), np.imag(kp), np.real(rhop), np.imag(rhop), 
#                                      np.abs(vol_vel), np.angle(vol_vel)])

# print("NMSE = {}:".format(lc.nmse(p_pred, field_nlr.pres_s[0][:,id_f])))
# # Sampling
# ism_b.pk_bayesian(n_live = 250, max_iter = 3000, max_up_attempts = 50, seed = 0)
#%%
# alpha_vp = ism_b.ref_coeff()
#%% Plot absorption
_, ax = ut_is.give_me_an_ax()
ut_is.plot_absorption(field_nlr.controls.freq, field_nlr.material.alpha, ax[0,0], 
                      label = 'JCA', color = 'k', linestyle = '--')
ut_is.plot_absorption(dcism.controls.freq, dcism.alpha.flatten(), ax[0,0], 
                      label = 'DCISM (Tikhonov)', color = 'm', linestyle = '--')
# ut_is.plot_absorption(ism_b.controls.freq, alpha_vp, ax[0,0], 
#                       label = 'ISM (Bayesian)', color = 'dodgerblue', linestyle = '-')
# ax[0,0].fill_between(ism_b.controls.freq, 10*np.log10(edc_recon_ci[0,:]/np.amax(edc_recon_ci[0,:])),
#                   10*np.log10(edc_recon_ci[1,:]/np.amax(edc_recon_ci[1,:])),
#                   color='grey', alpha=0.3)
plt.tight_layout()