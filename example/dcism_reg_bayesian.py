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
path = 'D:/Work/UFSM/Pesquisa/insitu_arrays/TAMURA_DCISM/dcism_locallyreacting_Rtheta/'
field_nlr.load(path = path, filename = "melamine_nlr_sim")
field_nlr.add_noise(snr = 30, uncorr=False)
field_nlr.material.jcal(resistivity = field_nlr.material.resistivity, 
                        porosity = field_nlr.material.porosity, 
                        tortuosity = field_nlr.material.tortuosity, 
                        lam = field_nlr.material.lam, lam_l = field_nlr.material.lam_l)
field_nlr.material.layer_over_rigid(thickness = field_nlr.material.thickness, theta = 0.0);
field_nlr.material.layer_over_rigid_theta(thickness = field_nlr.material.thickness)


# plt.figure(figsize=(8,3))
# for jrec in range(field_nlr.receivers.coord.shape[0]):
#     H = field_nlr.pres_s[0][jrec,:]/field_nlr.pres_s[0][0,:]
#     plt.semilogx(field_nlr.controls.freq, 20*np.log10(np.abs(H)))
#%% DCISM estimation (Tikhonov)
# dcism = Decomposition_QDT(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
#                        receivers = field_nlr.receivers, 
#                        source_coord = field_nlr.sources.coord[0], quad_order=51,
#                        a = 0, b = 30, retraction = 0, image_source_on = True,
#                        regu_par = 'gcv')
# dcism.gauss_legendre_sampling()
# dcism.pk_tikhonov(plot_l=False, method='Tikhonov')
# #decomp_mono.least_squares_pk()
# _ = dcism.zs(Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=[0], avgZs=True);

#%% True kp and rhop (single freq)
id_f = 5
print("Frequency is {} [Hz]".format(field_nlr.controls.freq[id_f]))
print(r"True $k_p$: {}".format(field_nlr.material.kp[id_f]))
print(r"True $\rho_p$: {}".format(field_nlr.material.rhop[id_f]))
kp = field_nlr.material.kp[id_f]
rhop = field_nlr.material.rhop[id_f]
Zp = rhop * field_nlr.controls.w[id_f]/kp
Zs = -1j*Zp*(1 / np.tan(kp*field_nlr.material.thickness))
Vp_true = (Zs-(air.rho0*air.c0))/(Zs+(air.rho0*air.c0))
alpha_true = 1-(np.abs(Vp_true))**2
#%% DCISM estimation (Bayesian)
dcism_b = DCISM_Bayesian(p_mtx = field_nlr.pres_s[0], controls=field_nlr.controls,
                         air = air, receivers = field_nlr.receivers, 
                         source = field_nlr.sources)
# dcism_b.show_available_models()
dcism_b.choose_forward_model(chosen_model = 3)
#dcism_b.show_chosen_model_parameters()
# dcism_b.set_prior_limits(lower_bounds = [3.00, -56.00, 1.19, -64.00, 1e-5, -np.pi], 
#                          upper_bounds = [189.00, -1.00, 17.00, -0.15, 1e-3, np.pi])
# dcism_b.set_prior_limits(lower_bounds = [3.00, -56.00, 1.19, -64.00], 
#                          upper_bounds = [189.00, -1.00, 17.00, -0.15])
dcism_b.set_prior_limits(lower_bounds = [15.00, -30.00, 1.22, -20.00], 
                          upper_bounds = [35.00, -1.00, 10.00, -0.05])
dcism_b.setup_dDCISM(T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
dcism_b.set_sample_thickness(t_p = field_nlr.material.thickness)
dcism_b.set_reference_sensor(ref_sens = 0)
ba = dcism_b.nested_sampling_single_freq(jf = id_f, n_live = 50, max_iter = 200,
                                          max_up_attempts = 50, seed = 0)
ba.confidence_interval(ci_percent = 99)
# dcism_b.pk_bayesian(n_live = 250, max_iter = 500, max_up_attempts = 50, seed = 0)
#%%
ba.plot_loglike()
axs = ba.plot_smooth_marginal_posterior(figshape = (2, 2))
axs[0,0].axvline(np.real(field_nlr.material.kp[id_f]), linestyle = '--', color = 'k', linewidth = 2)
axs[0,1].axvline(np.imag(field_nlr.material.kp[id_f]), linestyle = '--', color = 'k', linewidth = 2)
axs[1,0].axvline(np.real(field_nlr.material.rhop[id_f]), linestyle = '--', color = 'k', linewidth = 2)
axs[1,1].axvline(np.imag(field_nlr.material.rhop[id_f]), linestyle = '--', color = 'k', linewidth = 2)

#%%
kp_b, rhop_b, _, _, _, alpha_calc = dcism_b.recon_mat_props_nlr(ba, id_f, 
                                                            theta = np.deg2rad(field_nlr.material.theta_deg))
_, _, _, _, _, alpha_calc_l = dcism_b.recon_mat_props_nlr(ba, id_f,
                                                          theta = np.deg2rad(field_nlr.material.theta_deg),
                                                          mode = "lower_ci")
_, _, _, _, _, alpha_calc_u = dcism_b.recon_mat_props_nlr(ba, id_f,
                                                          theta = np.deg2rad(field_nlr.material.theta_deg),
                                                          mode = "upper_ci")

#%% Plot absorption
_, ax = ut_is.give_me_an_ax()
ut_is.plot_absorption_theta(field_nlr.material.theta_deg, field_nlr.material.alpha_ft[id_f,:], 
                            ax = ax[0,0], color = 'k', linewidth = 1.5, linestyle = '--',
                            alpha = 1.0, label = "JCA")
ut_is.plot_absorption_theta(field_nlr.material.theta_deg, alpha_calc, 
                            ax = ax[0,0], color = 'm', linewidth = 1.5, linestyle = '-',
                            alpha = 0.7, label = "Bayesian")
ax[0,0].fill_between(field_nlr.material.theta_deg, alpha_calc_l, alpha_calc_u,
                     color='grey', alpha=0.3)
plt.tight_layout()
#%%
_, ax = ut_is.give_me_an_ax()
field_nlr.material.layer_over_rigid(thickness = field_nlr.material.thickness, theta = 0.0);


ut_is.plot_absorption(field_nlr.controls.freq, field_nlr.material.alpha.flatten(), ax[0,0],  
                      label = 'JCA', color = 'k', linestyle = '--')
# ut_is.plot_absorption(dcism.controls.freq, dcism.alpha.flatten(), ax[0,0], 
#                       label = 'DCISM (Tikhonov)', color = 'm', linestyle = '--')
# ax[0,0].semilogx(field_nlr.controls.freq[id_f], alpha_true, 'ok')
ax[0,0].semilogx(field_nlr.controls.freq[id_f], field_nlr.material.alpha_ft[id_f, 0], 'ok')
ax[0,0].semilogx(field_nlr.controls.freq[id_f], alpha_calc[0], 'sm')
# ut_is.plot_absorption(ism_b.controls.freq, alpha_vp, ax[0,0], 
#                       label = 'ISM (Bayesian)', color = 'dodgerblue', linestyle = '-')
# ax[0,0].fill_between(ism_b.controls.freq, 10*np.log10(edc_recon_ci[0,:]/np.amax(edc_recon_ci[0,:])),
#                   10*np.log10(edc_recon_ci[1,:]/np.amax(edc_recon_ci[1,:])),
#                   color='grey', alpha=0.3)
plt.tight_layout()