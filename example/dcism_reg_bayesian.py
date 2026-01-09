# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 2025

@author: Eric Brandão
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AirProperties, AlgControls
from sources import Source
from receivers import Receiver
from material import PorousAbsorber  # Material
#from field_inf_nlr import NLRInfSph  # Field Inf NLR
from directDCISM import dDCISM
# from decomp_quad_v2 import Decomposition_QDT
from decomp_DCISM_bayesian import DCISM_Bayesian
# import lcurve_functions as lc
import utils_insitu as ut_is

#%% Air, Controls, Material
air = AirProperties(c0 = 343.0, rho0 = 1.21)
controls = AlgControls(c0 = air.c0, freq_vec = np.arange(100, 2000, 100))
thickness =  50/1000
material = PorousAbsorber(air, controls)
material.jcal(resistivity = 12200, porosity = 0.98, tortuosity = 1.01, 
              lam = 115e-6, lam_l = 116e-6)
material.layer_over_rigid_theta(thickness = thickness)
material.layer_over_rigid(thickness = thickness, theta = 0.0);
#%% Source and Recs
receivers = Receiver()
receivers.double_line_array(line_len = 0.85, step = 0.01, axis = 'y', start_at = 0, 
                            zr = 0.04, dz = 0.06)
source = Source(coord = [0, 0, 0.3])
#%% Field
field = dDCISM(air = air, controls = controls, source = source, receivers = receivers, 
               material = material, T0 = 7.5, dt = 0.1, tol = 1e-10, gamma=1.0)
field.predict_p_spk_nlr_layer();
field.add_noise(snr = 30)
#%% True kp and rhop (single freq)
id_f = 5
k0 = field.controls.k0[id_f]
print("Frequency is {} [Hz]".format(field.controls.freq[id_f]))
print(r"True $k_p$: {}".format(field.material.kp[id_f]))
print(r"True $\rho_p$: {}".format(field.material.rhop[id_f]))
kp = field.material.kp[id_f]
rhop = field.material.rhop[id_f]
Zp = rhop * field.controls.w[id_f]/kp
Zs = -1j*Zp*(1 / np.tan(kp*field.material.thickness))
Vp_true = (Zs-(air.rho0*air.c0))/(Zs+(air.rho0*air.c0))
alpha_true = 1-(np.abs(Vp_true))**2


#%% DCISM estimation (Bayesian) - NLR
dcism_b = DCISM_Bayesian(p_mtx = field.pres_mtx, controls = controls, air = air, 
                         receivers = receivers, source = source, 
                         sampling_scheme = 'single ellipsoid', enlargement_factor = 1.1)
dcism_b.choose_forward_model(chosen_model = 4)
dcism_b.show_chosen_model_parameters()
dcism_b.set_mic_pairs()
dcism_b.setup_dDCISM(T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
dcism_b.set_sample_thickness(t_p = field.material.thickness)
dcism_b.set_nested_sampling_parameters(n_live = 50, max_iter = 2000, 
                                       max_up_attempts = 50, seed = 0, dlogz = 0.1,
                                       ci_percent = 99)
#%% 
dcism_b.kp_rhop_range(resist = [3000, 20000], phi = [0.8, 0.99], alpha_inf = [1.0, 1.5], 
                      Lam = [100e-6, 300e-6], Lamlfac = [1.01, 2.0], n_samples = 20000)
lb = dcism_b.lb_mtx[:,id_f]
ub = dcism_b.ub_mtx[:,id_f]
lb, ub = dcism_b.set_prior_limits(lower_bounds = lb, upper_bounds = ub)
#%%
print("Running inference for {:.1f} [Hz]".format(controls.freq[id_f]))
ba = dcism_b.nested_sampling_single_freq(lb, ub, jf = id_f)
print(r"log(Z) = {:.2f} +/- {:.2f}".format(ba.logZ, ba.logZ_err))
# Log like and pde 1D
ba.plot_loglike()
axs = ba.plot_smooth_marginal_posterior(figshape = (2, 2))
axs[0,0].axvline(np.real(field.material.kp[id_f]/k0), linestyle = '--', color = 'k', linewidth = 1)
axs[0,1].axvline(np.imag(field.material.kp[id_f]/k0), linestyle = '--', color = 'k', linewidth = 1)
axs[1,0].axvline(np.real(field.material.rhop[id_f]/air.rho0), linestyle = '--', color = 'k', linewidth =1)
axs[1,1].axvline(np.imag(field.material.rhop[id_f]/air.rho0), linestyle = '--', color = 'k', linewidth = 1)
# PDE 2D
ba.plot_multi_2d_kde(cmap = 'Blues', mode = 'contour', limit_to_ci = False)

#%% Alpha vs angle for a single freq
_, alpha_calc = dcism_b.recon_vp_nlr_1layer(controls.k0[id_f], air.rho0, 
                                            ba.mean_values, np.deg2rad(field.material.theta_deg))
_, alpha_calc_l = dcism_b.recon_vp_nlr_1layer(controls.k0[id_f], air.rho0, 
                                             ba.ci[0,:], np.deg2rad(field.material.theta_deg))
_, alpha_calc_u = dcism_b.recon_vp_nlr_1layer(controls.k0[id_f], air.rho0, 
                                              ba.ci[1,:], np.deg2rad(field.material.theta_deg))

# Plot absorption
_, ax = ut_is.give_me_an_ax()
ut_is.plot_absorption_theta(field.material.theta_deg, field.material.alpha_ft[id_f,:], 
                            ax = ax[0,0], color = 'k', linewidth = 1.5, linestyle = '--',
                            alpha = 1.0, label = "JCA")
ut_is.plot_absorption_theta(field.material.theta_deg, alpha_calc, 
                            ax = ax[0,0], color = 'm', linewidth = 1.5, linestyle = '-',
                            alpha = 0.7, label = "Bayesian")
ax[0,0].fill_between(field.material.theta_deg, alpha_calc_l, alpha_calc_u,
                     color='m', alpha=0.3)
plt.tight_layout()

#%% Run SPK estimation
dcism_b.nested_sampling_spk()
#%% Reconstructions
dcism_b.get_kp_spk()
dcism_b.get_rhop_spk()
dcism_b.get_zp_spk()
dcism_b.get_vp_nlr_spk(theta = np.deg2rad(field.material.theta_deg))
#%% Plots kp, rhop, Zp,
_, ax = ut_is.give_me_an_ax(figshape = (1,2), figsize = (7,3))
ax[0,0].semilogx(controls.freq, np.real(material.kp), '--k')
ax[0,1].semilogx(controls.freq, np.imag(material.kp), '--k')
ax = dcism_b.plot_kp(ax = ax, color = 'olivedrab')

_, ax = ut_is.give_me_an_ax(figshape = (1,2), figsize = (7,3))
ax[0,0].semilogx(controls.freq, np.real(material.rhop), '--k')
ax[0,1].semilogx(controls.freq, np.imag(material.rhop), '--k')
ax = dcism_b.plot_rhop(ax = ax, color = 'olivedrab')

_, ax = ut_is.give_me_an_ax(figshape = (1,2), figsize = (7,3))
ax[0,0].semilogx(controls.freq, np.real(material.Zp), '--k')
ax[0,1].semilogx(controls.freq, np.imag(material.Zp), '--k')
ax = dcism_b.plot_Zp(ax = ax, color = 'olivedrab')

#%% Plots Vp
_, ax = ut_is.give_me_an_ax(figshape = (1,2), figsize = (7,3))
ax[0,0].semilogx(controls.freq, np.real(material.Vp_ft[:, 0]), '--r')
ax[0,1].semilogx(controls.freq, np.imag(material.Vp_ft[:, 0]), '--r')
ax = dcism_b.plot_Vp_spk(ax = ax, color = 'r', jtheta = 0)

ax[0,0].semilogx(controls.freq, np.real(material.Vp_ft[:, 45]), '--g')
ax[0,1].semilogx(controls.freq, np.imag(material.Vp_ft[:, 45]), '--g')
ax = dcism_b.plot_Vp_spk(ax = ax, color = 'g', jtheta = 45)

ax[0,0].semilogx(controls.freq, np.real(material.Vp_ft[:, 75]), '--b')
ax[0,1].semilogx(controls.freq, np.imag(material.Vp_ft[:, 75]), '--b')
ax = dcism_b.plot_Vp_spk(ax = ax, color = 'b', jtheta = 75)

#%% Plot alpha
_, ax = ut_is.give_me_an_ax(figshape = (1,1), figsize = (7,3))
ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 0], '--r')
ax[0,0] = dcism_b.plot_alpha_spk(ax = ax[0,0], color = 'r', jtheta = 0)

ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 30], '--g')
ax[0,0] = dcism_b.plot_alpha_spk(ax = ax[0,0], color = 'g', jtheta = 30)

ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 60], '--b')
ax[0,0] = dcism_b.plot_alpha_spk(ax = ax[0,0], color = 'b', jtheta = 60)

#%% DCISM estimation (Bayesian - LR)
dcism_b_lr = DCISM_Bayesian(p_mtx = field.pres_mtx, controls = controls, air = air, 
                         receivers = receivers, source = source, 
                         sampling_scheme = 'single ellipsoid', enlargement_factor = 1.1)
dcism_b_lr.choose_forward_model(chosen_model = 0)
dcism_b_lr.set_mic_pairs()
dcism_b_lr.setup_dDCISM(T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
dcism_b_lr.set_nested_sampling_parameters(n_live = 50, max_iter = 2000, 
                                          max_up_attempts = 50, seed = 0, dlogz = 0.1,
                                          ci_percent = 99)
#%%
dcism_b_lr.kp_rhop_range(resist = [3000, 20000], phi = [0.8, 0.99], alpha_inf = [1.0, 1.5], 
                      Lam = [100e-6, 300e-6], Lamlfac = [1.01, 2.0],
                      thickness = [2e-3, 20e-2], theta_deg = [0, 75], n_samples = 20000)
lb = dcism_b_lr.lb_mtx[:,id_f]
ub = dcism_b_lr.ub_mtx[:,id_f]
lb, ub = dcism_b_lr.set_prior_limits(lower_bounds = lb, upper_bounds = ub)
#%%
ba = dcism_b_lr.nested_sampling_single_freq(lb, ub, jf = id_f)
print(r"log(Z) = {:.2f} +/- {:.2f}".format(ba.logZ, ba.logZ_err))
#%%
ba.plot_loglike()
axs = ba.plot_smooth_marginal_posterior(figshape = (1, 2), figsize = (6,3))
ba.plot_multi_2d_kde(cmap = 'Blues', mode = 'contour', limit_to_ci = True)
#%%
_, alpha_calc = dcism_b_lr.recon_vp_lr(controls.k0[id_f], ba.mean_values, 
                                       np.deg2rad(field.material.theta_deg))
_, alpha_calc_l = dcism_b_lr.recon_vp_lr(controls.k0[id_f], ba.ci[0,:], 
                                         np.deg2rad(field.material.theta_deg))
_, alpha_calc_u = dcism_b_lr.recon_vp_lr(controls.k0[id_f], ba.ci[1,:], 
                                         np.deg2rad(field.material.theta_deg))

#%% Plot absorption
_, ax = ut_is.give_me_an_ax()
ut_is.plot_absorption_theta(field.material.theta_deg, field.material.alpha_ft[id_f,:], 
                            ax = ax[0,0], color = 'k', linewidth = 1.5, linestyle = '--',
                            alpha = 1.0, label = "JCA")
ut_is.plot_absorption_theta(field.material.theta_deg, alpha_calc, 
                            ax = ax[0,0], color = 'm', linewidth = 1.5, linestyle = '-',
                            alpha = 0.7, label = "Bayesian")
ax[0,0].fill_between(field.material.theta_deg, alpha_calc_l, alpha_calc_u,
                     color='grey', alpha=0.3)
plt.tight_layout()

#%% Run SPK estimation
dcism_b_lr.nested_sampling_spk()
dcism_b_lr.get_vp_lr_spk(theta = np.deg2rad(field.material.theta_deg))

#%% Plot logZ
_, ax = ut_is.give_me_an_ax(figshape = (1,1), figsize = (7,3))
dcism_b.plot_logZ_spk(ax = ax[0, 0], color = 'b')
dcism_b_lr.plot_logZ_spk(ax = ax[0, 0], color = 'g')
ax[0, 0].set_ylim((-10,120))
#%% Plot alpha
_, ax = ut_is.give_me_an_ax(figshape = (1,1), figsize = (7,3))
ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 0], '--r')
ax[0,0] = dcism_b_lr.plot_alpha_spk(ax = ax[0,0], color = 'r', jtheta = 0)

ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 30], '--g')
ax[0,0] = dcism_b_lr.plot_alpha_spk(ax = ax[0,0], color = 'g', jtheta = 30)

ax[0,0].semilogx(controls.freq, material.alpha_ft[:, 60], '--b')
ax[0,0] = dcism_b_lr.plot_alpha_spk(ax = ax[0,0], color = 'b', jtheta = 60)