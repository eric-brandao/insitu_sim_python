# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:45:20 2026

@author: Win11
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AirProperties, AlgControls
from sources import Source
# from receivers import Receiver
from material import PorousAbsorber  # Material
#from field_inf_nlr import NLRInfSph  # Field Inf NLR
from directDCISM import dDCISM
# from decomp_quad_v2 import Decomposition_QDT
from decomp_DCISM_bayesian import DCISM_Bayesian
# import lcurve_functions as lc
import utils_insitu as ut_is
#%% Exp DCISM
exp_folder = "D:/Work/UFSM/Pesquisa/insitu_arrays/DCISM/BayesianDCISM/experimental_study/ba_est/"
dcism_b_4_exp = DCISM_Bayesian()
dcism_b_4_exp.load(path = exp_folder, filename = 'ba_exp_model40_PET_s30cm')
dcism_b_4_exp.kp_rhop_range_miki(resist = [500, 60000], n_samples = 20000)
# sim_folder = "D:/Work/UFSM/Pesquisa/insitu_arrays/DCISM/BayesianDCISM/simulation_study/ba_est/"
# dcism_b_4_sim = DCISM_Bayesian()
# dcism_b_4_sim.load(path = sim_folder, filename = 'ba_dcism_model4_PET50mm_S50_DLA')
#%% Air, Controls, Material
air = AirProperties(c0 = 343.0, rho0 = 1.21)
controls = AlgControls(c0 = air.c0, freq_vec = np.arange(100, 2000, 100))
thickness =  50/1000
material = PorousAbsorber(air, controls)
material.jcal(resistivity = 12200, porosity = 0.90, tortuosity = 1.00, 
              lam = 362.1e-6, lam_l = 362.2e-6)
material.layer_over_rigid_theta(thickness = thickness, theta_end = 89)
material.layer_over_rigid(thickness = thickness, theta = 0.0);

#%% Sound field sim
source = Source(coord = [0,0, 0.3])
field = dDCISM(air = air, controls = controls, source = source,
                          receivers = dcism_b_4_exp.receivers, material = material,
                          T0 = 7.5, dt = 0.1, tol = 1e-10, gamma=1.0) # ppro_obj.meas_obj.source
field.predict_p_spk_nlr_layer();
field.add_noise(snr = 30)
#%% DCISM estimation (Bayesian) - NLR
dcism_b = DCISM_Bayesian(p_mtx = field.pres_mtx, controls = field.controls, air = field.air, 
                         receivers = field.receivers, source = field.source,
                         sampling_scheme = 'single ellipsoid', enlargement_factor = 1.0)
dcism_b.choose_forward_model(chosen_model = 40)
dcism_b.set_mic_pairs()
dcism_b.setup_dDCISM(T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0)
dcism_b.set_sample_thickness(t_p = field.material.thickness)
dcism_b.set_nested_sampling_parameters(n_live = 50, max_iter = 2000, max_up_attempts = 100, 
                                       seed = 0, dlogz = 1, ci_percent = 99)
# dcism_b.kp_rhop_range(resist = [2000, 60000], phi = [0.8, 0.99], alpha_inf = [1.0, 1.5], 
#                       Lam = [100e-6, 300e-6], Lamlfac = [1.01, 2.0], n_samples = 20000)
dcism_b.kp_rhop_range_miki(resist = [500, 80000], n_samples = 20000)

#%%
dcism_b.nested_sampling_spk2(freqs_init = [700, 800, 900], resist_range = 5000, res_factor = 2)

#%% Reconstructions
dcism_b.get_kp_spk()
dcism_b.get_rhop_spk()
dcism_b.get_zp_spk()
dcism_b.get_vp_nlr_spk(theta = np.deg2rad(field.material.theta_deg))

#%%
idf = ut_is.find_freq_index(dcism_b.controls.freq, freq_target=1000)
# true_vals = np.array([np.real(material.kp[idf])/controls.k0[idf], 
#                         np.imag(material.kp[idf])/controls.k0[idf],
#                         np.real(material.rhop[idf])/air.rho0,
#                         np.imag(material.rhop[idf])/air.rho0])
true_vals = np.array([np.real(material.kp[idf]), 
                        np.imag(material.kp[idf]),
                        np.real(material.rhop[idf]),
                        np.imag(material.rhop[idf])])
scale = np.array([controls.k0[idf], controls.k0[idf], air.rho0, air.rho0])

dcism_b.ba_list[idf].compute_statistics(ci_percent = 99.999999)
dcism_b.ba_list[idf].plot_multi_2d_kde(true_vals = true_vals, limit_to_ci=True, 
                                       mode = 'contour', color_true_vals = 'tomato',
                                       scale = scale)
#%%
ax = ut_is.plot_spk_re_imag(controls.freq, material.kp, xlims = None, ylims = None, 
              color = 'k', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'True')
# ax = ut_is.plot_spk_re_imag(dcism_b_4_exp.controls.freq, dcism_b_4_exp.kp_mean, ax, 
#               color = 'tab:blue', linewidth = 1.5, linestyle = '--',
#               alpha = 1.0, label = 'measured')
ax = ut_is.plot_spk_re_imag(dcism_b.controls.freq, dcism_b.kp_mean, ax, 
              color = 'tab:green', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'simulated')
ax[0,0].fill_between(dcism_b_4_exp.controls.freq, 
                     dcism_b_4_exp.controls.k0*dcism_b_4_exp.lb_mtx[0,:], 
                     dcism_b_4_exp.controls.k0*dcism_b_4_exp.ub_mtx[0,:], 
                     color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(dcism_b_4_exp.controls.freq, 
                     dcism_b_4_exp.controls.k0*dcism_b_4_exp.lb_mtx[1,:], 
                     dcism_b_4_exp.controls.k0*dcism_b_4_exp.ub_mtx[1,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,0].fill_between(controls.freq, controls.k0*dcism_b.lb_mtx[0,:], 
                     controls.k0*dcism_b.ub_mtx[0,:], color = 'crimson', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(controls.freq, controls.k0*dcism_b.lb_mtx[1,:], 
                     controls.k0*dcism_b.ub_mtx[1,:], color = 'crimson', alpha = 0.2, 
                     edgecolor = 'grey')

ax[0,0].set_ylim((0,80))
ax[0,1].set_ylim((-25, 0))
#%%
ax = ut_is.plot_spk_re_imag(controls.freq, material.rhop, xlims = None, ylims = None, 
              color = 'k', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'True')
# ax = ut_is.plot_spk_re_imag(controls.freq, air.rho0*np.ones(len(controls.freq)), ax,
#               color = 'r', linewidth = 1.5, linestyle = '--',
#               alpha = 1.0, label = 'air')
# ax = ut_is.plot_spk_re_imag(dcism_b_4_exp.controls.freq, dcism_b_4_exp.rhop_mean, ax, 
#               color = 'tab:blue', linewidth = 1.5, linestyle = '--',
#               alpha = 1.0, label = 'measured')
ax = ut_is.plot_spk_re_imag(dcism_b.controls.freq, dcism_b.rhop_mean, ax, 
              color = 'tab:green', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'simulated')
ax[0,0].fill_between(dcism_b_4_exp.controls.freq, 
                     air.rho0*dcism_b_4_exp.lb_mtx[2,:], 
                     air.rho0*dcism_b_4_exp.ub_mtx[2,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(dcism_b_4_exp.controls.freq, 
                     air.rho0*dcism_b_4_exp.lb_mtx[3,:], 
                     air.rho0*dcism_b_4_exp.ub_mtx[3,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,0].fill_between(controls.freq, air.rho0*dcism_b.lb_mtx[2,:], 
                     air.rho0*dcism_b.ub_mtx[2,:], color = 'crimson', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(controls.freq, air.rho0*dcism_b.lb_mtx[3,:], 
                     air.rho0*dcism_b.ub_mtx[3,:], color = 'crimson', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,0].set_ylim((0,5))
ax[0,1].set_ylim((-20,1))