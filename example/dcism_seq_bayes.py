# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:45:20 2026

@author: Win11
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AirProperties, AlgControls
# from sources import Source
# from receivers import Receiver
from material import PorousAbsorber  # Material
#from field_inf_nlr import NLRInfSph  # Field Inf NLR
# from directDCISM import dDCISM
# from decomp_quad_v2 import Decomposition_QDT
from decomp_DCISM_bayesian import DCISM_Bayesian
# import lcurve_functions as lc
import utils_insitu as ut_is
#%% Exp DCISM
exp_folder = "D:/Work/UFSM/Pesquisa/insitu_arrays/DCISM/BayesianDCISM/experimental_study/ba_est/"
dcism_b_4_s30 = DCISM_Bayesian()
dcism_b_4_s30.load(path = exp_folder, filename = 'ba_exp_dcism_model4_PET50mm_s30cm')

sim_folder = "D:/Work/UFSM/Pesquisa/insitu_arrays/DCISM/BayesianDCISM/simulation_study/ba_est/"
dcism_b_4_s30_s = DCISM_Bayesian()
dcism_b_4_s30_s.load(path = sim_folder, filename = 'ba_dcism_model4_PET50mm_S30_DLA')
#%% Air, Controls, Material
air = AirProperties(c0 = 343.0, rho0 = 1.21)
controls = AlgControls(c0 = air.c0, freq_vec = np.arange(100, 4000, 100))
thickness =  50/1000
material = PorousAbsorber(air, controls)
material.jcal(resistivity = 25000, porosity = 0.90, tortuosity = 1.00, 
              lam = 362.1e-6, lam_l = 362.2e-6)
material.layer_over_rigid_theta(thickness = thickness, theta_end = 89)
material.layer_over_rigid(thickness = thickness, theta = 0.0);

#%% DCISM estimation (Bayesian) - NLR
dcism_b = DCISM_Bayesian(controls = controls, air = air)
dcism_b.choose_forward_model(chosen_model = 4)
dcism_b.kp_rhop_range(resist = [1000, 60000], phi = [0.8, 0.99], alpha_inf = [1.0, 1.5], 
                      Lam = [50e-6, 500e-6], Lamlfac = [1.0, 2.0], n_samples = 20000)
# dcism_b.kp_rhop_range(resist = [1000, 60000], phi = [0.89, 0.91], alpha_inf = [1.0, 1.01], 
#                       Lam = [360e-6, 365e-6], Lamlfac = [1.0, 1.05], n_samples = 20000)

#%%
lb_mtx, ub_mtx = dcism_b_4_s30_s.nested_sampling_spk_seq(start_freq = 1000, fac = 5)

#%%
ax = ut_is.plot_spk_re_imag(controls.freq, material.kp, xlims = None, ylims = None, 
              color = 'k', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'True')
ax = ut_is.plot_spk_re_imag(dcism_b_4_s30.controls.freq, dcism_b_4_s30.kp_mean, ax, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'measured')
ax = ut_is.plot_spk_re_imag(dcism_b_4_s30_s.controls.freq, dcism_b_4_s30_s.kp_mean, ax, 
              color = 'tab:green', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'simulated')
ax[0,0].fill_between(controls.freq, controls.k0*dcism_b.lb_mtx[0,:], 
                     controls.k0*dcism_b.ub_mtx[0,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(controls.freq, controls.k0*dcism_b.lb_mtx[1,:], 
                     controls.k0*dcism_b.ub_mtx[1,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,0].set_ylim((np.amin(controls.k0*dcism_b.lb_mtx[0,:]), 
                  np.amax(controls.k0*dcism_b.ub_mtx[0,:])))
ax[0,1].set_ylim((np.amin(controls.k0*dcism_b.lb_mtx[1,:]), 
                  np.amax(controls.k0*dcism_b.ub_mtx[1,:])))
#%%
ax = ut_is.plot_spk_re_imag(controls.freq, material.rhop, xlims = None, ylims = None, 
              color = 'k', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'True')
# ax = ut_is.plot_spk_re_imag(controls.freq, air.rho0*np.ones(len(controls.freq)), ax,
#               color = 'r', linewidth = 1.5, linestyle = '--',
#               alpha = 1.0, label = 'air')
ax = ut_is.plot_spk_re_imag(dcism_b_4_s30.controls.freq, dcism_b_4_s30.rhop_mean, ax, 
              color = 'tab:blue', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'measured')
ax = ut_is.plot_spk_re_imag(dcism_b_4_s30_s.controls.freq, dcism_b_4_s30_s.rhop_mean, ax, 
              color = 'tab:green', linewidth = 1.5, linestyle = '--',
              alpha = 1.0, label = 'simulated')
ax[0,0].fill_between(controls.freq, air.rho0*dcism_b.lb_mtx[2,:], 
                     air.rho0*dcism_b.ub_mtx[2,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,1].fill_between(controls.freq, air.rho0*dcism_b.lb_mtx[3,:], 
                     air.rho0*dcism_b.ub_mtx[3,:], color = 'grey', alpha = 0.2, 
                     edgecolor = 'grey')
ax[0,0].set_ylim((np.amin(air.rho0*dcism_b.lb_mtx[2,:]), 
                  np.amax(air.rho0*dcism_b.ub_mtx[2,:])))
ax[0,1].set_ylim((np.amin(air.rho0*dcism_b.lb_mtx[3,:]), 
                  np.amax(air.rho0*dcism_b.ub_mtx[3,:])))