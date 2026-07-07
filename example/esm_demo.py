# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:20:15 2026

@author: Eric Brandao
"""
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AlgControls, AirProperties
import lcurve_functions as lc
import utils_insitu as ut_is
from esm_methods import ESMBase, RISESM
from decomposition_ev_ig import DecompositionEv2
from sources import Source
from receivers import Receiver
import scipy
#%% Import diffuser simulation
folder = 'D:/Work/UFSM/Pesquisa/Diffusers_NAH/Comsol simulations/Difusores_DTU_df20Hz/'
mat_1sph = scipy.io.loadmat(folder + 'comsol_1sphere.mat')
diff_geometry = {'coords': mat_1sph['mesh_coords'], 'connectivities': mat_1sph['mesh_connectivity']-1}

#%% Load PWE from simulation
study_foder = "D:/Work/UFSM/Pesquisa/Diffusers_NAH/Exp_Data_Caroline_DTU/PPRO_ERIC/simulation_study/"
pwe = DecompositionEv2()
pwe.load(path = study_foder, filename = 'BEM_holography_1sphere_snr30.pkl')
air = AirProperties(c0 = 343.0, rho0 = 1.21)
source = Source(coord = [0, 0, 2.45])
#%% Initialize ang get a bounding box for the scattering/radiating equi. sources
esm = RISESM(p_mtx = pwe.pres_s, controls = pwe.controls, air = air,
            source = source, receivers = pwe.receivers, geometry = diff_geometry,
            regu_par = 'gcv')

# esm.rectangular_bbox(bbox_size = [0.4, 0.25, 0.25])
esm.cylindrical_bbox(bbox_size = [0.4, 0.25])

#%% Sample the bounding box and source
esm.sample_bbox(dx = esm.get_dx(n_s_lam = 4, freq = esm.controls.freq[-1])) #esm.get_dx(n_s_lam = 4, freq = esm.controls.freq[-1])
esm.sample_source(source_radii = 0.01, sampling_scheme = 'octahedron')
#%% Plot the scene
esm.plotly_scene(plot_bbox=True)

#%% Solve single freq
idf = ut_is.find_freq_index(esm.controls.freq, 800)
rm_rs = esm.get_rm_rs_dist()
rm_rq = esm.get_rm_rq_dist(esm.receivers)
cosmq = esm.get_cosmq_dist(esm.receivers, rm_rq)
g_mtx, cond_vec = esm.get_sens_mtx(esm.controls.k0[idf], rm_rs, rm_rq, cosmq)
cond_mtx = np.diag(cond_vec)
g_mtx_cond = g_mtx @ cond_mtx
x,_ = esm.solve_freq(g_mtx_cond, esm.pres_s[:,idf], method='tikhonov', plot_reg_curve = True)
#%% Freq loop
esm.pk_ff_rigid_static()
#%%
semi_sphere = Receiver()
semi_sphere.hemispherical_array(radius = 100, n_rec_target = 642)

#%%
esm.recon_ps_static(semi_sphere)
esm.octave_avg()

#%%
fig, _ = esm.plot_directivity(semi_sphere, freq = 2000, dinrange = 45)
fig.show()


