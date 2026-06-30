# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:20:15 2026

@author: Eric Brandao
"""
import numpy as np
import matplotlib.pyplot as plt
from esm_methods import ESM
from decomposition_ev_ig import DecompositionEv2
from sources import Source
import scipy
#%% Import diffuser simulation
folder = 'D:/Work/UFSM/Pesquisa/Diffusers_NAH/Comsol simulations/Difusores_DTU_df20Hz/'
mat_1sph = scipy.io.loadmat(folder + 'comsol_1sphere.mat')
diff_geometry = {'coords': mat_1sph['mesh_coords'], 'connectivities': mat_1sph['mesh_connectivity']-1}

#%% Load PWE from simulation
study_foder = "D:/Work/UFSM/Pesquisa/Diffusers_NAH/Exp_Data_Caroline_DTU/PPRO_ERIC/simulation_study/"
pwe = DecompositionEv2()
pwe.load(path = study_foder, filename = 'BEM_holography_1sphere_snr30.pkl')

source = Source(coord = [0, 0, 1.0])
#%%
esm = ESM(source = source, receivers = pwe.receivers, 
          geometry = diff_geometry, bbox_size = [0.4, 0.25, 0.25],
          rec_bbox = False)
# eq_sources = esm.sample_rec_face(dx = 0.08, faceid = 2)
# eq_sources = esm.sample_rec_bbox()
eq_sources = esm.sample_cyl_bbox(dx = 0.05, while_step = 100)

esm.sample_source(source_radii = 0.1, sampling_scheme = 'dodecahedron')
#%%
# esm.plotly_scene(eq_sources=eq_sources)
esm.plotly_scene(eq_sources=eq_sources, plot_bbox=True)



