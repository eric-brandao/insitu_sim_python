# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:09:36 2025

@author: Eric Brandao - testing array pre-processing (remove, truncate, etc)
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from controlsair import AirProperties #AlgControls, 
from receivers import Receiver
from field_inf_nlr import NLRInfSph  # Field Inf NLR
# import lcurve_functions as lc
import utils_insitu as ut_is

#%% Import sound field for processing with ISM method
field_nlr = NLRInfSph()
path = 'D:/Work/UFSM/Pesquisa/insitu_arrays/TAMURA_DCISM/dcism_locallyreacting_Rtheta/'
field_nlr.load(path = path, filename = "melamine_nlr_sim")

#%%
new_array, new_pres_data = field_nlr.receivers.remove_z_coords(z = 0.02, 
                                                               pres_data = field_nlr.pres_s[0])

id_z_list = new_array.get_micpair_indices()
#%% 
new_array, new_pres_data = field_nlr.receivers.remove_z_coords(z = 0.1, 
                                                               pres_data = field_nlr.pres_s[0])

#%%
new_array, new_pres_data = field_nlr.receivers.truncate_array_radially(radius = 0.5, 
                                                                       pres_data = field_nlr.pres_s[0])

#%%
new_array, new_pres_data = field_nlr.receivers.truncate_array_ymax(y_max = 0.1, 
                                                                   pres_data = field_nlr.pres_s[0])

#%% 
new_array.plot_array(z_lim = [0, 0.2])

#%%
id_f = ut_is.find_freq_index(field_nlr.controls.freq, freq_target=1000)
fig, axs = plt.subplots(1, 1, figsize = (6, 3), subplot_kw={"projection": "3d"})
ut_is.plot_map_scatter_ax(axs, new_array.coord, new_pres_data[:, id_f], 
                          freq =field_nlr.controls.freq[id_f])