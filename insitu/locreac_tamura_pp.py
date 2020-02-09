#%%
# import general python modules
import math
import numpy as np
import matplotlib.pyplot as plt
import toml

# import impedance-python modules
from insitu.controlsair import AlgControls, AirProperties, load_cfg
from insitu.controlsair import plot_spk, compare_spk, compare_alpha
from insitu.material import PorousAbsorber
from insitu.sources import Source
from insitu.receivers import Receiver
from insitu.field_bemflush import BEMFlush
from insitu.field_qterm import LocallyReactiveInfSph
from insitu.parray_estimation import PArrayDeduction

# import impedance-py/C++ module
# import insitu_cpp
#%%
# step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# step 2 - set your controls
controls = AlgControls(c0 = air.c0, freq_step = 50, freq_end=2000)
# step 3 - set the boundary condition / absorber
material = PorousAbsorber(air, controls)
# material.jcal(resistivity = 10000)
material.delany_bazley(resistivity=10000)
material.layer_over_rigid(thickness = 0.04,theta = 0)
# material.plot_absorption()
#%%
# step 4 - set the sources
sources = Source(coord = [0.0, 0.0, 0.3])
# step 5  - set the receivers
receivers = Receiver()
receivers.double_planar_array(x_len=.15, y_len=0.15, n_x = 8, n_y = 8)

#%% step 7 - load field
saved_field = BEMFlush(air, controls, material, sources, receivers)
# saved_field.load(filename = 'my_bemflush_Lx_2.0m_Ly_2.0m')
# print('pres_s: ({}, {})'.format(len(saved_field.pres_s), len(saved_field.pres_s[0])))
# saved_field.plot_pres()
# saved_field.generate_mesh(Lx = 0.5, Ly = 0.5, Nel_per_wavelenth=3)

# saved_field.receivers = receivers
# # saved_field.plot_scene(mesh = False)
# saved_field.p_fps()
# saved_field.save(filename='my_bemflush_Lx_2.0m_Ly_2.0m_array64')

saved_field.load(filename = 'my_bemflush_Lx_2.0m_Ly_2.0m_array64')
saved_field.plot_scene(mesh = False)
saved_field.plot_pres()
#%% step 8 - create a deduction object with the loaded field sim
zs_ded_parray = PArrayDeduction(saved_field)
zs_ded_parray.tamura_pp()

#%% plotting stuff #################
# compare_alpha(controls.freq,
#     {'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'Plane Wave': zs_ded_qterm.alpha_pw_pu, 'color': 'grey', 'linewidth': 1},
#     {'PWA': zs_ded_qterm.alpha_pwa_pu, 'color': 'green', 'linewidth': 1},
#     {'q-term': zs_ded_qterm.alpha_q_pu, 'color': 'red', 'linewidth': 1})

#%% Compare spectrums with infinite calcls - if done
# compare_spk(controls.freq, {'BEM': field.pres_s[0][0]},
#     {'Inf': field_inf.pres_s[0][0]}, ref = 20e-6)
# compare_spk(controls.freq, {'BEM': field.uz_s[0][0]},
#     {'Inf': field_inf.uz_s[0][0]}, ref = 5e-8)
