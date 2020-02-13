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
from insitu.field_free import FreeField
from insitu.field_qterm import LocallyReactiveInfSph
from insitu.parray_estimation import PArrayDeduction

# import impedance-py/C++ module
# import insitu_cpp
#%%
# step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# step 2 - set your controls
# controls = AlgControls(c0 = air.c0, freq_init = 500, freq_step = 500, freq_end=2000)
controls = AlgControls(c0 = air.c0, freq_vec = [1000])

#%% step 4 - set the sources
sources = Source(coord = [0, 0, 50])
# step 5  - set the receivers
receivers = Receiver()
##### 3D Regular array
receivers.double_planar_array(x_len=0.6, y_len=0.8, n_x = 8, n_y = 8)
# receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.brick_array(x_len=1.2, y_len=1.6, z_len=0.5, n_x = 8, n_y = 8, n_z=3)
##### 3D Random array
# receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 192, zr = 0.1)

# #%% step 7 - run/load field
field = FreeField(air, controls, sources, receivers)
# field.plot_scene()
# field.monopole_ff()
field.planewave_ff(theta = 0, phi = 0)
# field.add_noise(snr = 3)
# field.mirrorsource()
# # field.plot_pres()
# # field.save(filename='bemflush_Lx_1.0m_Ly_1.0m_3darray')

#%% step 8 - create a deduction object with the loaded field sim
ff_ded_parray = PArrayDeduction(field)
ff_ded_parray.wavenum_dir(n_waves=2000, plot = False)
# ff_ded_parray.pk_a3d_constrained(xi = 0.1)
ff_ded_parray.pk_a3d_tikhonov(method = 'direct', lambd_value=[])
# ff_ded_parray.save(filename='ff_pw_regulara_60cmx80cmx25mm_192m')
# ff_ded_parray.load(filename = 'ff_pw_regulara_60cmx80cmx25mm_192m')
#%% plots
# ff_ded_parray.plot_pk_sphere(freq=100)
# ff_ded_parray.plot_pk_sphere(freq=500)
ff_ded_parray.plot_pk_sphere(freq=1000, db=False)
# ff_ded_parray.plot_pk_sphere(freq=2000, db=False)
ff_ded_parray.plot_pk_sphere(freq=1000, db=True, dinrange=30, save=False)
# ff_ded_parray.plot_pk_sphere(freq=2000)
plt.show()

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

#%% Save to mat file
# import scipy.io as sio
# receivers_m = receivers.coord
# sio.savemat('receivers.mat', {'receivers_m':receivers_m})
# freq_m = controls.freq
# sio.savemat('freq.mat', {'freq_m':freq_m})
# c0 = air.c0
# sio.savemat('c0.mat', {'c0':c0})
# dir_m = ff_ded_parray.dir
# sio.savemat('directions.mat', {'dir_m':dir_m})
# p_m = field.pres_s
# sio.savemat('pm.mat', {'p_m':p_m})
