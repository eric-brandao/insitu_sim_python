#%% import general python modules
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
from insitu.field_diffuse_pw import PWDifField
from insitu.field_pw import PWField
from insitu.parray_estimation import PArrayDeduction
from insitu.qterm_estimation import ImpedanceDeductionQterm

#%% step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# step 2 - set your controls
controls = AlgControls(c0 = air.c0, freq_init = 100, freq_step = 50, freq_end=3000)
# controls = AlgControls(c0 = air.c0, freq_vec = [1000])
theta = 15
res = 25000
thick = 0.025
material = PorousAbsorber(air, controls)
material.delany_bazley(resistivity = res)
material.layer_over_rigid(thickness = thick, theta = np.deg2rad(theta))

#%% step 4 - set the sources
sources = Source()
sources.set_ssph_sources(ns=10000, radius=1.0)
#%% step 5  - set the receivers
receivers = Receiver(coord = [0.0, 0.0, 0.01])
# receivers.double_rec(z_dist = 0.01)
##### 2 layer array
# receivers.double_planar_array(x_len=0.175, y_len=0.175, n_x = 8, n_y = 8)
##### 3D Random array
# receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.brick_array(x_len=1.2, y_len=1.6, z_len=0.5, n_x = 8, n_y = 8, n_z=3)
receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 290, zr = 0.01)

#%% step 7 - run/load field
field = PWDifField(air, controls, sources, receivers)
# field.p_fps(resistivity = res, thickness = thick, locally=True, randomize=True, seed=0)
# field.plot_scene()
# field.plot_pres()

# field = PWField(air, controls, material, receivers, theta = 0, phi = 0)
# field.p_fps()
#%% step 8 - create a deduction object with the loaded field sim
pwdif_ded = PArrayDeduction(field)
# pwdif_ded.wavenum_dir(n_waves=2000, plot = False)
# pwdif_ded.pk_tikhonov(method = 'scipy', lambd_value=[])
# pwdif_ded.plot_pk_sphere(freq=1000, db=False)
# pwdif_ded.plot_pk_sphere(freq=1000, db=True, dinrange=30, save=False)
# pwdif_ded.save('pk_difield_res25000_d10cm_loc_rand')
pwdif_ded.load('pk_difield_res25000_d25mm_nonloc_norand')
pwdif_ded.pk_interpolate(npts=10000)
pwdif_ded.plot_pk_map(freq=1000, db=True, dinrange=20, save=False)
pwdif_ded.alpha_from_array2(desired_theta=theta, target_range=3)
pwdif_ded.zs(Lx=0.1, Ly=0.1, n_x=1, n_y=1, theta = np.deg2rad(theta), avgZs=True)

# ff_ded_parray.save(filename='ff_pw_regulara_60cmx80cmx25mm_192m')
# ff_ded_parray.load(filename = 'ff_pw_regulara_60cmx80cmx25mm_192m')
#%% plots
# pwdif_ded.plot_pk_sphere(freq=1000, db=False)
# pwdif_ded.plot_pk_sphere(freq=1000, db=True, dinrange=15, save=False)
# plt.show()

#%% Deduction Otsuru
# zs = ImpedanceDeductionQterm(field)
# zs.z_eanoise_pp()
# zs.pw_pp()

#%%
compare_alpha({'freq': material.freq, 'ref': material.alpha, 'color': 'black', 'linewidth': 4},
    {'freq': pwdif_ded.controls.freq, 'avg alpha': pwdif_ded.alpha_avg, 'color': 'red', 'linewidth': 2},
    {'freq': pwdif_ded.controls.freq, 'backpropag to zs': pwdif_ded.alpha, 'color': 'blue', 'linewidth': 2})
    

# compare_alpha({'freq': material.freq, 'ref': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs.controls.freq, 'pp eanoise': zs.alpha_ea_pp, 'color': 'red', 'linewidth': 3},
#     {'freq': zs.controls.freq, 'pp regular': zs.alpha_pw_pp, 'color': 'blue', 'linewidth': 2}, freq_max=7000)