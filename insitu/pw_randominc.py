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
from insitu.parray_estimation import PArrayDeduction

#%% step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# step 2 - set your controls
controls = AlgControls(c0 = air.c0, freq_init = 100, freq_step = 50, freq_end=3000)
# controls = AlgControls(c0 = air.c0, freq_vec = [1000])

#%% step 4 - set the sources
sources = Source()
sources.set_ssph_sources(ns=2000, radius=1.0)
#%% step 5  - set the receivers
receivers = Receiver()
##### 3D Regular array
receivers.double_planar_array(x_len=0.175, y_len=0.175, n_x = 8, n_y = 8)
# receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.brick_array(x_len=1.2, y_len=1.6, z_len=0.5, n_x = 8, n_y = 8, n_z=3)
##### 3D Random array
# receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 290, zr = 0.1)

#%% step 7 - run/load field
field = PWDifField(air, controls, sources, receivers)
field.p_fps(resistivity = 25000, thickness = 0.025)
# field.plot_scene()
# field.plot_pres()

#%% step 8 - create a deduction object with the loaded field sim
pwdif_ded = PArrayDeduction(field)
pwdif_ded.wavenum_dir(n_waves=2000, plot = False)
pwdif_ded.pk_a3d_tikhonov(method = 'scipy', lambd_value=[])
pwdif_ded.zs(Lx=0.1, Ly=0.1, n_x=1, n_y=1, avgZs=True)

# ff_ded_parray.save(filename='ff_pw_regulara_60cmx80cmx25mm_192m')
# ff_ded_parray.load(filename = 'ff_pw_regulara_60cmx80cmx25mm_192m')
#%% plots
pwdif_ded.plot_pk_sphere(freq=1000, db=False)
pwdif_ded.plot_pk_sphere(freq=1000, db=True, dinrange=15, save=False)
plt.show()

#%%
compare_alpha(
    {'freq': pwdif_ded.controls.freq, 'array': pwdif_ded.alpha, 'color': 'red', 'linewidth': 2})