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
from insitu.field_pw import PWField
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
# controls = AlgControls(c0 = air.c0, freq_init=100, freq_step = 25, freq_end=6000)
controls = AlgControls(c0 = air.c0, freq_vec = [1000])

# step 3 - set the boundary condition / absorber
theta = np.deg2rad(70) # Angle of incidence in degrees
phi = np.deg2rad(0) # Azimuth
material = PorousAbsorber(air, controls)
# material.jcal(resistivity = 10000)
material.delany_bazley(resistivity=10900)
material.layer_over_rigid(thickness = 0.025,theta = theta)

# material.plot_absorption()
#%%
# # step 5  - set the receivers
receivers = Receiver()
# receivers.double_planar_array(x_len=.175, y_len=0.175, n_x = 8, n_y = 8, dz = 0.029, zr = 0.013)
# receivers.double_planar_array(x_len = 0.5, y_len=0.5, n_x = 8, n_y = 8, dz = 0.1, zr = 0.1)
# receivers.brick_array(x_len = 0.6, y_len = 0.8, z_len = 0.12, n_x = 8, n_y = 8, n_z = 3, zr = 0.1)
receivers.random_3d_array(x_len = 0.6, y_len = 0.8, z_len = 0.25, n_total= 290, zr = 0.1)

# # receivers.planar_array(x_len=.5, y_len=0.5, zr=0.1, n_x = 8, n_y = 8)
#%% step 7 - run/load field
### to run
field = PWField(air, controls, material, receivers, theta = theta, phi = phi)
field.p_fps()
# field.plot_scene()
#%% step 8 - create a deduction object with the loaded field sim
zs_ded_parray = PArrayDeduction(field)
zs_ded_parray.wavenum_dir(n_waves=2000, plot = False)
zs_ded_parray.pk_tikhonov(method='scipy', lambd_value=[])
zs_ded_parray.plot_pk_sphere(freq=1000, db=False)
zs_ded_parray.plot_pk_sphere(freq=1000, db=True, dinrange=20, save=False)
zs_ded_parray.pk_interpolate(npts=10000)
zs_ded_parray.plot_pk_map(freq=1000, db=True, dinrange=20, save=False)
# zs_ded_parray.plot_pk_sphere_interp(freq=1000, db=True, dinrange=50, save=False)
zs_ded_parray.plot_flat_pk(freq = 1000)
# zs_ded_parray.pk_cs(method = 'cvxpy')
# zs_ded_parray.plot_flat_pk(freq = 1000)
# zs_ded_parray.plot_pk_sphere(freq=1000, db=False)
# zs_ded_parray.plot_pk_sphere(freq=1000, db=True, dinrange=30, save=False)
# zs_ded_parray.save(filename='pk_Lx_100cm_Ly_100cm_3darray_theta0_s150cm_a2')

# zs_ded_parray.load(filename='pk_Lx_100cm_Ly_100cm_3darray_theta0_s150cm_a2')
# zs_ded_parray.plot_pk_sphere(freq=500, db=True, dinrange=20)
# zs_ded_parray.plot_pk_sphere(freq=1000, db=True, dinrange=20)
# zs_ded_parray.plot_pk_sphere(freq=2000, db=True, dinrange=20)
plt.show()

import scipy.io as sio
dir_m = zs_ded_parray.dir
sio.savemat('directions.mat', {'dir_m':dir_m})
p_k = zs_ded_parray.pk[:,0]
sio.savemat('pm.mat', {'pk':p_k})


# #%% plotting stuff #################
# zs_ded_parray.load(filename='pk_inf_2parray_theta60_d10cm')
# alpha1 = zs_ded_parray.zs(Lx = 0.1, Ly = 0.1, n_x = 1, n_y = 1)
# alpha2 = zs_ded_parray.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_ded_parray.save(filename='pk_inf_2parray_theta0_d10cm')
# alpha2 = zs_ded_parray.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)

# zs_ded_parray.alpha_from_array(desired_theta=15)
# compare_alpha(
#     {'freq': material.freq, '100 mm sample, 0deg inc': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_ded_parray.controls.freq, 'recovered (21 x 21 pts)': zs_ded_parray.alpha_avg, 'color': 'blue', 'linewidth': 2})


# compare_alpha(
#     {'freq': material.freq, '100 mm sample, 0deg inc': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_ded_parray.controls.freq, 'recovered (21 x 21 pts)': alpha2, 'color': 'blue', 'linewidth': 3},
#     {'freq': zs_ded_parray.controls.freq, 'recovered (1pt)': alpha1, 'color': 'red', 'linewidth': 2})

#%%step 5  - set the receivers
# receivers_reg = Receiver()
# receivers_reg.brick_array(x_len = 0.6, y_len = 0.8, z_len = 0.25, n_x = 9, n_y = 8, n_z = 4, zr = 0.1)

# receivers_rand = Receiver()
# receivers_rand.random_3d_array(x_len = 0.6, y_len = 0.8, z_len = 0.25, n_total= 290, zr = 0.1)

#%% step 7 - run/load field
### to run
# field_reg = PWField( air, controls, material, receivers_reg)
# field_reg.p_fps(theta = theta, phi = phi)
# field_reg.plot_scene()


# field_rand = PWField( air, controls, material, receivers_rand)
# field_rand.p_fps(theta = theta, phi = phi)
# field_rand.plot_scene()
# #%% step 8 - create a deduction object with the loaded field sim
# zs_reg = PArrayDeduction(field_reg)
# zs_reg.wavenum_dir(n_waves=2000, plot = False)
# zs_reg.pk_a3d_tikhonov(method='scipy', lambd_value=[])
# alpha1_reg = zs_reg.zs(Lx = 0.1, Ly = 0.1, n_x = 1, n_y = 1)
# alpha2_reg = zs_reg.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_reg.save(filename='pk_inf_3darray_theta0_d10cmm_reg')

# zs_rand = PArrayDeduction(field_rand)
# zs_rand.wavenum_dir(n_waves=2000, plot = False)
# zs_rand.pk_a3d_tikhonov(method='scipy', lambd_value=[])
# alpha1_rand = zs_rand.zs(Lx = 0.1, Ly = 0.1, n_x = 1, n_y = 1)
# alpha2_rand = zs_rand.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_rand.save(filename='pk_inf_3darray_theta0_d10cmm_rand')

#%% plotting stuff #################
# compare_alpha(
#     {'freq': material.freq, '10 cm sample, 45deg inc': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_reg.controls.freq, 'regular 3d array (288)': alpha1_reg, 'color': 'blue', 'linewidth': 3},
#     {'freq': zs_rand.controls.freq, 'random 3d array (290)': alpha1_rand, 'color': 'red', 'linewidth': 2})

# compare_alpha(
#     {'freq': material.freq, '10 cm sample, 45deg inc': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_reg.controls.freq, 'regular - 21x21 rec pt': alpha2_reg, 'color': 'blue', 'linewidth': 3},
#     {'freq': zs_reg.controls.freq, 'regular - 1 rec pt': alpha1_reg, 'color': 'red', 'linewidth': 2})


# compare_alpha(
#     {'freq': material.freq, '10 cm sample, 45deg inc': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_rand.controls.freq, 'random - 21x21 rec pt': alpha2_rand, 'color': 'blue', 'linewidth': 3},
#     {'freq': zs_rand.controls.freq, 'random - 1 rec pt': alpha1_rand, 'color': 'red', 'linewidth': 2})

#%% Test the reconstruction
# field = PWField(air, controls, material, receivers_reg) # just create an object (does not matter much what it has)
# zs = PArrayDeduction(field) # create a deduction with the field object
# file2load = 'pk_inf_2parray_theta0_d25mm' # File you want to load
# theta = np.deg2rad(0)
# zs.load(file2load) # Load the thing you want to test
# ######### if you want to retest a single point #################
# nx=1
# ny=1
# zs.zs(Lx = 0.1, Ly = 0.1, n_x = nx, n_y = ny)
# ############ Offset center point ####################
# # zs.grid = np.array([0.04, 0.04, 0])
# # zs.zs(Lx = 0.1, Ly = 0.1, n_x = nx, n_y = ny)
# ################# Save simulation ###############
# # zs.save(file2load)
# ########## Recalculate the impedance at that single point ##########
# Zs_mtx = np.divide(zs.p_s, zs.uz_s)
# Hzlist = [np.divide(Zs_mtx, (zs.material.Zs)/(zs.air.c0*zs.air.rho0))]
# ######## Create a receiver object so you can calculate p and u at surface
# recs_surf = Receiver()
# recs_surf.coord = np.reshape(zs.grid, (nx*ny, 3))
# print(recs_surf.coord)
# ######### Calculate pressure and velocity at the surface

# field_surf = PWField(air, controls, zs.material, recs_surf, theta = theta)
# field_surf.p_fps()
# field_surf.uz_fps()
# Hplist = [np.divide(zs.p_s, field_surf.pres_s[0])]
# Hulist = [np.divide(zs.uz_s, field_surf.uz_s[0])]

# plot_spk(field_surf.controls.freq, Hplist, ref=1, legendon=False, title='Hp = Prec(f)/Psurf(f)')
# plot_spk(field_surf.controls.freq, Hulist, ref=1, legendon=False, title='Hu = Uzrec(f)/Uzsurf(f)')
# plot_spk(field_surf.controls.freq, Hzlist, ref=1, legendon=False, title='Hz = Zsrec(f)/Zssurf(f)')
# plt.show()