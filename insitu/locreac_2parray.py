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
controls = AlgControls(c0 = air.c0, freq_init=100, freq_step = 25, freq_end=4000)
# step 3 - set the boundary condition / absorber
material = PorousAbsorber(air, controls)
# material.jcal(resistivity = 10000)
material.delany_bazley(resistivity=25000)
material.layer_over_rigid(thickness = 0.025,theta = 0)
# material.plot_absorption()
#%% step 4 - set the sources
sources = Source(coord = [0.0, 0.0, 1.5])
# step 5  - set the receivers
receivers = Receiver()
receivers.double_planar_array(x_len=.175, y_len=0.175, n_x = 8, n_y = 8, dz = 0.029, zr = 0.013)

#%% step 7 - run/load field
### You have some options: (1) - run a new simulation;
# (2) - load an existing simulation and change sound source
# (3) - load an existing simulation and change receivers (calculations at Field Points are easy)
# (data of matrix assembly is saved and you can run some needed steps)
######## (1) Running new and saving ###############################
field = BEMFlush(air, controls, material, sources, receivers)
field.generate_mesh(Lx=1, Ly=1, Nel_per_wavelenth=3)
# field.plot_scene(mesh = False)
field.assemble_gij()
field.psurf2()
field.p_fps()
field.save(filename = 'bemflush_Lx_100cm_Ly_100cm_theta0_s150cm')
field.plot_pres()

######## (2) Loading and changing something ###############################
# field.load(filename = 'bemflush_Lx_30cm_Ly_30cm_theta0_s150cm')
# field.receivers = receivers
# field.sources = sources
# field.p_fps()
# field.plot_pres()
# field.plot_scene(mesh = False)
# field.save(filename='bemflush_Lx_1.0m_Ly_1.0m_2parray')

### to load
# field.load(filename = 'bemflush_Lx_1.0m_Ly_1.0m_2parray')
# field.plot_pres()


#%% step 8 - create a deduction object with the loaded field sim
zs_ded_parray = PArrayDeduction(field)
# zs_ded_parray.wavenum_dir(n_waves=2000, plot = False)
# zs_ded_parray.pk_a3d_tikhonov(method='scipy', lambd_value=[])
zs_ded_parray.save(filename='pk_Lx_100cm_Ly_100cm_2parray_theta0_s150cm')

zs_ded_parray.load(filename='pk_Lx_100cm_Ly_100cm_2parray_theta0_s150cm')
zs_ded_parray.plot_pk_sphere(freq=500, db=True, dinrange=20)
zs_ded_parray.plot_pk_sphere(freq=1000, db=True, dinrange=20)
zs_ded_parray.plot_pk_sphere(freq=2000, db=True, dinrange=20)
plt.show()
zs_ded_parray.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)

#%% plotting stuff #################
compare_alpha(controls.freq,
    {'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
    {'Array 3D (back propagation of Zs)': zs_ded_parray.alpha, 'color': 'red', 'linewidth': 2})

#%% Compare spectrums with infinite calcls - if done
# compare_spk(controls.freq, {'BEM': field.pres_s[0][0]},
#     {'Inf': field_inf.pres_s[0][0]}, ref = 20e-6)
# compare_spk(controls.freq, {'BEM': field.uz_s[0][0]},
#     {'Inf': field_inf.uz_s[0][0]}, ref = 5e-8)

#%%
# rec = Receiver(coord = [0.0, 0.0, 0.01])
# field.receivers = rec
# field.p_fps()
# field.uz_fps()
# from qterm_estimation import ImpedanceDeductionQterm
# zs_ded_qterm = ImpedanceDeductionQterm(field)
# zs_ded_qterm.pw_pu()
# zs_ded_qterm.pwa_pu()
# zs_ded_qterm.zq_pu(zs_ded_qterm.Zs_pwa_pu)

#%% plotting stuff #################
# compare_alpha(controls.freq,
#     {'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'Plane Wave': zs_ded_qterm.alpha_pw_pu, 'color': 'grey', 'linewidth': 1},
#     {'PWA': zs_ded_qterm.alpha_pwa_pu, 'color': 'green', 'linewidth': 1},
#     {'q-term': zs_ded_qterm.alpha_q_pu, 'color': 'red', 'linewidth': 1})