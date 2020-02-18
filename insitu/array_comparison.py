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
#%% Initialize
air = AirProperties(temperature = 20)
controls = AlgControls(c0 = air.c0, freq_init=100, freq_step = 25, freq_end=4000)
material = PorousAbsorber(air, controls)
material.delany_bazley(resistivity=25000)
material.layer_over_rigid(thickness = 0.025,theta = 0)
#%% step 4 - set the sources
sources = Source(coord = [0.0, 0.0, 1.5])
# step 5  - set the receivers
receivers = Receiver()
#%% Load simulated data - BEM simulation
field = BEMFlush(air, controls, material, sources, receivers)
field.load('bemflush_Lx_30cm_Ly_30cm_theta0_s150cm')
#%% Load simulated data - deduced impedance (sample size comparison)
zs_a_15x15 = PArrayDeduction(field)
zs_a_30x30 = PArrayDeduction(field)
zs_a_50x50 = PArrayDeduction(field)
zs_a_100x100 = PArrayDeduction(field)

zs_a_15x15.load('pk_Lx_15cm_Ly_15cm_2parray_theta0_s150cm')
zs_a_30x30.load('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm')
zs_a_50x50.load('pk_Lx_50cm_Ly_50cm_2parray_theta0_s150cm')
zs_a_100x100.load('pk_Lx_100cm_Ly_100cm_2parray_theta0_s150cm')

zs_a_30x30_162 = PArrayDeduction(field)
zs_a_30x30_642 = PArrayDeduction(field)
zs_a_30x30_162.load('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_nw162')
zs_a_30x30_642.load('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_nw600')

zs_a_30x30_sarray = PArrayDeduction(field)
zs_a_30x30_larray = PArrayDeduction(field)
zs_a_30x30_sarray.load('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_smallerarray')
zs_a_30x30_larray.load('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_largearray')
#%% Plot some color maps
zs_a_30x30.plot_pk_sphere(freq=1000, db=True, dinrange=20,save=True, name='30x30')
zs_a_30x30.plot_pk_sphere(freq=2000, db=True, dinrange=20,save=True, name='30x30')
zs_a_30x30_sarray.plot_pk_sphere(freq=1000, db=True, dinrange=20,save=True, name='30x30_small')
zs_a_30x30_sarray.plot_pk_sphere(freq=2000, db=True, dinrange=20,save=True, name='30x30_small')
zs_a_30x30_larray.plot_pk_sphere(freq=1000, db=True, dinrange=20,save=True, name='30x30_large')
zs_a_30x30_larray.plot_pk_sphere(freq=2000, db=True, dinrange=20,save=True, name='30x30_large')


# zs_a_15x15.plot_pk_sphere(freq=1000, db=True, dinrange=20,save=True, name='15x15')
# zs_a_15x15.plot_pk_sphere(freq=2000, db=True, dinrange=20,save=True, name='15x15')

# zs_a_100x100.plot_pk_sphere(freq=1000, db=True, dinrange=20,save=True, name='100x100')
# zs_a_100x100.plot_pk_sphere(freq=2000, db=True, dinrange=20,save=True, name='100x100')
plt.show()

#%% Deduce alpha and make a plot
# zs_a_15x15.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_15x15.save('pk_Lx_15cm_Ly_15cm_2parray_theta0_s150cm')
# zs_a_30x30.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_30x30.save('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm')
# zs_a_50x50.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_50x50.save('pk_Lx_50cm_Ly_50cm_2parray_theta0_s150cm')
# zs_a_100x100.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_100x100.save('pk_Lx_100cm_Ly_100cm_2parray_theta0_s150cm')

# zs_a_30x30_162.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_30x30_162.save('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_nw162')
# zs_a_30x30_642.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_30x30_642.save('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_nw600')

# zs_a_30x30_sarray.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_30x30_sarray.save('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_smallerarray')
# zs_a_30x30_larray.zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21)
# zs_a_30x30_larray.save('pk_Lx_30cm_Ly_30cm_2parray_theta0_s150cm_largearray')

# %% qterm
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
# compare_alpha(
#     {'freq': material.freq, 'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_a_30x30.controls.freq, '30cm x 30cm': zs_a_30x30.alpha, 'color': 'red', 'linewidth': 2},
#     {'freq': zs_a_50x50.controls.freq, '50cm x 50cm': zs_a_50x50.alpha, 'color': 'm', 'linewidth': 2},
#     {'freq': zs_a_100x100.controls.freq, '100cm x 100cm': zs_a_100x100.alpha, 'color': 'blue', 'linewidth': 2})

# compare_alpha(
#     {'freq': material.freq, 'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_a_30x30.controls.freq, '2562': zs_a_30x30.alpha, 'color': 'red', 'linewidth': 2},
#     {'freq': zs_a_30x30_642.controls.freq, '642': zs_a_30x30_642.alpha, 'color': 'm', 'linewidth': 2},
#     {'freq': zs_a_30x30_162.controls.freq, '162': zs_a_30x30_162.alpha, 'color': 'blue', 'linewidth': 2})

# compare_alpha(
#     {'freq': material.freq, 'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
#     {'freq': zs_a_30x30.controls.freq, '17.5 cm x 17.5 cm': zs_a_30x30.alpha, 'color': 'red', 'linewidth': 2},
#     {'freq': zs_a_30x30_sarray.controls.freq, '8.75 cm x 8.75 cm': zs_a_30x30_sarray.alpha, 'color': 'm', 'linewidth': 2},
#     {'freq': zs_a_30x30_larray.controls.freq, '35.0 cm x 35.0 cm': zs_a_30x30_larray.alpha, 'color': 'blue', 'linewidth': 2},
#     {'freq': zs_ded_qterm.controls.freq, 'single PU probe': zs_ded_qterm.alpha_q_pu, 'color': 'darkcyan', 'linewidth': 2})