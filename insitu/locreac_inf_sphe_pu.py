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
from insitu.field_qterm import LocallyReactiveInfSph, load_simu
from insitu.qterm_estimation import ImpedanceDeductionQterm

# import impedance-py/C++ module
# import insitu_cpp
#%%
# step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# step 2 - set your controls
controls = AlgControls(c0 = air.c0, freq_step = 100, freq_end=10000)
# step 3 - set the boundary condition / absorber
material = PorousAbsorber(air, controls)
material.jcal(resistivity = 10000)
# material.delany_bazley(resistivity=10000)
material.layer_over_rigid(thickness = 0.04,theta = 0)
# material.plot_absorption()
#%% step 4 - set the sources
sources = Source(coord = [0.0, 0.0, 0.3])

# step 5  - set the receivers
receivers = Receiver(coord = [1, 1, 0.01])
#%% step 6 - setup scene and run field calculations
field = LocallyReactiveInfSph(air, controls, material, sources, receivers)
field.p_loc()
field.uz_loc()
field.plot_pres()
field.plot_uz()
# field.plot_scene()
field.save(filename='pu_sim1')
#%% step 7 - load field
saved_field = LocallyReactiveInfSph(air, controls, material, sources, receivers)
saved_field.load(filename = 'pu_sim1')
# saved_field.plot_pres()
# print(saved_field.pres_s[0][0])
# step 8 - create a deduction object with the loaded field sim
zs_ded_qterm = ImpedanceDeductionQterm(saved_field)
zs_ded_qterm.pw_pu()
zs_ded_qterm.pwa_pu()
zs_ded_qterm.zq_pu(zs_ded_qterm.Zs_pwa_pu)
#%% #### plotting stuff #################
compare_alpha(controls.freq,
    {'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
    {'Plane Wave': zs_ded_qterm.alpha_pw_pu, 'color': 'grey', 'linewidth': 1},
    {'PWA': zs_ded_qterm.alpha_pwa_pu, 'color': 'green', 'linewidth': 1},
    {'q-term': zs_ded_qterm.alpha_q_pu, 'color': 'red', 'linewidth': 1})
