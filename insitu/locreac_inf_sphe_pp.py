#%%  import general python modules
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
#%% step 1 - set air properties (2 opts: (i) - set manually; (ii) - by toml file)
# (i) - set manually;
air = AirProperties(temperature = 20)
# (ii) - by toml file
# config = load_cfg('simulation.toml')
# temp = config['air']['Temperature']
# air = AirProperties(temperature = temp)
# print(air.c0)

#%% step 2 - set your controls
controls = AlgControls(c0 = air.c0, freq_step = 100, freq_end=10000)
#%% step 3 - set the boundary condition / absorber
material = PorousAbsorber(air, controls)
# material.jcal()
material.delany_bazley(resistivity=10000)
material.layer_over_rigid(thickness = 0.04,theta = 0)
# material.plot_absorption()
#%% step 4 - set the sources
sources = Source(coord = [0.0, 0.0, 0.3])
#%% step 5  - set the receivers
receivers = Receiver(coord = [2, 0.0, 0.01])
receivers.double_rec(z_dist = 0.01)
# print(receivers.coord)
# print(receivers.coord.shape)
#%% step 6 - setup scene and run field calculations
field = LocallyReactiveInfSph(air, controls, material, sources, receivers)
field.p_loc()
field.plot_pres()
field.plot_scene()
field.save()

#%% step 7 - load field
saved_field = LocallyReactiveInfSph(air, controls, material, sources, receivers)
saved_field.load()
# saved_field.plot_pres()
# print(saved_field.pres_s[0][0])

#%% step 8 - create a deduction object with the loaded field sim
zs_ded_qterm = ImpedanceDeductionQterm(saved_field)
zs_ded_qterm.pw_pp()
zs_ded_qterm.pwa_pp()
zs_ded_qterm.zq_pp(zs_ded_qterm.Zs_pwa_pp)

#%%  #### plotting stuff #################
compare_alpha(controls.freq,
    {'Reference': material.alpha, 'color': 'black', 'linewidth': 4},
    {'Plane Wave': zs_ded_qterm.alpha_pw_pp, 'color': 'grey', 'linewidth': 1},
    {'PWA': zs_ded_qterm.alpha_pwa_pp, 'color': 'green', 'linewidth': 1},
    {'q-term': zs_ded_qterm.alpha_q_pp, 'color': 'red', 'linewidth': 1})
# plt.figure()
# plt.title('Porous material measurement comparison')
# plt.plot(controls.freq, material.alpha, 'k-', label = 'Reference', linewidth=4)
# plt.plot(controls.freq, zs_ded_qterm.alpha_pw_pp, 'grey', label = 'plane wave')
# plt.plot(controls.freq, zs_ded_qterm.alpha_pwa_pp, 'g-', label = 'pwa')
# plt.plot(controls.freq, zs_ded_qterm.alpha_q_pp, 'r', label = 'q-term')
# plt.grid(linestyle = '--', which='both')
# plt.xscale('log')
# plt.legend(loc = 'lower right')
# plt.xticks([50, 100, 500, 1000, 5000, 10000],
#     ['50', '100', '500', '1000', '5000', '10000'])
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('absorption coefficient [-]')
# # plt.ylim((-0.2, 1.2))
# plt.xlim((0.8 * controls.freq[0], 1.2*controls.freq[-1]))
# plt.show()




#     ##### Test Alg controls ########
#     controls = AlgControls('simulation.toml')
#     # print(controls.freq)
#     # print(controls.freq)
#     # print(controls.Dt)
#     # print(controls.alow_growth)

#     ##### Alg controls ########
#     air = AirProperties('simulation.toml')
#     # print(air.rho0)
#     # print(air.c0)
#     # print(air.m)

#     ##### material to test absorption #############
#     air.c0 = 340
#     air.rho0 = 1.21
#     material = PorousAbsorber('simulation.toml', air, controls.freq)
#     material.jcal()
#     material.layer_over_rigid(0.025)
#     # print("surface impedance")
#     # print(material.Zp)
#     # print(material.kp)
#     # print(material.Zs)
#     # material.plot_absorption()

#     #### sound field ###############################
#     hs = 0.3
#     r = 0.0
#     z1 = 0.01
#     z2 = 0.02
#     # rec 1
#     r1_lr = LocallyReactive(controls.freq, material.Zs, hs, r, z1, air.c0, air.rho0)
#     r1_lr.p_loc()
#     # rec 2
#     r2_lr = LocallyReactive(controls.freq, material.Zs, hs, r, z2, air.c0, air.rho0)
#     r2_lr.p_loc()
#     # print("sound pressure")
#     # print(r1_lr.pres)
#     # print(r2_lr.pres)
#     # pu_lr.uz_loc()
#     # pu_lr.ur_loc()
#     # pu_lr.plot_pres()
#     # pu_lr.plot_uz()
#     # pu_lr.plot_ur()
#     # plt.show()
#     ##### Recover surface impedance #########################
#     imp_ded = ImpedanceDeduction(r1_lr, r2_lr)
#     imp_ded.pw_pp()
#     imp_ded.pwa_pp()
#     # print("Vp, alpha, Zs - PWA")
#     # print(imp_ded.Vp_pwa_pp)
#     # print(imp_ded.alpha_pwa_pp)
#     # print(imp_ded.Zs_pwa_pp)

#     imp_ded.zq_pp(imp_ded.Zs_pwa_pp)
#     # print("Vp, alpha, Zs - PWA")
#     # print(imp_ded.Vp_q_pp)
#     # print(imp_ded.alpha_q_pp)
#     # print(imp_ded.Zs_q_pp)
#     # Vp_pw, Zs_pw, alpha_pw = pw_pp(r1_lr, r2_lr)


#     ##### Test sources initiation ########
#     # sources = setup_sources('simulation.toml', rays)
#     # print(sources[0].coord)
#     # for js, s in enumerate(sources):
#     #     print("Source {} coord: {}.".format(js, s.coord))
#     #     print("Source {} orientation: {}.".format(js, s.orientation))
#     #     print("Source {} power dB: {}.".format(js, s.power_dB))
#     #     print("Source {} eq dB: {}.".format(js, s.eq_dB))
#     #     print("Source {} power (linear): {}.".format(js, s.power_lin))
#     #     print("Source {} delay: {}. [s]".format(js, s.delay))

#     ##### Test receiver initiation ########
#     # receivers = setup_receivers('simulation.toml')
#     # for jrec, r in enumerate(receivers):
#     #     print("Receiver {} coord: {}.".format(jrec, r.coord))
#     #     # r.orientation = r.point_to_source(np.array(sources[1].coord))
#     #     print("Receiver {} orientation: {}.".format(jrec, r.orientation))

# if __name__ == '__main__':
#     locre_infsample_sph_pu()