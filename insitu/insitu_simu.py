import math
import codecs
import json
import time

# import collada as co
import numpy as np
import matplotlib.pyplot as plt
import toml
from tqdm import tqdm

import insitu_cpp


from insitu.receivers import setup_receivers
from insitu.sources import setup_sources
from insitu.controlsair import AlgControls
from insitu.controlsair import AirProperties
from insitu.material import PorousAbsorber
from insitu.field_calc import LocallyReactive

from insitu.absorption_database import load_matdata_from_mat
from insitu.absorption_database import get_alpha_s
from insitu.zs_estimation import ImpedanceDeduction



def main():

    ##### Test Alg controls ########
    controls = AlgControls('simulation.toml')
    # print(controls.freq)
    # print(controls.freq)
    # print(controls.Dt)
    # print(controls.alow_growth)

    ##### Alg controls ########
    air = AirProperties('simulation.toml')
    # print(air.rho0)
    # print(air.c0)
    # print(air.m)

    ##### material to test absorption #############
    air.c0 = 340
    air.rho0 = 1.21
    material = PorousAbsorber('simulation.toml', air, controls.freq)
    material.jcal()
    material.layer_over_rigid(0.025)
    # print("surface impedance")
    # print(material.Zp)
    # print(material.kp)
    # print(material.Zs)
    # material.plot_absorption()

    #### sound field ###############################
    hs = 0.3
    r = 0.0
    z1 = 0.01
    z2 = 0.02
    # rec 1
    r1_lr = LocallyReactive(controls.freq, material.Zs, hs, r, z1, air.c0, air.rho0)
    r1_lr.p_loc()
    # rec 2
    r2_lr = LocallyReactive(controls.freq, material.Zs, hs, r, z2, air.c0, air.rho0)
    r2_lr.p_loc()
    # print("sound pressure")
    # print(r1_lr.pres)
    # print(r2_lr.pres)
    # pu_lr.uz_loc()
    # pu_lr.ur_loc()
    # pu_lr.plot_pres()
    # pu_lr.plot_uz()
    # pu_lr.plot_ur()
    # plt.show()
    ##### Recover surface impedance #########################
    imp_ded = ImpedanceDeduction(r1_lr, r2_lr)
    imp_ded.pw_pp()
    imp_ded.pwa_pp()
    # print("Vp, alpha, Zs - PWA")
    # print(imp_ded.Vp_pwa_pp)
    # print(imp_ded.alpha_pwa_pp)
    # print(imp_ded.Zs_pwa_pp)

    imp_ded.zq_pp(imp_ded.Zs_pwa_pp)
    # print("Vp, alpha, Zs - PWA")
    # print(imp_ded.Vp_q_pp)
    # print(imp_ded.alpha_q_pp)
    # print(imp_ded.Zs_q_pp)
    # Vp_pw, Zs_pw, alpha_pw = pw_pp(r1_lr, r2_lr)

    #### plotting stuff #################
    plt.figure()
    plt.title('Porous material measurement comparison')
    plt.plot(controls.freq, material.alpha, 'k-', label = 'Reference', linewidth=4)
    plt.plot(controls.freq, imp_ded.alpha_pw_pp, 'grey', label = 'plane wave')
    plt.plot(controls.freq, imp_ded.alpha_pwa_pp, 'g-', label = 'pwa')
    plt.plot(controls.freq, imp_ded.alpha_q_pp, 'r', label = 'q-term')
    plt.grid(linestyle = '--', which='both')
    plt.xscale('log')
    plt.legend(loc = 'lower right')
    plt.xticks([50, 100, 500, 1000, 5000, 10000],
        ['50', '100', '500', '1000', '5000', '10000'])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('absorption coefficient [-]')
    plt.ylim((-0.2, 1.2))
    plt.xlim((0.8 * controls.freq[0], 1.2*controls.freq[-1]))
    plt.show()
    ##### Test sources initiation ########
    # sources = setup_sources('simulation.toml', rays)
    # print(sources[0].coord)
    # for js, s in enumerate(sources):
    #     print("Source {} coord: {}.".format(js, s.coord))
    #     print("Source {} orientation: {}.".format(js, s.orientation))
    #     print("Source {} power dB: {}.".format(js, s.power_dB))
    #     print("Source {} eq dB: {}.".format(js, s.eq_dB))
    #     print("Source {} power (linear): {}.".format(js, s.power_lin))
    #     print("Source {} delay: {}. [s]".format(js, s.delay))

    ##### Test receiver initiation ########
    # receivers = setup_receivers('simulation.toml')
    # for jrec, r in enumerate(receivers):
    #     print("Receiver {} coord: {}.".format(jrec, r.coord))
    #     # r.orientation = r.point_to_source(np.array(sources[1].coord))
    #     print("Receiver {} orientation: {}.".format(jrec, r.orientation))

if __name__ == '__main__':
    main()
