# import general python modules
import math
import numpy as np
import matplotlib.pyplot as plt
import toml

# import impedance-python modules
from insitu.controlsair import AlgControls, AirProperties, load_cfg
from insitu.material import PorousAbsorber
from insitu.sources import Source
from insitu.receivers import Receiver
from insitu.field_bemflush import BEMFlush
from insitu.field_qterm import LocallyReactiveInfSph
from insitu.qterm_estimation import ImpedanceDeductionQterm

# import impedance-py/C++ module
# import insitu_cpp

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

# step 4 - set the boundary condition / absorber
sources = Source(coord = [0.0, 0.0, 0.3])

# step 5  - set the boundary condition / absorber
receivers = Receiver(coord = [0.0, 0.0, 0.01])
receivers.double_rec()

# step 6 - setup scene and run field calculations
field = BEMFlush(air, controls, material, sources, receivers)
field.generate_mesh(Lx = 0.5, Ly = 0.5, Nel_per_wavelenth=3)
field.psurf()
field.p_fps()
# field.plot_pres()
# field.plot_uz()
# field.plot_scene()
# field.plot_colormap()
# field.save(filename='pu_sim1')

# Compare to infinite sample
# field_inf = LocallyReactiveInfSph(air, controls, material, sources, receivers)
# field_inf.p_loc()
# field_inf.plot_pres()

# step 7 - load field
# saved_field = LocallyReactiveInfSph(air, controls, material, sources, receivers)
# saved_field.load(filename = 'pu_sim1')
# # saved_field.plot_pres()
# # print(saved_field.pres_s[0][0])
# step 8 - create a deduction object with the loaded field sim
zs_ded_qterm = ImpedanceDeductionQterm(field)
zs_ded_qterm.pw_pp()
zs_ded_qterm.pwa_pp()
zs_ded_qterm.zq_pp(zs_ded_qterm.Zs_pwa_pp)

#     #### plotting stuff #################
plt.figure()
plt.title('Porous material measurement comparison')
plt.plot(controls.freq, material.alpha, 'k-', label = 'Reference', linewidth=4)
plt.plot(controls.freq, zs_ded_qterm.alpha_pw_pp, 'grey', label = 'plane wave')
plt.plot(controls.freq, zs_ded_qterm.alpha_pwa_pp, 'g-', label = 'pwa')
plt.plot(controls.freq, zs_ded_qterm.alpha_q_pp, 'r', label = 'q-term')
plt.grid(linestyle = '--', which='both')
plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xticks([50, 100, 500, 1000, 5000, 10000],
    ['50', '100', '500', '1000', '5000', '10000'])
plt.xlabel('Frequency [Hz]')
plt.ylabel('absorption coefficient [-]')
# plt.ylim((-0.2, 1.2))
plt.xlim((0.8 * controls.freq[0], 1.2*controls.freq[-1]))
plt.show()


# figp, axs = plt.subplots(2,1)
# for js, p_s_mtx in enumerate(field.pres_s):
#     for jrec, p_spk in enumerate(p_s_mtx):
#         leg = 'source ' + str(js+1) + ' receiver ' + str(jrec+1)
#         axs[0].semilogx(controls.freq, 20 * np.log10(np.abs(p_spk) / 20e-6), label = 'BEM')
#         p_spk_inf = field_inf.pres_s[0][0]
#         axs[0].semilogx(controls.freq, 20 * np.log10(np.abs(p_spk_inf) / 20e-6), label = 'Inf')
# axs[0].grid(linestyle = '--', which='both')
# axs[0].legend(loc = 'best')
# # axs[0].set(xlabel = 'Frequency [Hz]')
# axs[0].set(ylabel = '|p(f)| [dB]')
# for p_s_mtx in field.pres_s:
#     for p_ph in p_s_mtx:
#         axs[1].semilogx(controls.freq, np.angle(p_ph), label='BEM')
#         p_spk_inf = field_inf.pres_s[0][0]
#         axs[1].semilogx(controls.freq, np.angle(p_spk_inf), label = 'Inf')
# axs[1].grid(linestyle = '--', which='both')
# axs[1].set(xlabel = 'Frequency [Hz]')
# axs[1].set(ylabel = 'phase [-]')
# plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
# xticklabels=['50', '100', '500', '1000', '5000', '10000'])
# plt.setp(axs, xlim=(0.8 * controls.freq[0], 1.2*controls.freq[-1]))
# plt.show()