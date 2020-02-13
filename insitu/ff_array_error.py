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
receivers.brick_array(x_len=0.6, y_len=0.8, z_len=0.25, n_x = 8, n_y = 8, n_z=3)
# receivers.brick_array(x_len=1.2, y_len=1.6, z_len=0.5, n_x = 8, n_y = 8, n_z=3)
##### 3D Random array
# receivers.random_3d_array(x_len=0.6, y_len=0.8, z_len=0.25, n_total = 192, zr = 0.1)

# #%% step 7 - Create a reference case
field_ref = FreeField(air, controls, sources, receivers)
# field.plot_scene()
field_ref.planewave_ff(theta = np.pi/2, phi = 0)
# field.add_noise(snr = np.Inf)
# step 8 - create a deduction object with the loaded field sim
ff_ded_parray = PArrayDeduction(field_ref)
ff_ded_parray.wavenum_dir(n_waves=2000, plot = False)
# ff_ded_parray.pk_a3d_constrained(xi = 0.1)
solving_method = 'scipy'
pk_ref = ff_ded_parray.pk_a3d_tikhonov(method = solving_method, lambd_value=[])
# ff_ded_parray.save(filename='ff_pw_regulara_60cmx80cmx25mm_192m')
# ff_ded_parray.load(filename = 'ff_pw_regulara_60cmx80cmx25mm_192m')
# plots
# ff_ded_parray.plot_pk_sphere(freq=1000, db=False)
# plt.show()
# print(pk_ref.shape)

#%% Compare scipy, direct, cvxpy (no noise) 
# pk_direct = ff_ded_parray.pk_a3d_tikhonov(method = 'direct', lambd_value=[])
# error_dir = np.abs(np.abs(np.reshape(pk_ref, len(pk_ref))) - np.abs(np.reshape(pk_direct, len(pk_ref))))
# pk_cvx = ff_ded_parray.pk_a3d_tikhonov(method = 'cvx', lambd_value=[])
# error_cvx = np.abs(np.abs(np.reshape(pk_ref, len(pk_ref))) - np.abs(np.reshape(pk_cvx, len(pk_ref))))

# fig = plt.figure()
# fig.canvas.set_window_title("Comparing Solving methods")
# plt.plot(10*np.log10(error_dir), label = 'direct vs scipy')
# plt.plot(10*np.log10(error_cvx), label = 'cvxpy vs scipy')
# plt.grid(linestyle = '--')
# plt.legend()
# plt.title('With regularization parameter from L-curve')
# plt.xlabel('Direction of arrival [-]')
# plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{method}$')
# plt.ylim((-50, 10))
# plt.savefig('data/methodsens_lsqr_optimlambda', format='pdf')
# plt.show()

#%% Create lamda case to compare to ref.
# lams = np.logspace(start=-5, stop=0, num=10)
# pk_lams = np.zeros((len(pk_ref), len(lams)), dtype=complex)

# for jl, lam in enumerate(lams):
#     # copy and add noise
#     field = field_ref
#     # create deduction obj
#     ff_ded_l = PArrayDeduction(field)
#     ff_ded_l.wavenum_dir(n_waves=2000, plot = False)
#     print('solving for lambda {}'.format(lam))
#     pk_n = ff_ded_parray.pk_a3d_tikhonov(method = solving_method, lambd_value=lam)
#     pk_lams[:, jl] = np.reshape(pk_n, len(pk_n))

# fig = plt.figure()
# fig.canvas.set_window_title("Comparing lambdas")
# for jl, lam in enumerate(lams):
#     legend = 'lambda = ' +str(float("{0:.5f}".format(lam)))#+ str(lam)
#     error = np.abs(np.abs(np.reshape(pk_ref, len(pk_n))) - np.abs(np.reshape(pk_lams[:, jl], len(pk_n))))
#     plt.plot(10*np.log10(error), label = legend)
# plt.grid(linestyle = '--')
# plt.legend()
# plt.title('With no noise')
# plt.xlabel('Direction of arrival [-]')
# plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{noisy}$')
# plt.ylim((-50, 10))
# plt.savefig('data/lambdasens_lsqr_snrinf', format='pdf')
# plt.show()

#%% Create noise case to compare to ref.
snrs = np.array([40, 30, 20, 10, 6, 3])
pk_noisy = np.zeros((len(pk_ref), len(snrs)), dtype=complex)

for jsnr, snr in enumerate(snrs):
    # copy and add noise
    field = field_ref
    field.add_noise(snr = snr)
    # create deduction obj
    ff_ded_noisy = PArrayDeduction(field)
    ff_ded_noisy.wavenum_dir(n_waves=2000, plot = False)
    print('solving for SNR of {} dB'.format(snr))
    pk_n = ff_ded_parray.pk_a3d_tikhonov(method = solving_method, lambd_value=[])
    pk_noisy[:, jsnr] = np.reshape(pk_n, len(pk_n))

fig = plt.figure()
fig.canvas.set_window_title("Comparing SNR")
for jsnr, snr in enumerate(snrs):
    legend = 'SNR = ' + str(snr) + ' dB'
    error = np.abs(np.abs(np.reshape(pk_ref, len(pk_n))) - np.abs(np.reshape(pk_noisy[:, jsnr], len(pk_n))))
    plt.plot(10*np.log10(error), label = legend)
plt.grid(linestyle = '--')
plt.legend()
plt.title('With regularization parameter from L-curve')
plt.xlabel('Direction of arrival [-]')
plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{noisy}$')
plt.ylim((-50, 10))
plt.savefig('data/noisesens_lsqr_optimlambda', format='pdf')
plt.show()

#%% Create noise case to compare to ref. (by solving constrained optimization)
# snrs = np.array([40, 30, 20, 10])
# pk_noisy = np.zeros((len(pk_ref), len(snrs)), dtype=complex)

# ind = 3
# field = field_ref
# field.add_noise(snr = snrs[ind])
# ff_ded_noisy = PArrayDeduction(field)
# ff_ded_noisy.wavenum_dir(n_waves=2000, plot = False)
# print('solving for SNR of {} dB'.format(snrs[ind]))
# epsilon = 10 ** (-snrs[-1]/10)
# # print(epsilon)
# pk_n = ff_ded_noisy.pk_a3d_constrained(epsilon = epsilon)
# print(pk_n)
# ff_ded_noisy.plot_pk_sphere(freq=1000, db=False)



# fig = plt.figure()
# fig.canvas.set_window_title("Comparing SNR for constrained optimization")
# legend = 'SNR = ' + str(snrs[ind]) + ' dB'
# error = np.abs(np.abs(np.reshape(pk_ref, len(pk_n))) - np.abs(np.reshape(pk_n, len(pk_n))))
# plt.plot(10*np.log10(error), label = legend)
# plt.grid(linestyle = '--')
# plt.legend()
# plt.title('With regularization parameter from L-curve')
# plt.xlabel('Direction of arrival [-]')
# plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{noisy}$')
# # plt.ylim((-50, 10))
# plt.savefig('data/noisesens_lsqr_optimlambda', format='pdf')
# plt.show()

# for jsnr, snr in enumerate(snrs):
#     # copy and add noise
#     field = field_ref
#     field.add_noise(snr = snr)
#     # create deduction obj
#     ff_ded_noisy = PArrayDeduction(field)
#     ff_ded_noisy.wavenum_dir(n_waves=2000, plot = False)
#     print('solving for SNR of {} dB'.format(snr))
#     epsilon = 10 ** (-3/10)
#     # print(epsilon)
#     pk_n = ff_ded_noisy.pk_a3d_constrained(epsilon = epsilon)
#     print(pk_n)
#     pk_noisy[:, jsnr] = np.reshape(pk_n, len(pk_n))

# fig = plt.figure()
# fig.canvas.set_window_title("Comparing SNR for constrained optimization")
# for jsnr, snr in enumerate(snrs):
#     legend = 'SNR = ' + str(snr) + ' dB'
#     error = np.abs(np.abs(np.reshape(pk_ref, len(pk_n))) - np.abs(np.reshape(pk_noisy[:, jsnr], len(pk_n))))
#     plt.plot(10*np.log10(error), label = legend)
# plt.grid(linestyle = '--')
# plt.legend()
# plt.title('With regularization parameter from L-curve')
# plt.xlabel('Direction of arrival [-]')
# plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{noisy}$')
# # plt.ylim((-50, 10))
# plt.savefig('data/noisesens_lsqr_optimlambda', format='pdf')
# plt.show()


# fig.canvas.set_window_title("Comparing SNR for constrained optimization")
# for jsnr, snr in enumerate(snrs):
#     fig = plt.figure()
#     legend = 'SNR = ' + str(snr) + ' dB'
#     error = np.abs(np.abs(np.reshape(pk_ref, len(pk_n))) - np.abs(np.reshape(pk_noisy[:, jsnr], len(pk_n))))
#     plt.plot(10*np.log10(error), label = legend)
#     plt.grid(linestyle = '--')
#     plt.legend()
#     plt.title('With regularization parameter from L-curve')
#     plt.xlabel('Direction of arrival [-]')
#     plt.ylabel(r'error $|p(k)|_{ref} - |p(k)|_{noisy}$')
#     # plt.ylim((-50, 10))
#     # plt.savefig('data/noisesens_lsqr_optimlambda', format='pdf')
# plt.show()
