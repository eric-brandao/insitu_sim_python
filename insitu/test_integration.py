import numpy as np
import scipy.integrate as integrate
import scipy as spy

h_s = 0.3
r_r = 0
z_r = 0.01
Zs = 2
beta = 1 / Zs
freq = 1000
c0 = 343
w = 2 * np.pi * freq
k0 = w / c0

f_qr = lambda q: np.real((np.exp(-q * k0 * beta)) *
    ((np.exp(-1j * k0 * (r_r**2 + (h_s + z_r - 1j*q)**2)**0.5)) /
    ((r_r**2 + (h_s + z_r - 1j*q)**2) ** 0.5)))
# print(f_qr(0.01))

f_qi = lambda q: np.imag((np.exp(-q * k0 * beta)) *
    ((np.exp(-1j * k0 * (r_r**2 + (h_s + z_r - 1j*q)**2)**0.5)) /
    ((r_r**2 + (h_s + z_r - 1j*q)**2) ** 0.5)))
# print(f_qi(0.01))

Iq_real = integrate.quad(f_qr, 0.0, 20.0)
Iq_imag = integrate.quad(f_qi, 0.0, 20.0)

Iq = -2 * k0 * beta * (Iq_real[0] + 1j * Iq_imag[0])
print(Iq)


################ Test progress bar $####################
import time
import sys

toolbar_width = 40

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

for i in np.arange(toolbar_width):
    i = i+1 # do real work here
    # update the bar
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("]\n") # this ends the progress bar