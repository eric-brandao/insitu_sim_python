import numpy as np
import matplotlib.pyplot as plt
import toml
from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
from insitu.field_calc import LocallyReactive

class ImpedanceDeduction(object):
    '''
    Impedance deduction class receives two signals (measurement objects)
    '''
    def __init__(self, rec1, rec2):
        self.rec1 = rec1
        self.rec2 = rec2

    def pw_pp(self):
        '''
        mehtod to estimate the surface impedance assuming plane waves and normal incidence.
        input: spectrum of two microphones
        output: Zs, Vp, alpha
        '''
        # determine the distance between the microphones
        x12 = np.abs(self.rec1.z_r - self.rec2.z_r)
        # determine the farthest microphone
        if (self.rec1.z_r > self.rec2.z_r): # right order
            Hw = self.rec2.pres / self.rec1.pres
            x1 = self.rec1.z_r
        else:
            Hw = self.rec1.pres / self.rec2.pres
            x1 = self.rec2.z_r
        self.Vp_pw_pp = ((Hw - np.exp(-1j * self.rec1.k0 * x12)) /
            (np.exp(1j * self.rec1.k0 * x12) - Hw)) * np.exp(2 * 1j * self.rec1.k0 * x1)
        self.Zs_pw_pp = (1 - self.Vp_pw_pp) / (1 + self.Vp_pw_pp)
        self.alpha_pw_pp = 1 - (np.abs(self.Vp_pw_pp)) ** 2
        return self.Vp_pw_pp, self.Zs_pw_pp, self.alpha_pw_pp

    def pwa_pp(self):
        '''
        mehtod to estimate the surface impedance assuming spherical waves,
        that reflect specularly.
        input: spectrum of two microphones
        output: Zs, Vp, alpha
        '''
        k0 = self.rec1.k0
        r = self.rec1.r_r
        h_s = self.rec1.h_s
        # determine the farthest microphone
        if (self.rec1.z_r < self.rec2.z_r): # right order
            H12 = self.rec1.pres / self.rec2.pres
            r11 = self.rec1.r_1
            r12 = self.rec1.r_2
            r21 = self.rec2.r_1
            r22 = self.rec2.r_2
        else:
            H12 = self.rec2.pres / self.rec1.pres
            r11 = self.rec2.r_1
            r12 = self.rec2.r_2
            r21 = self.rec1.r_1
            r22 = self.rec1.r_2
        self.Vp_pwa_pp = (np.exp(-1j * k0 * r11) / r11 - H12 * np.exp(-1j * k0 * r21) / r21) / \
            (H12 * np.exp(-1j * k0 * r22) / r22 - np.exp(-1j * k0 * r12) / r12)
        self.Zs_pwa_pp = ((1 + self.Vp_pwa_pp) / (1 - self.Vp_pwa_pp)) * \
            (((r ** 2 + h_s **2)**0.5) / h_s) * \
            (1j * k0 * (r ** 2 + h_s **2)**0.5) / (1 + 1j * k0 * (r ** 2 + h_s **2)**0.5)
        self.alpha_pwa_pp = 1 - np.abs(self.Vp_pwa_pp) ** 2 #1 - (np.abs(self.Vp_pwa_pp)) ** 2
        return self.Vp_pwa_pp, self.Zs_pwa_pp, self.alpha_pwa_pp

    def zq_pp(self, Zs1, tol = 1e-6, max_iter = 40):
        '''
        mehtod to estimate the surface impedance assuming spherical waves
        and local reaction. With full reflection using q-term formulation.
        It is based on the paper: An iterative method for determining the
        surface impedance of acoustic materials in situ, Alvarez and Jacobsen,
        Internoise 2008.
        input: spectrum of two microphones, Zs1 (initial guess of surface impedance,
        generally the Zs_pwa), tol (tolerance - default 0.000001),
        max_iter (maximum number of iterations - default 40)
        output: Zs, Vp, alpha
        '''
        c0 = self.rec1.c0
        rho0 = self.rec1.rho0
        freq = self.rec1.freq
        k0 = self.rec1.k0
        r = self.rec1.r_r
        h_s = self.rec1.h_s
        # determine the farthest microphone
        if (self.rec1.z_r < self.rec2.z_r): # right order
            H12 = self.rec1.pres / self.rec2.pres
            r11 = self.rec1.r_1
            r12 = self.rec1.r_2
            z1 = self.rec1.z_r
            r21 = self.rec2.r_1
            r22 = self.rec2.r_2
            z2 = self.rec2.z_r
        else:
            H12 = self.rec2.pres / self.rec1.pres
            r11 = self.rec2.r_1
            r12 = self.rec2.r_2
            z1 = self.rec2.z_r
            r21 = self.rec1.r_1
            r22 = self.rec1.r_2
            z2 = self.rec1.z_r
        # allocate memory for the estimated impedance
        self.Zs_q_pp = np.zeros(len(k0), dtype = np.csingle)
        # setup progressbar
        bar = ChargingBar('Calculating the surface impedance (q-term)', max=len(k0), suffix='%(percent)d%%')
        for jf, kf in enumerate(k0): # scan all frequencies
            error = 1000
            iteration = 1
            Zg = Zs1[jf] #1000.0 + 1j*0.0
            Zg1 = 1.0 + 1j*0.0
            fZg = 1.0 + 1j*0.0
            fZg1 = 1.0 + 1j*0.0
            fZg2 = 1.0 + 1j*0.0
            while (error > tol and iteration < max_iter):
                if (iteration == 1):        # first iteration the guessed impedance is the measured impedance
                    Zg = Zs1[jf]
                    # print('Zg before, Zg after, fZg, Zg-fZg')
                    # print(Zg)
                    Zg1 = 1.0 + 1j*0.0
                    fZg = 1.0 + 1j*0.0
                    # print(Zg)
                    # print(fZg)
                    # print(Zg-fZg)
                elif (iteration == 2):   # second iteration the guessed impedance is first calculated imp
                    Zg1 = Zg
                    fZg1 = fZg
                    # print(iteration)
                    # print(Zg)
                    Zg = Zg-fZg
                    # print(Zg)
                # all other iterations the secant method is used to guess the impedance
                elif (np.isfinite(Zg) and Zg != Zg1 and fZg1 != fZg): # To avoid errors
                    Zg2 = Zg1
                    fZg2 = fZg1
                    Zg1 = Zg
                    fZg1 = fZg
                    # if isfinite(Zg) && Zg~=Zg1 && fZg1~=fZg % To avoid errors
                    Zg = Zg1 - ((Zg1 - Zg2) / (fZg1 - fZg2)) * fZg1
                # Now we use the estimates of Zs to estimate the transfer function between the mics
                # and compare it to the measured transfer function
                if (np.isfinite(Zg) and Zg != Zg1 and np.abs(fZg) > tol): # To avoid errors
                    # print(iteration)
                    # print(Zg)

                    p1 = p_integral(kf, Zg, h_s, r, z1)
                    p2 = p_integral(kf, Zg, h_s, r, z2)
                    # p1 = LocallyReactive(freq, Zg, h_s, r, z1, c0, rho0)
                    # p1.p_loc()
                    # p2 = LocallyReactive(freq, Zg, h_s, r, z2, c0, rho0)
                    # p2.p_loc()
                    fZg = H12[jf] - p1 / p2
                    error = np.abs(fZg)
                iteration += 1
            self.Zs_q_pp[jf] = Zg
            bar.next()
        bar.finish()
        self.Vp_q_pp = (self.Zs_q_pp - 1) / (self.Zs_q_pp + 1)
        self.alpha_q_pp = 1 - (np.abs(self.Vp_q_pp)) ** 2
        return self.Vp_q_pp, self.Zs_q_pp, self.alpha_q_pp


def p_integral(k0, Zg, h_s, r_r, z_r):
    '''
    calculate estimative of sound pressure - single frequency
    '''
    beta = 1 / Zg
    r_1 = (r_r ** 2 + (h_s - z_r) ** 2) ** 0.5
    r_2 = (r_r ** 2 + (h_s + z_r) ** 2) ** 0.5
    f_qr = lambda q: np.real((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r_r**2 + (h_s + z_r - 1j*q)**2)**0.5)) /
        ((r_r**2 + (h_s + z_r - 1j*q)**2) ** 0.5)))
    f_qi = lambda q: np.imag((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r_r**2 + (h_s + z_r - 1j*q)**2)**0.5)) /
        ((r_r**2 + (h_s + z_r - 1j*q)**2) ** 0.5)))
    Iq_real = integrate.quad(f_qr, 0.0, 20.0)
    Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
    Iq = Iq_real[0] + 1j * Iq_imag[0]
    pres = (np.exp(-1j * k0 * r_1) / r_1) + \
        (np.exp(-1j * k0 * r_2) / r_2) - 2 * k0 * beta * Iq
    return pres

# def pw_pp(rec1, rec2):
#     '''
#     mehtod to estimate the surface impedance assuming plane waves and normal incidence.
#     input: spectrum of two microphones
#     output: Zs, Vp, alpha
#     '''
#     # determine the distance between the microphones
#     x12 = np.abs(rec1.z_r - rec2.z_r)
#     # determine the farthest microphone
#     if (rec1.z_r > rec2.z_r): # right order
#         Hw = rec2.pres / rec1.pres
#         x1 = rec1.z_r
#     else:
#         Hw = rec1.pres / rec2.pres
#         x1 = rec2.z_r
#     Vp = ((Hw - np.exp(-1j * rec1.k0 * x12)) /
#         (np.exp(1j * rec1.k0 * x12) - Hw)) * np.exp(2 * 1j * rec1.k0 * x1)
#     Zs = (1 - Vp) / (1 + Vp)
#     alpha = 1 - (np.abs(Vp)) ** 2
#     return Vp, Zs, alpha
