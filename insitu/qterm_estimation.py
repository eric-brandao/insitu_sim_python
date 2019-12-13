import numpy as np
import matplotlib.pyplot as plt
import toml
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
# from insitu.field_calc import LocallyReactive

class ImpedanceDeductionQterm(object):
    '''
    Impedance deduction class receives two signals (measurement objects)
    '''
    def __init__(self, sim_field, source_num = 0):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        self.air = sim_field.air
        self.controls = sim_field.controls
        self.material = sim_field.material
        self.sources = sim_field.sources
        self.receivers = sim_field.receivers
        self.pres_s = sim_field.pres_s[source_num] #FixMe
        try:
            self.pres_s = sim_field.pres_s[source_num] #FixMe
        except:
            self.pres_s = []
        try:
            self.uz_s = sim_field.uz_s[source_num] #FixMe
        except:
            self.uz_s = []
        # self.r1 = self.receivers.coord[0]

    def pw_pp(self):
        '''
        mehtod to estimate the surface impedance assuming plane waves and normal incidence.
        input: spectrum of two microphones
        output: Zs, Vp, alpha
        '''
        # determine the distance between the microphones
        x12 = np.abs(self.receivers.coord[0,2] - self.receivers.coord[1,2])
        # determine the farthest microphone
        if (self.receivers.coord[0,2] > self.receivers.coord[1,2]): # right order
            Hw = self.pres_s[1] / self.pres_s[0]
            x1 = self.receivers.coord[0,2]
        else:
            Hw = self.pres_s[0] / self.pres_s[1]
            x1 = self.receivers.coord[1,2]
        self.Vp_pw_pp = ((Hw - np.exp(-1j * self.controls.k0 * x12)) /
            (np.exp(1j * self.controls.k0 * x12) - Hw)) * np.exp(2 * 1j * self.controls.k0 * x1)
        self.Zs_pw_pp = (1 - self.Vp_pw_pp) / (1 + self.Vp_pw_pp)
        self.alpha_pw_pp = 1 - (np.abs(self.Vp_pw_pp)) ** 2
        # return self.Vp_pw_pp, self.Zs_pw_pp, self.alpha_pw_pp

    def pw_pu(self):
        '''
        mehtod to estimate the surface impedance assuming plane waves and normal incidence.
        input: spectrum of pressure and particle velocity
        output: Zs, Vp, alpha
        '''
        # determine the distance between the microphones
        zr = self.receivers.coord[0,2]
        # determine the farthest microphone
        Zm = self.pres_s[0] / self.uz_s[0]
        self.Vp_pw_pu = ((Zm-1)/(Zm+1)) * np.exp(-2 * 1j * self.controls.k0 * zr)
        self.Zs_pw_pu = (1 - self.Vp_pw_pu) / (1 + self.Vp_pw_pu)
        self.alpha_pw_pu = 1 - (np.abs(self.Vp_pw_pu)) ** 2

    def pwa_pp(self):
        '''
        mehtod to estimate the surface impedance assuming spherical waves,
        that reflect specularly.
        input: spectrum of two microphones
        output: Zs, Vp, alpha
        '''
        k0 = self.controls.k0
        r = ((self.sources.coord[0][0] - self.receivers.coord[0,0])**2 +
            (self.sources.coord[0][1] - self.receivers.coord[0,1])**2)**0.5
        # ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1]**2)) self.rec1.r_r
        h_s = self.sources.coord[0][2]
        # determine the farthest microphone
        if (self.receivers.coord[0,2] < self.receivers.coord[1,2]): # right order
            H12 = self.pres_s[0] / self.pres_s[1]
            r11 = (r ** 2 + (h_s - self.receivers.coord[0,2]) ** 2) ** 0.5
            r12 = (r ** 2 + (h_s + self.receivers.coord[0,2]) ** 2) ** 0.5
            r21 = (r ** 2 + (h_s - self.receivers.coord[1,2]) ** 2) ** 0.5
            r22 = (r ** 2 + (h_s + self.receivers.coord[1,2]) ** 2) ** 0.5
        else:
            H12 = self.pres_s[1] / self.pres_s[0]
            r11 = (r ** 2 + (h_s - self.receivers.coord[1,2]) ** 2) ** 0.5
            r12 = (r ** 2 + (h_s + self.receivers.coord[1,2]) ** 2) ** 0.5
            r21 = (r ** 2 + (h_s - self.receivers.coord[0,2]) ** 2) ** 0.5
            r22 = (r ** 2 + (h_s + self.receivers.coord[0,2]) ** 2) ** 0.5

        self.Vp_pwa_pp = (np.exp(-1j * k0 * r11) / r11 - H12 * np.exp(-1j * k0 * r21) / r21) / \
            (H12 * np.exp(-1j * k0 * r22) / r22 - np.exp(-1j * k0 * r12) / r12)
        self.Zs_pwa_pp = ((1 + self.Vp_pwa_pp) / (1 - self.Vp_pwa_pp)) * \
            (((r ** 2 + h_s **2)**0.5) / h_s) * \
            (1j * k0 * (r ** 2 + h_s **2)**0.5) / (1 + 1j * k0 * (r ** 2 + h_s **2)**0.5)
        self.alpha_pwa_pp = 1 - np.abs(self.Vp_pwa_pp) ** 2 #1 - (np.abs(self.Vp_pwa_pp)) ** 2
        # return self.Vp_pwa_pp, self.Zs_pwa_pp, self.alpha_pwa_pp

    def pwa_pu(self):
        '''
        mehtod to estimate the surface impedance assuming spherical waves,
        that reflect specularly.
        input: spectrum of pressure and particle velocity (z-dir)
        output: Zs, Vp, alpha
        '''
        k0 = self.controls.k0
        Zm = (self.pres_s[0] / self.uz_s[0]) #/ (self.air.rho0 * self.air.c0)
        r = ((self.sources.coord[0][0] - self.receivers.coord[0,0])**2 +
            (self.sources.coord[0][1] - self.receivers.coord[0,1])**2)**0.5
        # print(self.pres_s[0:10])
        # print(self.uz_s[0:10])
        # ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1]**2)) self.rec1.r_r
        h_s = self.sources.coord[0][2]
        
        # determine the farthest microphone
        zr = self.receivers.coord[0,2]
        r1 = (r ** 2 + (h_s - zr) ** 2) ** 0.5
        r2 = (r ** 2 + (h_s + zr) ** 2) ** 0.5
        self.Vp_pwa_pu = (r2/r1) * ((Zm * ((h_s - zr)/r1) * (1/(1j * k0 * r1) + 1) - 1)/\
            (Zm * ((h_s + zr)/r2) * (1/(1j * k0 * r2) + 1) + 1)) * np.exp(-1j*k0*(r1-r2))
        self.Zs_pwa_pu = ((1 + self.Vp_pwa_pu) / (1 - self.Vp_pwa_pu)) * \
            (((r ** 2 + h_s **2)**0.5) / h_s) * \
            (1j * k0 * (r ** 2 + h_s **2)**0.5) / (1 + 1j * k0 * (r ** 2 + h_s **2)**0.5)
        self.alpha_pwa_pu = 1 - np.abs(self.Vp_pwa_pu) ** 2 #1 - (np.abs(self.Vp_pwa_pp)) ** 2
        # return self.Vp_pwa_pp, self.Zs_pwa_pp, self.alpha_pwa_pp

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
        # c0 = self.air.c0
        # rho0 = self.air.rho0
        # freq = self.controls.freq
        k0 = self.controls.k0
        r = ((self.sources.coord[0][0] - self.receivers.coord[0,0])**2 +
            (self.sources.coord[0][1] - self.receivers.coord[0,1])**2)**0.5
        # ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1]**2)) self.rec1.r_r
        h_s = self.sources.coord[0][2]
        # determine the farthest microphone
        if (self.receivers.coord[0,2] < self.receivers.coord[1,2]): # right order z1 < z2
            H12 = self.pres_s[0] / self.pres_s[1]
            z1 = self.receivers.coord[0,2]
            z2 = self.receivers.coord[1,2]
        else:
            H12 = self.pres_s[1] / self.pres_s[0]
            z1 = self.receivers.coord[1,2]
            z2 = self.receivers.coord[0,2]
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
                    Zg1 = 1.0 + 1j*0.0
                    fZg = 1.0 + 1j*0.0
                elif (iteration == 2):   # second iteration the guessed impedance is first calculated imp
                    Zg1 = Zg
                    fZg1 = fZg
                    Zg = Zg-fZg
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
                    p1 = p_integral(kf, Zg, h_s, r, z1)
                    p2 = p_integral(kf, Zg, h_s, r, z2)
                    fZg = H12[jf] - p1 / p2
                    error = np.abs(fZg)
                iteration += 1
            self.Zs_q_pp[jf] = Zg
            bar.next()
        bar.finish()
        self.Vp_q_pp = (self.Zs_q_pp - 1) / (self.Zs_q_pp + 1)
        self.alpha_q_pp = 1 - (np.abs(self.Vp_q_pp)) ** 2
        return self.Vp_q_pp, self.Zs_q_pp, self.alpha_q_pp

    def zq_pu(self, Zs1, tol = 1e-6, max_iter = 40):
        '''
        mehtod to estimate the surface impedance assuming spherical waves
        and local reaction. With full reflection using q-term formulation.
        It is based on the paper: An iterative method for determining the
        surface impedance of acoustic materials in situ, Alvarez and Jacobsen,
        Internoise 2008.
        input: spectrum of pu-probe, Zs1 (initial guess of surface impedance,
        generally the Zs_pwa), tol (tolerance - default 0.000001),
        max_iter (maximum number of iterations - default 40)
        output: Zs, Vp, alpha
        '''
        # c0 = self.air.c0
        # rho0 = self.air.rho0
        # freq = self.controls.freq
        k0 = self.controls.k0
        r = ((self.sources.coord[0][0] - self.receivers.coord[0,0])**2 +
            (self.sources.coord[0][1] - self.receivers.coord[0,1])**2)**0.5
        h_s = self.sources.coord[0][2]
        # determine the farthest microphone
        Zm = self.pres_s[0] / self.uz_s[0]
        zr = self.receivers.coord[0,2]

        # allocate memory for the estimated impedance
        self.Zs_q_pu = np.zeros(len(k0), dtype = np.csingle)
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
                    Zg1 = 1.0 + 1j*0.0
                    fZg = 1.0 + 1j*0.0
                elif (iteration == 2):   # second iteration the guessed impedance is first calculated imp
                    Zg1 = Zg
                    fZg1 = fZg
                    Zg = Zg-fZg
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
                    p = p_integral(kf, Zg, h_s, r, zr)
                    uz = uz_integral(kf, Zg, h_s, r, zr)
                    fZg = Zm[jf] - p / uz
                    error = np.abs(fZg)
                iteration += 1
            self.Zs_q_pu[jf] = Zg
            bar.next()
        bar.finish()
        self.Vp_q_pu = (self.Zs_q_pu - 1) / (self.Zs_q_pu + 1)
        self.alpha_q_pu = 1 - (np.abs(self.Vp_q_pu)) ** 2
        # return self.Vp_q_pp, self.Zs_q_pp, self.alpha_q_pp

def p_integral(k0, Zg, hs, r, zr):
    '''
    calculate estimative of sound pressure - single frequency
    '''
    beta = 1 / Zg
    r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
    r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
    f_qr = lambda q: np.real((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
    f_qi = lambda q: np.imag((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
    Iq_real = integrate.quad(f_qr, 0.0, 20.0)
    Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
    # Iq_real = integrate.quadrature(f_qr, 0.0, 20.0, maxiter = 500)
    # Iq_imag = integrate.quadrature(f_qi, 0.0, 20.0, maxiter = 500)
    Iq = Iq_real[0] + 1j * Iq_imag[0]
    pres = (np.exp(-1j * k0 * r1) / r1) + \
        (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * beta * Iq
    return pres

def uz_integral(k0, Zg, hs, r, zr):
    '''
    calculate estimative of particle velocity (z-dir) - single frequency
    '''
    beta = 1 / Zg
    r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
    r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
    f_qr = lambda q: np.real(((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5))) *
        ((hs + zr - 1j*q) / (r**2 + (hs + zr - 1j*q)**2)**0.5) *
        (1 + (1 / (1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5))))
    f_qi = lambda q: np.imag(((np.exp(-q * k0 * beta)) *
        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5))) *
        ((hs + zr - 1j*q) / (r**2 + (hs + zr - 1j*q)**2)**0.5) *
        (1 + (1 / (1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5))))
    Iq_real = integrate.quad(f_qr, 0.0, 20.0)
    Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
    # Iq_real = integrate.quadrature(f_qr, 0.0, 20.0, maxiter = 500)
    # Iq_imag = integrate.quadrature(f_qi, 0.0, 20.0, maxiter = 500)
    Iq = Iq_real[0] + 1j * Iq_imag[0]
    uz = (np.exp(-1j * k0 * r1) / r1) *\
        (1 + (1 / (1j * k0 * r1))) * ((hs - zr)/r1) -\
        (np.exp(-1j * k0 * r2) / r2) *\
        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) +\
        2 * k0 * beta * Iq
    return uz

# class ImpedanceDeduction(object):
#     '''
#     Impedance deduction class receives two signals (measurement objects)
#     '''
#     def __init__(self, rec1, rec2):
#         self.rec1 = rec1
#         self.rec2 = rec2

#     def pw_pp(self):
#         '''
#         mehtod to estimate the surface impedance assuming plane waves and normal incidence.
#         input: spectrum of two microphones
#         output: Zs, Vp, alpha
#         '''
#         # determine the distance between the microphones
#         x12 = np.abs(self.rec1.z_r - self.rec2.z_r)
#         # determine the farthest microphone
#         if (self.rec1.z_r > self.rec2.z_r): # right order
#             Hw = self.rec2.pres / self.rec1.pres
#             x1 = self.rec1.z_r
#         else:
#             Hw = self.rec1.pres / self.rec2.pres
#             x1 = self.rec2.z_r
#         self.Vp_pw_pp = ((Hw - np.exp(-1j * self.rec1.k0 * x12)) /
#             (np.exp(1j * self.rec1.k0 * x12) - Hw)) * np.exp(2 * 1j * self.rec1.k0 * x1)
#         self.Zs_pw_pp = (1 - self.Vp_pw_pp) / (1 + self.Vp_pw_pp)
#         self.alpha_pw_pp = 1 - (np.abs(self.Vp_pw_pp)) ** 2
#         return self.Vp_pw_pp, self.Zs_pw_pp, self.alpha_pw_pp

#     def pwa_pp(self):
#         '''
#         mehtod to estimate the surface impedance assuming spherical waves,
#         that reflect specularly.
#         input: spectrum of two microphones
#         output: Zs, Vp, alpha
#         '''
#         k0 = self.rec1.k0
#         r = self.rec1.r_r
#         h_s = self.rec1.h_s
#         # determine the farthest microphone
#         if (self.rec1.z_r < self.rec2.z_r): # right order
#             H12 = self.rec1.pres / self.rec2.pres
#             r11 = self.rec1.r_1
#             r12 = self.rec1.r_2
#             r21 = self.rec2.r_1
#             r22 = self.rec2.r_2
#         else:
#             H12 = self.rec2.pres / self.rec1.pres
#             r11 = self.rec2.r_1
#             r12 = self.rec2.r_2
#             r21 = self.rec1.r_1
#             r22 = self.rec1.r_2
#         self.Vp_pwa_pp = (np.exp(-1j * k0 * r11) / r11 - H12 * np.exp(-1j * k0 * r21) / r21) / \
#             (H12 * np.exp(-1j * k0 * r22) / r22 - np.exp(-1j * k0 * r12) / r12)
#         self.Zs_pwa_pp = ((1 + self.Vp_pwa_pp) / (1 - self.Vp_pwa_pp)) * \
#             (((r ** 2 + h_s **2)**0.5) / h_s) * \
#             (1j * k0 * (r ** 2 + h_s **2)**0.5) / (1 + 1j * k0 * (r ** 2 + h_s **2)**0.5)
#         self.alpha_pwa_pp = 1 - np.abs(self.Vp_pwa_pp) ** 2 #1 - (np.abs(self.Vp_pwa_pp)) ** 2
#         return self.Vp_pwa_pp, self.Zs_pwa_pp, self.alpha_pwa_pp

#     def zq_pp(self, Zs1, tol = 1e-6, max_iter = 40):
#         '''
#         mehtod to estimate the surface impedance assuming spherical waves
#         and local reaction. With full reflection using q-term formulation.
#         It is based on the paper: An iterative method for determining the
#         surface impedance of acoustic materials in situ, Alvarez and Jacobsen,
#         Internoise 2008.
#         input: spectrum of two microphones, Zs1 (initial guess of surface impedance,
#         generally the Zs_pwa), tol (tolerance - default 0.000001),
#         max_iter (maximum number of iterations - default 40)
#         output: Zs, Vp, alpha
#         '''
#         c0 = self.rec1.c0
#         rho0 = self.rec1.rho0
#         freq = self.rec1.freq
#         k0 = self.rec1.k0
#         r = self.rec1.r_r
#         h_s = self.rec1.h_s
#         # determine the farthest microphone
#         if (self.rec1.z_r < self.rec2.z_r): # right order z1 < z2
#             H12 = self.rec1.pres / self.rec2.pres
#             r11 = self.rec1.r_1
#             r12 = self.rec1.r_2
#             z1 = self.rec1.z_r
#             r21 = self.rec2.r_1
#             r22 = self.rec2.r_2
#             z2 = self.rec2.z_r
#         else: # z2<z1
#             H12 = self.rec2.pres / self.rec1.pres
#             r11 = self.rec2.r_1
#             r12 = self.rec2.r_2
#             z1 = self.rec2.z_r
#             r21 = self.rec1.r_1
#             r22 = self.rec1.r_2
#             z2 = self.rec1.z_r
#         # allocate memory for the estimated impedance
#         self.Zs_q_pp = np.zeros(len(k0), dtype = np.csingle)
#         # setup progressbar
#         bar = ChargingBar('Calculating the surface impedance (q-term)', max=len(k0), suffix='%(percent)d%%')
#         for jf, kf in enumerate(k0): # scan all frequencies
#             error = 1000
#             iteration = 1
#             Zg = Zs1[jf] #1000.0 + 1j*0.0
#             Zg1 = 1.0 + 1j*0.0
#             fZg = 1.0 + 1j*0.0
#             fZg1 = 1.0 + 1j*0.0
#             fZg2 = 1.0 + 1j*0.0
#             while (error > tol and iteration < max_iter):
#                 if (iteration == 1):        # first iteration the guessed impedance is the measured impedance
#                     Zg = Zs1[jf]
#                     # print('Zg before, Zg after, fZg, Zg-fZg')
#                     # print(Zg)
#                     Zg1 = 1.0 + 1j*0.0
#                     fZg = 1.0 + 1j*0.0
#                     # print(Zg)
#                     # print(fZg)
#                     # print(Zg-fZg)
#                 elif (iteration == 2):   # second iteration the guessed impedance is first calculated imp
#                     Zg1 = Zg
#                     fZg1 = fZg
#                     # print(iteration)
#                     # print(Zg)
#                     Zg = Zg-fZg
#                     # print(Zg)
#                 # all other iterations the secant method is used to guess the impedance
#                 elif (np.isfinite(Zg) and Zg != Zg1 and fZg1 != fZg): # To avoid errors
#                     Zg2 = Zg1
#                     fZg2 = fZg1
#                     Zg1 = Zg
#                     fZg1 = fZg
#                     # if isfinite(Zg) && Zg~=Zg1 && fZg1~=fZg % To avoid errors
#                     Zg = Zg1 - ((Zg1 - Zg2) / (fZg1 - fZg2)) * fZg1
#                 # Now we use the estimates of Zs to estimate the transfer function between the mics
#                 # and compare it to the measured transfer function
#                 if (np.isfinite(Zg) and Zg != Zg1 and np.abs(fZg) > tol): # To avoid errors
#                     # print(iteration)
#                     # print(Zg)

#                     p1 = p_integral(kf, Zg, h_s, r, z1)
#                     p2 = p_integral(kf, Zg, h_s, r, z2)
#                     # p1 = LocallyReactive(freq, Zg, h_s, r, z1, c0, rho0)
#                     # p1.p_loc()
#                     # p2 = LocallyReactive(freq, Zg, h_s, r, z2, c0, rho0)
#                     # p2.p_loc()
#                     fZg = H12[jf] - p1 / p2
#                     error = np.abs(fZg)
#                 iteration += 1
#             self.Zs_q_pp[jf] = Zg
#             bar.next()
#         bar.finish()
#         self.Vp_q_pp = (self.Zs_q_pp - 1) / (self.Zs_q_pp + 1)
#         self.alpha_q_pp = 1 - (np.abs(self.Vp_q_pp)) ** 2
#         return self.Vp_q_pp, self.Zs_q_pp, self.alpha_q_pp


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
