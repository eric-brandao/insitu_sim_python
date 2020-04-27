import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy.special as sp
from scipy import optimize
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
from controlsair import plot_spk
# import quadpy.line_segment._gauss_lobatto as qp

class NLRInfSph(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using the non-locally reactive formulation (exact for spherical waves on 
    non-locally reactive and infinite samples)
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air = [], controls = [], material = [], sources = [], receivers = [], sing_step = 0.01):
        self.limits = np.arange(start = 0, stop = 1 + sing_step, step = sing_step)
        ## air
        self.air = air
        self.controls = controls
        self.material = material
        ## material
        try:
            self.Zp = material.Zp
            self.kp = material.kp
            # self.thick = material.thickness
            self.cp = np.divide(2*np.pi*self.controls.freq, self.kp)
            self.rhop = np.divide(self.Zp * self.kp, 2*np.pi*self.controls.freq)
            self.m = np.divide(self.air.rho0, self.rhop)
            self.n = np.divide(self.kp, self.controls.k0)
            self.thickness = material.thickness
            self.beta = (self.air.rho0 * self.air.c0) / material.Zs  # normalized surface admitance
        except:
            self.Zp = []
            self.kp = []
            # self.thick = material.thickness
            self.cp = []
            self.rhop = []
            self.m = []
            self.n = []
            self.thickness = []
            self.beta = []  # normalized surface admitance
        ## sources / receivers
        self.sources = sources
        self.receivers = receivers
        ## quantities
        self.pres_s = []
        self.uz_s = []
        self.singularities_r = []
        self.singularities_i = []

    def determine_singularities(self, ):
        '''
        A method to find the singulatities in the integrands. They are independent of source-receiver configuration,
        and the same for pressure and particle velocity
        '''
        bar = ChargingBar('Finding singularities', max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            # the denominator
            fs_den = lambda s: (np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j)))
            fs_den_r = lambda s: np.real(fs_den(s))
            fs_den_i = lambda s: np.imag(fs_den(s))
            sol_r = optimize.root_scalar(fs_den_r,x0 = 1, x1 = 1-self.limits[1], method='secant')
            sol_i = optimize.root_scalar(fs_den_i,x0 = 1, x1 = 1-self.limits[1], method='secant')
            self.singularities_r.append(sol_r.root)
            self.singularities_i.append(sol_i.root)
            bar.next()
        bar.finish()

    def p_nlr(self, upper_int_limit = 10):
        '''
        This method calculates the sound pressure spectrum for all sources and receivers
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
        # self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # setup progressbar
                print('Calculate sound pressure for source {} and receiver {}'.format(js+1, jrec+1))
                bar = ChargingBar('Processing sound pressure (NLR)', max=len(self.controls.k0), suffix='%(percent)d%%')
                # pres = []
                for jf, k0 in enumerate(self.controls.k0):
                    # integrand
                    fs = lambda s: ((2*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*\
                        k0*s*sp.jv(0,k0*s*r))/\
                        ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                        np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                    fs_r = lambda s: np.real(fs(s))
                    fs_i = lambda s: np.imag(fs(s))
                    # Integrate
                    if not self.singularities_r: # in case singularities were not computed
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                    else:
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                    # Iq_real = qp(fs_r, [0.0, upper_int_limit])
                    # Iq_imag = qp(fs_i, [0.0, upper_int_limit])
                    I_nlr = Iq_real[0] + 1j * Iq_imag[0]
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) - (np.exp(-1j * k0 * r2) / r2) + I_nlr
                    bar.next()
                bar.finish()
            self.pres_s.append(pres_rec)

    def uz_nlr(self, upper_int_limit = 10):
        '''
        This method calculates the sound particle velocity spectrum (z-dir) for all sources and receivers
        Inputs:
            upper_int_limit (default 10) - upper integral limit for truncation
        Outputs:
            uz_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a particle
            velocity (z-dir) for a receiver. Each column is a set of particle velocity (z-dir)
            measured by the receivers for a given frequency
        '''
        # setup progressbar
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
                bar = ChargingBar('Processing particle velocity z-dir (NLR)',
                    max=len(self.controls.k0), suffix='%(percent)d%%')
                for jf, k0 in enumerate(self.controls.k0):
                    # fs = lambda s: ((2*((s**2-1)**0.5)*np.exp(-k0*((s**2-1)**0.5)*(hs+zr)))*k0*s*sp.jv(0,k0*s*r))/\
                    #     (((s**2-1)**0.5)+self.m[jf]*((s**2-self.n[jf]**2)**0.5)*\
                    #     np.tanh(k0*self.material.thickness*((s**2-self.n[jf]**2)**0.5)))

                    fs = lambda s: ((2*(np.sqrt(s**2-1+0j))*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*k0*s*sp.jv(0,k0*s*r))/\
                        ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                        np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                    fs_r = lambda s: np.real(fs(s))
                    fs_i = lambda s: np.imag(fs(s))
                    # Integrate
                    if not self.singularities_r: # in case singularities were not computed
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                    else:
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                    Iq = Iq_real[0] + 1j * Iq_imag[0]
                    # particle velocity
                    uz_rec[jrec, jf] = ((np.exp(-1j * k0 * r1) / r1)*\
                        (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)+\
                        (np.exp(-1j * k0 * r2) / r2) *\
                        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) -(1/1j)*Iq)

                    bar.next()
                bar.finish()
            self.uz_s.append(uz_rec)

    def p_mult(self, upper_int_limit = 10):
        '''
        This method calculates the sound pressure spectrum for a distribution of sources at all receivers.
        It considers that the integration can be done once by considering a sumation of the contributions of
        all sound sources in the integrand.
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
        # get array of source heights
        hs = self.sources.coord[:,2]
        # initialize
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        # loop over receivers
        for jrec, r_coord in enumerate(self.receivers.coord):
            # get arrays of distances
            r = ((self.sources.coord[:,0] - r_coord[0])**2 + (self.sources.coord[:,1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
            zr = r_coord[2]  # receiver height
            r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
            r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
            # setup progressbar
            print('Calculate sound pressure for receiver {}'.format(jrec+1))
            bar = ChargingBar('Processing sound pressure (NLR)', max=len(self.controls.k0), suffix='%(percent)d%%')
            for jf, k0 in enumerate(self.controls.k0):
                # integrand
                fs = lambda s: np.sum((2*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*\
                    k0*s*sp.jv(0,k0*s*r))/\
                    ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                    np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                fs_r = lambda s: np.real(fs(s))
                fs_i = lambda s: np.imag(fs(s))
                # Integrate
                if not self.singularities_r: # in case singularities were not computed
                    Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                    Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                else:
                    Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                        limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                    Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                        limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                # Iq_real = qp(fs_r, [0.0, upper_int_limit])
                # Iq_imag = qp(fs_i, [0.0, upper_int_limit])
                I_nlr = Iq_real[0] + 1j * Iq_imag[0]
                pres_rec[jrec, jf] = np.sum(np.exp(-1j * k0 * r1) / r1 - np.exp(-1j * k0 * r2) / r2) + I_nlr
                bar.next()
            bar.finish()
        self.pres_s.append(pres_rec)

    def plot_scene(self, vsam_size = 3):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # vertexes plot
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        # ax.scatter(vertices[:,0], vertices[:,1],
        #     vertices[:,2], color='blue')
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=0.9, edgecolor = 'gray')
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=50)
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        plt.xticks([], [])
        ax.set_ylabel('Y axis')
        plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='none')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        # ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zlim((0, 1))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=5, azim=-55)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'my_nlr', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to save the simulation object
        '''
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_nlr', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)