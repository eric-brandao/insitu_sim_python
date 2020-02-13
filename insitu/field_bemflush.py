import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
import time

# import impedance-py/C++ module and other stuff
import insitu_cpp
from controlsair import plot_spk

class BEMFlush(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using the BEM for a flush sample on a baffle (exact for spherical waves on locally reactive and
    finite samples)
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air, controls, material, sources, receivers):
        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        self.pres_s = []
        self.uz_s = []
        # Load Gauss points and weights
        with open('/home/eric/dev/insitu/data/' + 'gauss_data' + '.pkl', 'rb') as input:
            Nzeta = pickle.load(input)
            Nweights = pickle.load(input)
        self.Nzeta = Nzeta
        self.Nweights = Nweights

    def generate_mesh(self, Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 6):
        '''
        This method is used to generate the mesh for simulation
        '''
        # Get the maximum frequency to estimate element size required
        self.Lx = Lx
        self.Ly = Ly
        freq_max = self.controls.freq[-1]
        el_size = self.air.c0 / (Nel_per_wavelenth * freq_max)
        print('The el_size is: {}'.format(el_size))
        # Number of elementes spaning x and y directions
        Nel_x = np.int(np.ceil(Lx / el_size))
        Nel_y = np.int(np.ceil(Ly / el_size))
        # x and y coordinates of element's center
        xjc = np.linspace(-Lx/2 + el_size/2, Lx/2 - el_size/2, Nel_x)
        yjc = np.linspace(-Ly/2 + el_size/2, Ly/2 - el_size/2, Nel_y)
        # A Nel_x * Nel_y by 2 matrix containing the x and y coords of element centers
        self.el_center = np.zeros((Nel_x*Nel_y, 2), dtype = np.float32)
        self.jacobian = np.float32((el_size**2)/4.0)
        # x and y coordinates of elements edges
        xje = np.linspace(-Lx/2, Lx/2, Nel_x+1)
        yje = np.linspace(-Ly/2, Ly/2, Nel_y+1)
        # A Nel_x * Nel_y by 4 matrix containing the x and y coords of element x and y corners
        self.node_x = np.zeros((Nel_x*Nel_y, 4), dtype = np.float32)
        self.node_y = np.zeros((Nel_x*Nel_y, 4), dtype = np.float32)
        # form a matrix of coordinates of centers and corners
        d = 0
        for m in np.arange(len(yjc)):
            for n in np.arange(len(xjc)):
                self.el_center[d,:]=[xjc[n], yjc[m]] # each line of matrix nodes is a node xj and yj coordinate
                self.node_x[d,:]=[xje[n], xje[n]+el_size, xje[n]+el_size, xje[n]]
                self.node_y[d,:]=[yje[m], yje[m], yje[m]+el_size, yje[m]+el_size]
                d += 1
        # print(xje)
        # print(self.el_center)
        # print(xje)

    def psurf(self,):
        '''
        This method is used to calculate the acoustic field (pressure/velocity)
        as a function of frequency. It will assemble the BEM matrix and solve for
        the surface pressure based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are Nel_x*Nel_y vs N_freq element in this matrix.
        This method saves memory compared to the use of assemble_gij and psurf2.
        On the other hand, if you want to simulate with a different source(s) configuration
        you will have to re-compute the BEM simulation.
        '''
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(len(self.el_center), dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),len(self.el_center),axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        # Assemble the r-matrix (this runs once and stays in memory for freq loop)
        # print("I am assembling a matrix of gauss distances once...")
        # r_mtx = insitu_cpp._bemflush_rmtx(self.el_center,
        #     self.node_x, self.node_y, self.Nzeta)
        # Set a time count for performance check
        tinit = time.time()
        bar = ChargingBar('Calculating the surface pressure for each frequency step (method 1)',
            max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            fakebeta = np.array(0.02+1j*0.2)
            # Assemble the bem matrix (c++)
            # print("Assembling matrix for freq: {} Hz.".format(self.controls.freq[jf]))
            #Version 1 (distances in loop)
            gij = insitu_cpp._bemflush_mtx(self.el_center, self.node_x, self.node_y,
            self.Nzeta, self.Nweights.T, k0, self.beta[jf])

            # Version 2 (distances in memory)
            # gij = insitu_cpp._bemflush_mtx2(self.Nweights.T,
            #     r_mtx, self.jacobian, k0, self.beta[jf])
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            # print('Assembling the matrix for frequency {} Hz'.format(self.controls.freq[jf]))
            bar.next()
        bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def assemble_gij(self,):
        '''
        This method is used to assemble the BEM matrix to be used in further computations.
        You are assembling a Nel x Nel matrix of complex numbers for each frequency step run.
        So, this is memory consuming. On the other hand, if you store this matrix you can change
        freely the positions of sound sources and just reinvert the system of equations.
        This is nice because it can save a lot of time in other simulations, so you just need
        to run your big case once. The method will only assemble the BEM matrix. The calculation of
        surface pressure (based on the incident sound pressure) should be done later.
        '''
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        # Nel = len(self.el_center)
        # self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        # c_mtx = 0.5 * np.identity(len(self.el_center), dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        # rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),len(self.el_center),axis=0)-\
        #     el_3Dcoord
        # r_unpt = np.linalg.norm(rsel, axis = 1)
        # Assemble the r-matrix (this runs once and stays in memory for freq loop)
        # print("I am assembling a matrix of gauss distances once...")
        # r_mtx = insitu_cpp._bemflush_rmtx(self.el_center,
        #     self.node_x, self.node_y, self.Nzeta)
        # Set a time count for performance check
        tinit = time.time()
        bar = ChargingBar('Assembling BEM matrix for each frequency step',
            max=len(self.controls.k0), suffix='%(percent)d%%')
        self.gij_f = []
        for jf, k0 in enumerate(self.controls.k0):
            # fakebeta = np.array(0.02+1j*0.2)
            # Assemble the bem matrix (c++)
            # print("Assembling matrix for freq: {} Hz.".format(self.controls.freq[jf]))
            #Version 1 (distances in loop)
            gij = insitu_cpp._bemflush_mtx(self.el_center, self.node_x, self.node_y,
            self.Nzeta, self.Nweights.T, k0, self.beta[jf])
            self.gij_f.append(gij)
            # Version 2 (distances in memory)
            # gij = insitu_cpp._bemflush_mtx2(self.Nweights.T,
            #     r_mtx, self.jacobian, k0, self.beta[jf])
            # Calculate the unperturbed pressure
            # p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            # self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            # print('Assembling the matrix for frequency {} Hz'.format(self.controls.freq[jf]))
            bar.next()
        bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def psurf2(self,):
        '''
        This method is used to calculate the surface pressure as a function of frequency. 
        It will use assembled BEM matrix (from assemble_gij) and solve for
        the surface pressure based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are Nel_x*Nel_y vs N_freq element in this matrix.
        You need to run this if you change sound source(s) configuration (no need to assemble matrix again).
        '''
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(len(self.el_center), dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),len(self.el_center),axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        tinit = time.time()
        bar = ChargingBar('Calculating the surface pressure for each frequency step (method 2)',
            max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            gij = self.gij_f[jf]
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            # print('Assembling the matrix for frequency {} Hz'.format(self.controls.freq[jf]))
            bar.next()
        bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def p_fps(self,):
        '''
        This method calculates the sound pressure spectrum for all sources and receivers
        Inputs:

        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver

                # print((s_coord[0] - r_coord[0])**2)
                # print((s_coord[1] - r_coord[1])**2)
                # print((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)
                # print(((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)**0.5)
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # print('Calculate p_scat and p_fp for rec: {}'.format(r_coord))
                print('Calculate sound pressure for source {} at ({}) and receiver {} at ({})'.format(js+1, s_coord, jrec+1, r_coord))
                bar = ChargingBar('Processing sound pressure at field point', max=len(self.controls.k0), suffix='%(percent)d%%')
                for jf, k0 in enumerate(self.controls.k0):
                    # print('the ps passed is: {}'.format(self.p_surface[:,jf]))
                    fakebeta = np.array(0.02+1j*0.2)
                    # r_coord = np.reshape(np.array([0, 0, 0.01], dtype = np.float32), (1,3))
                    p_scat = insitu_cpp._bemflush_pscat(r_coord, self.node_x, self.node_y,
                        self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    # print('p_scat for freq {} Hz is: {}'.format(self.controls.freq[jf], p_scat))
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                    bar.next()
                bar.finish()
                    # print('p_fp for freq {} Hz is: {}'.format(self.controls.freq[jf], pres_rec[jrec, jf]))
            self.pres_s.append(pres_rec)

    def uz_fps(self,):
        '''
        This method calculates the sound particle velocity spectrum (z-dir) for all sources and receivers
        Inputs:
        Outputs:
            uz_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a particle
            velocity (z-dir) for a receiver. Each column is a set of particle velocity (z-dir)
            measured by the receivers for a given frequency
        '''
        # Loop the receivers
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
                bar = ChargingBar('Processing particle velocity z-dir (q-term)',
                    max=len(self.controls.k0), suffix='%(percent)d%%')
                for jf, k0 in enumerate(self.controls.k0):
                    uz_scat = insitu_cpp._bemflush_uzscat(r_coord, self.node_x, self.node_y,
                        self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    # print('p_scat for freq {} Hz is: {}'.format(self.controls.freq[jf], p_scat))
                    uz_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                        (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)-\
                        (np.exp(-1j * k0 * r2) / r2) *\
                        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) - uz_scat
                    # Progress bar stuff
                    bar.next()
                bar.finish()
            self.uz_s.append(uz_rec)

    def plot_scene(self, vsam_size = 2, mesh = True):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # vertexes plot
        if mesh:
            for jel in np.arange(len(self.el_center)):
                nodex_el = self.node_x[jel]
                nodey_el = self.node_y[jel]
                nodex = np.reshape(nodex_el.flatten(), (4, 1))
                nodey = np.reshape(nodey_el.flatten(), (4, 1))
                nodez = np.reshape(np.zeros(4), (4, 1))
                vertices = np.concatenate((nodex, nodey, nodez), axis=1)
                verts = [list(zip(vertices[:,0],
                        vertices[:,1], vertices[:,2]))]
                # mesh points
                for v in verts[0]:
                    ax.scatter(v[0], v[1], v[2],
                    color='black',  marker = "o", s=1)
                # patch plot
                collection = Poly3DCollection(verts,
                    linewidths=1, alpha=0.9, edgecolor = 'gray', zorder=1)
                collection.set_facecolor('silver')
                ax.add_collection3d(collection)
        # baffle
        else:
            vertices = np.array([[-self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, self.Ly/2, 0.0],
                [-self.Lx/2, self.Ly/2, 0.0]])
            verts = [list(zip(vertices[:,0],
                    vertices[:,1], vertices[:,2]))]
            # patch plot
            collection = Poly3DCollection(verts,
                linewidths=2, alpha=0.9, edgecolor = 'black', zorder=2)
            collection.set_facecolor('silver')
            ax.add_collection3d(collection)
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=200)
        # plot receiver
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=30, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot

    def plot_pres(self):
        '''
        Method to plot the spectrum of the sound pressure
        '''
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_uz(self):
        '''
        Method to plot the spectrum of the particle velocity in zdir
        '''
        plot_spk(self.controls.freq, self.uz_s, ref = 5e-8)

    def plot_colormap(self, n_pts = 10):
        pass
        # x = np.linspace(-2*self.Lx, 2*self.Lx, n_pts)
        # y = np.zeros(len(x))
        # z = np.linspace(0.01, 1.0, n_pts)
        # xv, zv = np.meshgrid(x, z)
        # r_fpts = np.zeros((xv.shape[0]**2, 3))
        # print(xv.shape)
        # # print(yv.shape)
        # print(zv.shape)
        # for jrec in np.arange(len(r_fpts)):
        #     r_fpts[jrec, :] = np.array([xv[jrec,jrec],
        #         0.0, zv[jrec,jrec]])
        # print(r_fpts)

    def save(self, filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to save the simulation object
        '''
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_qterm', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)


