import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
from tqdm import tqdm
import pickle
import time

# import impedance-py/C++ module and other stuff
try:
    import insitu_cpp
except:
    print("I could not find insitu_cpp. You should be able to load BEM files and add noise.")
try:
    from insitu.controlsair import plot_spk
except:
    from controlsair import plot_spk

class BEMFlush(object):
    """ Calculates the sound field above a finite locally reactive squared sample.

    It is used to calculate the sound pressure and particle velocity using
    the BEM formulation for an absorbent sample flush mounted  on a hard baffle
    (exact for spherical waves on locally reactive and finite samples)

    Attributes
    ----------
    beta : numpy array
        material normalized surface admitance
    Nzeta : numpy ndarray
        functions for quadrature integration (loaded from picke)
    Nweights : numpy ndarray
        wights for quadrature integration (loaded from picke)
    el_center : (Nelx2) numpy array
        coordinates of element's center (Nel is the number of elements)
    jacobian : float
        the Jacobian of the mesh
    node_x : (Nelx4) numpy array
        x-coordinates of element's vertices
    node_y : (Nelx4) numpy array
        y-coordinates of element's vertices
    p_surface : (NelxNfreq) numpy array
        The sound pressure at the center of each element on the mesh (Nfreq = len(freq)).
        Solved by the BEM C++ module.
    gij_f : list of (NelxNfreq) numpy arrays
        The BEM matrix for each frequency step.
    pres_s - list of receiver pressure spectrums for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
        Each line of the matrix is a spectrum of a sound pressure for a receiver.
        Each column is a set of sound pressure at all receivers for a frequency.
    ux_s - list of receiver velocity spectrums (x-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
    uy_s - list of receiver velocity spectrums (y-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
    uz_s - list of receiver velocity spectrums (z-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.

    Methods
    ----------
    generate_mesh(Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 6)
        Generate the mesh for simulation

    def psurf()
        Calculates the surface pressure of the BEM mesh

    assemble_gij()
        Assemble the BEM matrix.

    psurf2()
        Calculates p_surface using assembled gij_f matrixes.

    p_fps()
        Calculates the total sound pressure spectrum at the receivers coordinates.

    uz_fps()
        Calculates the total particle velocity spectrum at the receivers coordinates.

    add_noise(snr = 30, uncorr = False)
        Add gaussian noise to the simulated data.

    plot_scene(vsam_size = 2, mesh = True)
        Plot of the scene using matplotlib - not redered

    plot_pres()
        Plot the spectrum of the sound pressure for all receivers

    plot_uz()
        Plot the spectrum of the particle velocity in zdir for all receivers

    plot_colormap(freq = 1000):
        Plots a color map of the pressure field.

    plot_intensity(self, freq = 1000)
        Plots a color map of the pressure field.

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, air = [], controls = [], material = [], sources = [], receivers = []):
        """

        Parameters
        ----------
        air : object (AirProperties)
            The relevant properties of the air: c0 (sound speed) and rho0 (air density)
        controls : object (AlgControls)
            Controls of the simulation (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance)
        sources : object (Source)
            The sound sources in the field
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        try:
            self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        except:
            self.beta = []
        self.pres_s = []
        self.ux_s = []
        self.uy_s = []
        self.uz_s = []
        # Load Gauss points and weights
        # with open('/home/eric/dev/insitu/data/' + 'gauss_data' + '.pkl', 'rb') as input:
        #     Nzeta = pickle.load(input)
        #     Nweights = pickle.load(input)
        # self.Nzeta = Nzeta
        # self.Nweights = Nweights
        self.Nzeta, self.Nweights = zeta_weights()
        # print("pause")

    def generate_mesh(self, Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 6):
        """ Generate the mesh for simulation

        The mesh will consists of rectangular elements. Their size is a function
        of the sample's size (Lx and Ly) and the maximum frequency intended and
        the number of elements per wavelength (recomended: 6)

        Parameters
        ----------
        Lx : float
            Sample's lenght
        Ly : float
            Sample's width
        Nel_per_wavelenth : int
            Number of elements per wavelength. The default value is 6.
        """
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

    def psurf(self,):
        """ Calculates the surface pressure of the BEM mesh.

        Uses C++ implemented module.
        The surface pressure calculation represents the first step in a BEM simulation.
        It will assemble the BEM matrix and solve for the surface pressure
        based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are N vs len(self.controls.freq) entries in this matrix.
        This method saves memory, compared to the use of assemble_gij and psurf2.
        On the other hand, if you want to simulate with a different source(s) configuration
        you will have to re-compute the BEM simulation.
        """
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(Nel, dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((Nel, 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),Nel,axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        # Assemble the r-matrix (this runs once and stays in memory for freq loop)
        # print("I am assembling a matrix of gauss distances once...")
        # r_mtx = insitu_cpp._bemflush_rmtx(self.el_center,
        #     self.node_x, self.node_y, self.Nzeta)
        # Set a time count for performance check
        tinit = time.time()
        # bar = ChargingBar('Calculating the surface pressure for each frequency step (method 1)',
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating the surface pressure for each frequency step (method 1)')
        for jf, k0 in enumerate(self.controls.k0):
            # fakebeta = np.array(0.02+1j*0.2)
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
            bar.update(1)
        bar.close()
        #     bar.next()
        # bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def assemble_gij(self,):
        """ Assemble the BEM matrix.

        Uses C++ implemented module.
        Assembles a Nel x Nel matrix of complex numbers for each frequency step.
        It is memory consuming. On the other hand, it is independent of the sound sources.
        If you store this matrix, you can change the positions of sound sources
        and use the information in memory to calculate the p_surface attribute.
        This can save time in simulations where you vary such parametes. The calculation of
        surface pressure (based on the incident sound pressure) should be done later.
        """
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        # Set a time count for performance check
        tinit = time.time()
        # bar = ChargingBar('Assembling BEM matrix for each frequency step',
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Assembling BEM matrix for each frequency step')
        self.gij_f = []
        # print(dir(insitu_cpp))
        for jf, k0 in enumerate(self.controls.k0):
            gij = insitu_cpp._bemflush_mtx(self.el_center, self.node_x, self.node_y,
            self.Nzeta, self.Nweights.T, k0, self.beta[jf])
            self.gij_f.append(gij)
            bar.update(1)
        bar.close()
        #     bar.next()
        # bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def psurf2(self,):
        """ Calculates p_surface using assembled gij_f matrixes.

        Uses C++ implemented module.
        The surface pressure calculation represents the first step in a BEM simulation.
        It will use assembled BEM matrix (from assemble_gij) and solve for
        the surface pressure based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are N vs len(self.controls.freq) entries in this matrix.
        This method saves processing time, compared to the use of psurf. You need to run it
        if you change sound source(s) configuration (no need to run assemble_gij again)..
        """
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
        # bar = ChargingBar('Calculating the surface pressure for each frequency step (method 2)',
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating the surface pressure for each frequency step (method 2)')
        for jf, k0 in enumerate(self.controls.k0):
            gij = self.gij_f[jf]
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            bar.update(1)
        bar.close()
        #     bar.next()
        # bar.finish()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def p_fps(self,):
        """ Calculates the total sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # print('Calculate p_scat and p_fp for rec: {}'.format(r_coord))
                print('Calculate sound pressure for source {} at ({}) and receiver {} at ({})'.format(js+1, s_coord, jrec+1, r_coord))
                # bar = ChargingBar('Processing sound pressure at field point', max=len(self.controls.k0), suffix='%(percent)d%%')
                bar = tqdm(total = len(self.controls.k0),
                    desc = 'Processing sound pressure at field point')
                for jf, k0 in enumerate(self.controls.k0):
                    # print('the ps passed is: {}'.format(self.p_surface[:,jf]))
                    # fakebeta = np.array(0.02+1j*0.2)
                    # r_coord = np.reshape(np.array([0, 0, 0.01], dtype = np.float32), (1,3))
                    p_scat = insitu_cpp._bemflush_pscat(r_coord, self.node_x, self.node_y,
                        self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                    bar.update(1)
                bar.close()
                #     bar.next()
                # bar.finish()
                    # print('p_fp for freq {} Hz is: {}'.format(self.controls.freq[jf], pres_rec[jrec, jf]))
            self.pres_s.append(pres_rec)

    def uz_fps(self, compute_ux = False, compute_uy = False):
        """ Calculates the total particle velocity spectrum at the receivers coordinates.

        The particle velocity spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total particle velocity = incident + scattered.
        The z-direction of particle velocity is always computed. x and y directions are optional.

        Parameters
        ----------
        compute_ux : bool
            Whether to compute x component of particle velocity or not (Default is False)
        compute_uy : bool
            Whether to compute y component of particle velocity or not (Default is False)
        """
        # Loop the receivers
        if compute_ux and compute_uy:
            message = 'Processing particle velocity (x,y,z dir at field point)'
        elif compute_ux:
            message = 'Processing particle velocity (x,z dir at field point)'
        elif compute_uy:
            message = 'Processing particle velocity (y,z dir at field point)'
        else:
            message = 'Processing particle velocity (z dir at field point)'

        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            if compute_ux:
                ux_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            if compute_uy:
                uy_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
                # bar = ChargingBar('Processing particle velocity z-dir',
                #     max=len(self.controls.k0), suffix='%(percent)d%%')
                bar = tqdm(total = len(self.controls.k0),
                    desc = message)
                for jf, k0 in enumerate(self.controls.k0):
                    uz_scat = insitu_cpp._bemflush_uzscat(r_coord, self.node_x, self.node_y,
                        self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    # print(uz_scat)
                    # print('p_scat for freq {} Hz is: {}'.format(self.controls.freq[jf], p_scat))
                    uz_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                        (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)-\
                        (np.exp(-1j * k0 * r2) / r2) *\
                        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) - uz_scat
                    if compute_ux:
                        ux_scat = insitu_cpp._bemflush_uxscat(r_coord, self.node_x, self.node_y,
                            self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                        ux_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                            (1 + (1 / (1j * k0 * r1)))* (-r_coord[0]/r1)-\
                            (np.exp(-1j * k0 * r2) / r2) *\
                            (1 + (1 / (1j * k0 * r2))) * (-r_coord[0]/r2) - ux_scat
                    if compute_uy:
                        uy_scat = insitu_cpp._bemflush_uyscat(r_coord, self.node_x, self.node_y,
                            self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                        uy_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                            (1 + (1 / (1j * k0 * r1)))* (-r_coord[1]/r1)-\
                            (np.exp(-1j * k0 * r2) / r2) *\
                            (1 + (1 / (1j * k0 * r2))) * (-r_coord[1]/r2) - uy_scat
                    # Progress bar stuff
                    bar.update(1)
                bar.close()
            self.uz_s.append(uz_rec)
            if compute_ux:
                self.ux_s.append(ux_rec)
            if compute_uy:
                self.uy_s.append(uy_rec)

    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure and particle velocity data.
        it reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        """
        signal = self.pres_s[0]
        try:
            signal_u = self.uz_s[0]
        except:
            signal_u = np.zeros(1)
        if uncorr:
            signalPower_lin = (np.abs(signal)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            # signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
            signalPower_lin = ((np.mean(np.abs(signal), axis=0))/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
            if signal_u.any() != 0:
                signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
                signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
                noisePower_dB_u = signalPower_dB_u - snr
                noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        # noise = 2*np.sqrt(noisePower_lin)*\
        #     (np.random.randn(signal.shape[0], signal.shape[1]) + 1j*np.random.randn(signal.shape[0], signal.shape[1]))
        self.pres_s[0] = signal + noise
        if signal_u.any() != 0:
            # print('Adding noise to particle velocity')
            noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
            self.uz_s[0] = signal_u + noise_u

    def plot_scene(self, vsam_size = 2, mesh = False):
        """ Plot of the scene using matplotlib - not redered

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        mesh : bool
            Whether to plot the sample mesh or not. Default is False. In this way,
            the sample is represented by a grey rectangle.
        """
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
                color='blue',  marker = "o", alpha = 0.35)
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        # ax.set_zlim((0, 0.3))
        # ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 0.1, 0.2, 0.3))
        ax.set_xticks((-1, -0.5, 0.0, 0.5, 1.0))
        ax.set_yticks((-1, -0.5, 0.0, 0.5, 1.0))

        ax.view_init(elev=10, azim=45)
        # ax.invert_zaxis()
        plt.show() # show plot

    def plot_pres(self,):
        """ Plot the spectrum of the sound pressure for all receivers
        """
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_uz(self):
        """ Plot the spectrum of the particle velocity in zdir for all receivers
        """
        plot_spk(self.controls.freq, self.uz_s, ref = 5e-8)

    def plot_colormap(self, freq = 1000, total_pres = True,  dinrange = 20):
        """Plots a color map of the pressure field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.
        total_pres : bool
            Whether to plot the total sound pressure (Default = True) or the reflected only.
            In the later case, we subtract the incident field Green's function from the total
            sound field.
        dinrange : float
            Dinamic range of the color map

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        # color parameter
        if total_pres:
            color_par = 20*np.log10(np.abs(self.pres_s[0][:,id_f])/np.amax(np.abs(self.pres_s[0][:,id_f])))
        else:
            r1 = np.linalg.norm(self.sources.coord - self.receivers.coord, axis = 1)
            color_par = np.abs(self.pres_s[0][:,id_f]-\
                np.exp(-1j * self.controls.k0[id_f] * r1) / r1)
            color_par = 20*np.log10(color_par/np.amax(color_par))

        # Create triangulazition
        triang = tri.Triangulation(self.receivers.coord[:,0], self.receivers.coord[:,2])
        # Figure
        fig = plt.figure() #figsize=(8, 8)
        # fig = plt.figure()
        fig.canvas.set_window_title('pressure color map')
        plt.title('Reference |P(f)| (BEM sim)')
        # p = plt.tricontourf(triang, color_par, np.linspace(-15, 0, 15), cmap = 'seismic')
        p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap = 'seismic')
        fig.colorbar(p)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt

    def plot_intensity(self, freq = 1000):
        """Plots a vector map of the intensity field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        # Intensities
        Ix = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.ux_s[0][:,id_f]))
        Iy = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.uy_s[0][:,id_f]))
        Iz = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.uz_s[0][:,id_f]))
        I = np.sqrt(Ix**2+Iy**2+Iz**2)
        # # Figure
        fig = plt.figure() #figsize=(8, 8)
        fig.canvas.set_window_title('Intensity distribution map')
        cmap = 'viridis'
        plt.title('Reference Intensity (BEM sim)')
        # if streamlines:
        #     q = plt.streamplot(self.receivers.coord[:,0], self.receivers.coord[:,2],
        #         Ix/I, Iz/I, color=I, linewidth=2, cmap=cmap)
        #     fig.colorbar(q.lines)
        # else:
        q = plt.quiver(self.receivers.coord[:,0], self.receivers.coord[:,2],
            Ix/I, Iz/I, I, cmap = cmap, width = 0.010)
        #fig.colorbar(q)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt
        # Figure
        # fig = plt.figure() #figsize=(8, 8)
        # ax = fig.gca(projection='3d')
        # cmap = 'seismic'
        # # fig = plt.figure()
        # # fig.canvas.set_window_title('Intensity distribution map')
        # plt.title('|I|')
        # q = ax.quiver(self.receivers.coord[:,0], self.receivers.coord[:,1],
        #     self.receivers.coord[:,2], Ix, Iy, Iz,
        #     cmap = cmap, length=0.01, normalize=True)
        # c = I
        # c = getattr(plt.cm, cmap)(c)
        # # fig.colorbar(p)
        # fig.colorbar(q)
        # q.set_edgecolor(c)
        # q.set_facecolor(c)
        # plt.xlabel(r'$x$ [m]')
        # plt.ylabel(r'$z$ [m]')

    def save(self, filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/'):
        """ To save the simulation object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/'):
        """ Load a simulation object.

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

def zeta_weights():
    """ Calculates Nzeta and Nweights
    """
    zeta = np.array([-0.93246951, -0.66120939, -0.23861918,
    0.23861918, 0.66120939, 0.93246951], dtype=np.float32)

    weigths = np.array([0.17132449, 0.36076157, 0.46791393,
        0.46791393, 0.36076157, 0.17132449], dtype=np.float32)

    # Create vectors of size 1 x 36 for the zetas
    N1 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
    N2 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
    N3 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))
    N4 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))

    N1 = np.reshape(N1, (1,zeta.size**2))
    N2 = np.reshape(N2, (1,zeta.size**2))
    N3 = np.reshape(N3, (1,zeta.size**2))
    N4 = np.reshape(N4, (1,zeta.size**2))

    # Let each line of the following matrix be a N vector
    Nzeta = np.zeros((4, zeta.size**2), dtype=np.float32)
    Nzeta[0,:] = N1
    Nzeta[1,:] = N2
    Nzeta[2,:] = N3
    Nzeta[3,:] = N4

    # Create vector of size 1 x 36 for the weights
    Nweigths = np.matmul(np.reshape(weigths, (zeta.size,1)),  np.reshape(weigths, (1,zeta.size)))
    Nweigths = np.reshape(Nweigths, (1,zeta.size**2))
    # print('I have calculated!')
    return Nzeta, Nweigths

