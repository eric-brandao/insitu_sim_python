import numpy as np
#import toml
import matplotlib.pyplot as plt
import time, sys

class AirProperties():
    def __init__(self, c0 = 343.0, rho0 = 1.21, temperature = 20.0, humid = 50.0, p_atm = 101325.0):
        '''
        Set up air properties
        Inputs:
            c0 - sound speed (default 343 m/s - it can be overwriten using standardized calculation)
            rho0 - sound speed (default 1.21 kg/m3 - it can be overwriten using standardized calculation)
            temperature - temperature in degrees (default 20 C)
            humid - relative humidity (default 50 %)
            p_atm - atmospheric pressure (default 101325.0 Pa)
        '''
        # config = load_cfg(config_file)
        # self.c0 = np.array(c0, dtype = np.float32)
        # self.rho0 = np.array(rho0, dtype = np.float32)
        self.c0 = np.array(c0)
        self.rho0 = np.array(rho0)
        self.temperature = np.array(temperature, dtype = np.float32)
        self.hr = np.array(humid, dtype = np.float32)
        self.p_atm = np.array(p_atm, dtype = np.float32)

    def standardized_c0_rho0(self,):
        '''
        This method is used to calculate the standardized value of the sound speed and
        air density based on measurements of temperature, humidity and atm pressure.
        It will overwrite the user supplied values
        '''
        # kappla = 0.026
        temp_kelvin = self.temperature + 273.16 # temperature in [K]
        R = 287.031                 # gas constant
        rvp = 461.521               # gas constant for water vapor
        # pvp from Pierce Acoustics 1955 - pag. 555
        pvp = 0.0658 * temp_kelvin**3 - 53.7558 * temp_kelvin**2 \
            + 14703.8127 * temp_kelvin - 1345485.0465
        # Air viscosity
        # vis = 7.72488e-8 * temp_kelvin - 5.95238e-11 * temp_kelvin**2
        # + 2.71368e-14 * temp_kelvin**3
        # Constant pressure specific heat
        cp = 4168.8 * (0.249679 - 7.55179e-5 * temp_kelvin \
            + 1.69194e-7 * temp_kelvin**2 \
            - 6.46128e-11 * temp_kelvin**3)
        cv = cp - R                 # Constant volume specific heat
        # b2 = vis * cp / kappla      # Prandtl number
        gam = cp / cv               # specific heat constant ratio
        # Air density
        self.rho0 = self.p_atm / (R * temp_kelvin) \
            - (1/R - 1/rvp) * self.hr/100 * pvp/temp_kelvin
        # Air sound speed
        self.c0 = (gam * self.p_atm/self.rho0)**0.5

    def air_absorption(self, freq):
        '''
        Calculates the air aborption coefficient in [m^-1]
        '''
        # temp, p0, rh, freqs = self.temp, self.p0, self.rh, self.freqs

        T_0 = 293.15                # Reference temperature [k]
        T_01 = 273.15               # 0 [C] in [k]
        temp_kelvin = self.temperature + 273.15 # Input temp in [k]
        patm_atm = self.p_atm / 101325 # atmosferic pressure [atm]
        F = freq / patm_atm         # relative frequency
        a_ps_ar = np.zeros(F.shape)
        # Saturation pressure
        psat = patm_atm * 10**(-6.8346 * (T_01/temp_kelvin)**1.261 \
            + 4.6151)
        h = patm_atm * self.hr *(psat/patm_atm)
        # Oxygen gas molecule (N2) relaxation frequency
        F_rO = 1/patm_atm * (24 + 4.04 * 10**4 * h * (0.02 + h) \
            / (0.391 + h))
        # Nytrogen gas molecule (N2) relaxation frequency
        F_rN = 1/patm_atm * (T_0/temp_kelvin)**(1/2) * \
            (9 + 280 * h *np.exp(-4.17 * ((T_0/temp_kelvin)**(1/3) - 1)) )
        # Air absorption in [dB/m]
        alpha_ps = 100 * F**2 / patm_atm * (1.84 \
            * 10**(-11) * (temp_kelvin/T_0)**(1/2) \
                + (temp_kelvin/T_0)**(-5/2) \
            * (0.01278 * np.exp(-2239.1/temp_kelvin) \
                / (F_rO + F**2 / F_rO) \
            + 0.1068*np.exp(-3352/temp_kelvin) / (F_rN + F**2 / F_rN)))
        a_ps_ar = alpha_ps * 20 / np.log(10)
        # Air absorption in [1/m]
        self.m = (1/100) * a_ps_ar * patm_atm \
            / (10 * np.log10(np.exp(1)))
        # return self.m


class AlgControls():
    def __init__(self, c0, freq_init = 100.0, freq_end = 10000.0, freq_step = 10, freq_vec = []):
        '''
        Set up algorithm controls. You set-up your frequency span:
        Inputs:
            freq_init (default - 100 Hz)
            freq_end (default - 10000 Hz)
            freq_step (default - 10 Hz)
        '''
        # config = load_cfg(config_file)
        freq_vec = np.array(freq_vec)
        if freq_vec.size == 0:
            self.freq_init = np.array(freq_init)
            self.freq_end = np.array(freq_end)
            self.freq_step = np.array(freq_step)
            self.freq = np.arange(self.freq_init, self.freq_end + self.freq_step, self.freq_step)
        else:
            self.freq_init = np.array(freq_vec[0])
            self.freq_end = np.array(freq_vec[-1])
            self.freq = freq_vec
        self.c0 = c0
        self.w = 2.0 * np.pi * self.freq
        self.k0 = self.w / self.c0

    def octave_spk(self, tol = 0.1, n_freq_per_band = 3):
        '''
        method to create a log spaced frequency vector which has the same number
        of frequencies per third octave band
        '''
        freq_13 = np.array([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
            12500, 16000, 20000])
        # find the third octave spectrum contained in the limits
        id_f = np.where(np.logical_and(freq_13>=self.freq_init, freq_13<=self.freq_end))
        freq_13_target = freq_13[id_f[0]]
        # Limits of the third octave bands
        fd = 2**(1/(2 * 3))
        fupper = freq_13_target * fd
        flower = freq_13_target / fd
        # loop trough all targeted bands and create a log spaced frequency spectrum
        self.freq = np.zeros(n_freq_per_band*len(freq_13_target))
        for jf, freq in enumerate(freq_13_target):
            fstart = flower[jf] + tol
            fend = fupper[jf] - tol
            self.freq[int(jf*n_freq_per_band):int(jf*n_freq_per_band+(n_freq_per_band))] =\
                np.logspace(np.log2(fstart), np.log2(fend),
                num=n_freq_per_band, endpoint=True, base=2)
        self.w = 2.0 * np.pi * self.freq
        self.k0 = self.w / self.c0
### Function to read the .toml file
def load_cfg(cfgfile):
    '''
    This function is used to read configurations from a .toml file
    '''
    with open(cfgfile, 'r') as f:
        config = toml.loads(f.read())
    return config

### Function to make spk plots
def plot_spk(freq, spk_in_sources, ref = 1.0, legendon = True, title='Spectrum'):
    '''
    This function is used to make plots of the spectrum of pressure or
    particle velocity
    '''
    fig, axs = plt.subplots(2,1)
    for js, spk_mtx in enumerate(spk_in_sources):
        # print('outer loop: {}'.format(js+1))
        # print(spk_mtx.shape)
        for jrec, spk in enumerate(spk_mtx[0:5,:]):
            # print('inner loop: {}'.format(js+1))
            leg = 'source ' + str(js+1) + ' receiver ' + str(jrec+1)
            axs[0].semilogx(freq, 20 * np.log10(np.abs(spk) / ref), label = leg)
            axs[1].semilogx(freq, np.rad2deg(np.angle(spk)), label=leg)
            # axs[0].semilogx(self.controls.freq, np.abs(p_spk), label = leg)
    axs[0].grid(linestyle = '--', which='both')
    if legendon:
        axs[0].legend(loc = 'best')
    axs[0].set(ylabel = '|H(f)| [dB]')
    axs[0].set(title = title)
    # axs[0].set(ylim = dblimq)
    axs[1].grid(linestyle = '--', which='both')
    axs[1].set(xlabel = 'Frequency [Hz]')
    axs[1].set(ylabel = 'phase [-]')
    plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
    xticklabels=['50', '100', '500', '1000', '5000', '10000'])
    plt.setp(axs, xlim=(0.8 * freq[0], 1.2*freq[-1]))
    # plt.show()

### Function to compare spectrums
def compare_spk(freq, *spks, ref = 1.0):
    '''
    This function is used to compare spectrums of pressure or
    particle velocity
    '''
    fig, axs = plt.subplots(2,1)
    for spk_dict in spks:
        spk_leg = list(spk_dict.keys())[0]
        spk = spk_dict[spk_leg]
        axs[0].semilogx(freq, 20 * np.log10(np.abs(spk) / ref), label = spk_leg)
        axs[1].semilogx(freq, np.angle(spk), label = spk_leg)
    axs[0].grid(linestyle = '--', which='both')
    axs[0].legend(loc = 'best')
    axs[0].set(ylabel = '|p(f)| [dB]')
    axs[1].grid(linestyle = '--', which='both')
    axs[1].set(xlabel = 'Frequency [Hz]')
    axs[1].set(ylabel = 'phase [-]')
    plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
    xticklabels=['50', '100', '500', '1000', '5000', '10000'])
    plt.setp(axs, xlim=(0.8 * freq[0], 1.2*freq[-1]))
    plt.show()

### Function to compare spectrums
def compare_alpha(*alphas, title = 'absorption comparison', freq_max=4000,
    save = False, path = '', fname = '', figsize=(6, 6)):
    """ Compare the absorption coefficients of several estimations

    Parameters
        ----------
        *alphas : dictionary
            dictionary containing sample information
            alpha = {
                'freq' : freq vec (1darray)
                'label' : label of the data (str)
                'color' : color of fig (str - matplotlib compatible)
                'linewidth' : width of line (double - matplotlib compatible)
                'linetype' : type of line (str - matplotlib compatible)
            }
        freq_max : float
            maximum frequency of the graph
        title : str
            title of graph
        save : bool
                    Whether to save or not the figure. PDF file with simple standard name
        path : str
            Path to save the figure file
        fname : str
            File name to save the figure file
        figsize : tuple
            size of the figure
    """
    SMALL_SIZE = 14
    BIGGER_SIZE = 16
    #plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rc('legend', fontsize=SMALL_SIZE)
    #plt.rc('title', fontsize=SMALL_SIZE)
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)


    plt.figure()
    plt.title(title)
    for alpha_dict in alphas:
        alpha_color = alpha_dict['color']
        alpha_lw = alpha_dict['linewidth']
        try:
            alpha_lt = alpha_dict['linetype']
        except:
            alpha_lt = '-'
        freq_leg = list(alpha_dict.keys())[0]
        alpha_leg = list(alpha_dict.keys())[1]
        freq = alpha_dict[freq_leg]
        alpha = alpha_dict[alpha_leg]
        plt.semilogx(freq, alpha, alpha_lt, color = alpha_color, label = alpha_leg, linewidth = alpha_lw)
    plt.grid(linestyle = '--', which='both')
    plt.xscale('log')
    plt.legend(loc = 'best')
    plt.xticks([50, 100, 500, 1000, 2000, 4000, 8000, 10000],
        ['50', '100', '500', '1k', '2k', '4k', '8k', '10k'])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$\alpha$ [-]')
    plt.ylim((-0.2, 1.2))
    plt.xlim((80, freq_max))
    plt.tight_layout()
    if save:
        filename = path + fname
        plt.savefig(fname = filename, format='pdf')
    
    # plt.show()

### Function to compare impedances
def compare_zs(*zs, title = 'surface impedance comparison', freq_max=4000, save = False, path = '', fname = ''):
    '''
    This function is used to compare the absorption coefficients of several estimations
    '''
    SMALL_SIZE = 14
    BIGGER_SIZE = 16
    #plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rc('legend', fontsize=SMALL_SIZE)
    #plt.rc('title', fontsize=SMALL_SIZE)
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)
    fig, axs = plt.subplots(2,1)
    for zs_dict in zs:
        zs_color = zs_dict['color']
        zs_lw = zs_dict['linewidth']
        try:
            zs_lt = zs_dict['linetype']
        except:
            zs_lt = '-'
        freq_leg = list(zs_dict.keys())[0]
        zs_leg = list(zs_dict.keys())[1]
        freq = zs_dict[freq_leg]
        zs = zs_dict[zs_leg]
        axs[0].semilogx(freq, np.real(zs), zs_color, linewidth = zs_lw, linestyle = zs_lt)
        axs[1].semilogx(freq, np.imag(zs), zs_color, label = zs_leg, linewidth = zs_lw, linestyle = zs_lt)
        # plt.semilogx(freq, alpha, alpha_color, label = alpha_leg, linewidth = alpha_lw)
    axs[0].grid(linestyle = '--', which='both')
    axs[0].set(ylabel = 'Re{Zs} [-]')
    axs[0].set(ylim = (0.0, 3.0))
    axs[0].set(title = title)
    plt.setp(axs[0], xticks=[50, 100, 500, 1000, 2000, 4000, 8000, 10000], 
        xticklabels=['50', '100', '500', '1k', '2k', '4k', '8k', '10k'])
    axs[1].grid(linestyle = '--', which='both')
    axs[1].set(xlabel = 'Frequency [Hz]')
    axs[1].set(ylabel = 'Im{Zs} [-]')
    axs[1].set(ylim = (-20.0, 5.0))
    # axs[1].legend(loc = 'lower right')
    plt.setp(axs[1], xticks=[50, 100, 500, 1000, 2000, 4000, 8000, 10000], 
        xticklabels=['50', '100', '500', '1k', '2k', '4k', '8k', '10k'])
    # plt.xticks([50, 100, 500, 1000, 4000, 6000, 10000],
    #     ['50', '100', '500', '1k', '4k', '6k', '10k'])
    plt.xlabel('Frequency [Hz]')
    # plt.xlim((0.8 * freq[0], freq_max))
    plt.setp(axs[0], xlim = (0.8 * freq[0], freq_max))
    plt.setp(axs[1], xlim = (0.8 * freq[0], freq_max))
    plt.setp(axs[1], ylim = (-30, 5))
    plt.tight_layout()
    if save:
        filename = path + fname
        plt.savefig(fname = filename, format='pdf')
    # plt.show()

def sph2cart(r, theta, phi):
    '''
    this function is used to convert from spherical to cartesian coordinates
    Inputs:
        r - the radius of the sphere
        theta - the elevation angle
        phi - the azimuth angle
    '''
    # x = r*np.sin(theta)*np.cos(phi)
    # y = r*np.sin(theta)*np.sin(phi)
    # z = r*np.cos(theta)
    # x = r*np.sin(phi)*np.cos(theta)
    # y = r*np.sin(phi)*np.sin(theta)
    # z = r*np.cos(phi)
    ## Same as Matlab
    x = r*np.cos(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.cos(theta)
    z = r*np.sin(theta)
    return x, y, z

def cart2sph(x,y,z):
    '''
    this function is used to convert from cartesian to spherical coordinates
    Inputs:
        x, y, z - cartesian coordinates over the sphere
    '''
    # phi = np.arctan2(y,z) # azimuth
    # theta = np.arctan2(np.sqrt(z**2 + y**2), x) # elevation
    # r = np.sqrt(x**2 + y**2 + z**2)
    ## Same as Matlab
    phi = np.arctan2(y,x) # azimuth
    theta = np.arctan2(z, np.sqrt(x**2 + y**2)) # elevation
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, theta, phi

def plot_material(fig_obj, Lx, Ly, xlim):
    """
    Plots the material in a given figure scene
    """
    # sample
    fig_obj.plot(np.linspace(-Lx/2, Lx/2, 100), np.zeros(100),
        color = 'grey', linewidth = 3.0)
    # baffle
    fig_obj.plot(np.linspace(-xlim, -Lx/2, 500), np.zeros(500),
        color = 'k', linewidth = 3.0)
    fig_obj.plot(np.linspace(Lx/2, xlim, 500), np.zeros(500),
        color = 'k', linewidth = 3.0)
    return fig_obj

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    # progress = float("{0:.2f}".format(progress))
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), float("{0:.2f}".format(progress*100)), status)
    sys.stdout.write(text)
    sys.stdout.flush()