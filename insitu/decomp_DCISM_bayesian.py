# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 2025

@author: Eric Brandão
"""

import numpy as np
from receivers import Receiver
from sources import Source
import material as mat
from material import PorousAbsorber, kp_rhop_range_study
from material import get_min_kp_rhop, get_max_kp_rhop, get_min_beta, get_max_beta
from directDCISM import dDCISM
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from lcurve_functions_EU import csvd, l_curve, tikhonov
import lcurve_functions as lc

import utils_insitu as ut_is
# from decomp_quad_v2 import Decomposition_QDT
from baysian_sampling import BayesianSampler

class DCISM_Bayesian(object):
    """ Bayesian Decomposition of the sound field using the dDCISM.

    The class has methods to perform sound field decomposition into a set of 
    source and complex image source componensts. It is a Bayesian implementation
    that uses the dDCISM as a forward model (as implemented by M. Eser)
    """
    def __init__(self, p_mtx=None, controls=None, air = None, 
                 receivers = None, source = None, sampling_scheme = 'single ellipsoid',
                 enlargement_factor = 1.25):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        air : object (AirProperties)
            air properties
        receivers : object (Receivers)
            Receiver properties.
        source : object (Receivers)
            Source properties

        The objects are stored as attributes in the class (easier to retrieve).
        """
        
        
        self.pres_s = p_mtx
        self.controls = controls
        self.air = air
        self.receivers = receivers
        self.source = source
        self.sampling_scheme = sampling_scheme
        self.enlargement_factor = enlargement_factor
        # self.set_reference_sensor(ref_sens = 0)
        # self.parameters_names = ["Re(s)", "Im(s)", "Re(is)", "Im(is)"]
        # Get receiver data (frequency independent)
        # self.r, self.zr, self.r1, self.r2 = self.get_rec_parameters(self.receivers)
        
    def dict_forward_models(self,):
        """ Builds a dict of possible forward models and print it on screen.
        """
        available_models = {0:
                            {"name": "Locally reacting sample - H(f)",
                             "num_model_par": 2,
                             "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$"]},
                            1:
                            {"name": "Locally reacting sample - P(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]},                   
                            2:
                            {"name": "NLR semi-infinite layer - H(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$"]},
                            3:
                            {"name": "NLR semi-infinite layer - P(f)",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$",
                                                   r"$|Q|$", r"$\angle Q$"]},
                            4:
                            {"name": "NLR single layer with known thickness - H(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$"]},
                            5:
                            {"name": "NLR single layer with unknown thickness - H(f)",
                             "num_model_par": 5,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$t_p$"]},
                            6:
                            {"name": "NLR single layer with known thickness - P(f)",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$",
                                                   r"$|Q|$", r"$\angle Q$"]},
                            7:
                            {"name": "NLR single layer with unknown thickness - P(f)",
                             "num_model_par": 7,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$t_p$",
                                                   r"$|Q|$", r"$\angle Q$"]}
                            }
        return available_models
    
    def show_available_models(self,):
        """ Display (terminal) the available models.
        """
        available_models  = self.dict_forward_models()
        for jm in range(len(available_models)):
            print("Model {}:".format(jm))
            print("\tName: {}".format(available_models[jm]['name']))
            print("\tNumber of model parameters: {}".format(available_models[jm]['num_model_par']))
            print("\tParameters names: {}".format(available_models[jm]['parameters_names']))
            
    def choose_forward_model(self, chosen_model = 1):
        """ Chooses a model from the available models.
        """
        self.chosen_model = chosen_model
        available_models  = self.dict_forward_models()
        self.chosen_model_name = available_models[self.chosen_model]['name']
        self.num_model_par = available_models[self.chosen_model]['num_model_par']
        self.parameters_names = available_models[self.chosen_model]['parameters_names']
        self.get_model_fun()
        
    def get_model_fun(self,):
        """ Gets callable model function
        """
        if self.chosen_model == 0:
            self.model_fun = self.forward_model_0
        elif self.chosen_model == 4:
            self.model_fun = self.forward_model_4
        elif self.chosen_model == 5:
            self.model_fun = self.forward_model_5
        elif self.chosen_model == 6:
            self.model_fun = self.forward_model_6
        elif self.chosen_model == 7:
            self.model_fun = self.forward_model_7
        
    def show_chosen_model_parameters(self,):
        """ Display (terminal) the chosen model.
        """
        available_models  = self.dict_forward_models()
        print("Model {}:".format(self.chosen_model))
        print("\tName: {}".format(available_models[self.chosen_model]['name']))
        print("\tNumber of model parameters: {}".format(available_models[self.chosen_model]['num_model_par']))
        print("\tParameters names: {}".format(available_models[self.chosen_model]['parameters_names']))
    
    def set_sample_thickness(self, t_p = 0.05):
        """ Set the sample thickness for model 1 (layer with known thickness)
        
        Parameters
        ----------
        t_p : float
            Material layer thickness
        """
        self.t_p = t_p
    
    def set_prior_limits(self, 
                         lower_bounds = [1.01, -5.00, 1.00, -5.00, 0.04, 0.5, -np.pi], 
                         upper_bounds = [3.00, -0.05, 1.20, -0.40, 0.06, 2.00, np.pi]):
        """ Set the uniform prior limits.
        
        The specified range on k_p and \rho_p was set with simulation. 
        
        Parameters
        ----------
        lower_bonds : numpy1dArray
            Array containing the lower bonds of the parameter space
        upper_bonds : numpy1dArray
            Array containing the upper bonds of the parameter space
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(lower_bounds) != self.num_model_par or len(upper_bounds) != self.num_model_par:
            raise ValueError("Lower and Upper bonds must be a vector with size {}".format(self.num_model_par))
        self.lower_bounds = lower_bounds      
        self.upper_bounds = upper_bounds
        
        # if self.chosen_model != 0 or self.chosen_model != 1:
        self.set_physical_limits()
        
    def set_physical_limits(self,):
        """ Set sensible physical limits to wave-number and density
        """
        if self.chosen_model != 0 and self.chosen_model != 1:
            # Re{k_p} constraint
            if self.lower_bounds[0] < 1.00:
                self.lower_bounds[0] = 1.00
            # Im{k_p} constraint
            if self.upper_bounds[1] > -0.01:
                self.upper_bounds[1] = -0.01
            # Re{rho_p} constraint    
            if self.lower_bounds[2] < 0.9: # 1.18/1.3
                self.lower_bounds[2] = 0.9
            # Im{rho_p} constraint    
            if self.upper_bounds[3] > -0.01: # 1.18/1.3
                self.upper_bounds[3] = -0.01
        else:
            if self.lower_bounds[0] < 0.001: # Do not allow for real part lower than zero
                self.lower_bounds[0] = 0.001
            
    
    def wide_prior_range(self, widening_factor = 2):
        """ Wide your prior by a given factor
        """
        assert widening_factor >= 0
        delta = np.array(self.upper_bounds) - np.array(self.lower_bounds)
        delta2sum = (delta/2)*((widening_factor-1))#-(delta/4)
        self.lower_bounds -= delta2sum
        self.upper_bounds += delta2sum
        if self.chosen_model != 0 or self.chosen_model != 1:
            self.set_physical_limits()
    

    def set_reference_sensor(self, ref_sens = 0):
        """ Set the reference sensor index, new receiver array and measured data.
        
        Parameters
        ----------
        ref_sens : int
            reference sensor index
        """
        # Set ref sensor
        self.ref_sens = ref_sens
        # Set new receiver array
        new_receivers = Receiver()
        new_receivers.coord = np.delete(self.receivers.coord, self.ref_sens, axis = 0)
        self.receivers = new_receivers 
        # Set new measurement data
        self.pres_s = self.pres_s / self.pres_s[self.ref_sens,:]
        self.pres_s = np.delete(self.pres_s, self.ref_sens, axis = 0)
        
    def set_mic_pairs(self,):
        """ Get the indexes of mic-pair transfer functions
        """
        # Collect mic-pair transfer funtions
        self.id_z_list = self.receivers.get_micpair_indices()
        self.tf_array = Receiver()
        self.tf_array.coord = self.receivers.coord[self.id_z_list[0],:]
        # Change measured data to transfer function
        p_exp_line_1 = self.pres_s[self.id_z_list[0],:]
        p_exp_line_2 = self.pres_s[self.id_z_list[1],:]
        self.pres_s = p_exp_line_1/p_exp_line_2
            
    def setup_dDCISM(self,  T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0):
        """ Initialization of object:
            
        Parameters
        -------------
        T0 : float
            End value of integration path
        dt : float
            Sample rate at which reflection coeffcient is sampled
        tol : float
            Truncation threshold for SVD in MP
        gamma : float
            Scaling factor for extended range analysis
        """
        self.dDCISMsf = dDCISM(air = self.air, controls = self.controls, 
                               source = self.source, receivers = self.receivers,
                               T0 = T0, dt = dt, tol = tol, gamma = gamma)
    
    def forward_model_0(self, x_meas, 
                        model_par = [0.5, 1.00]):
        """ Forward model for forward prediction
        
        Valid for Model 0 - LR single layer - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{beta}, Im{beta}].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex surface admittance
        beta = model_par[0] + 1j*model_par[1]       
        green_fun = self.dDCISMsf.predict_p_lr(k = self.current_k0, 
                                               beta = beta)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_4(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50]):
        """ Forward model for forward prediction
        
        Valid for Model 4 - NLR single layer with known thickness - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = self.t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_5(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.05]):
        """ Forward model for forward prediction
        
        Valid for Model 5 - NLR single layer with unknown thickness - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        # unknown thickness
        t_p = model_par[4]
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_6(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 1.00, -np.pi]):
        """ Forward model for forward prediction
        
        Valid for Model 6 - NLR single layer with known thickness - P(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, |S|, \angle(S)].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        # Source strength
        source_strength = model_par[4]*np.exp(1j*model_par[5])
        
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, rho_p = rho_p, 
                                                      t_p = self.t_p)
        pred = source_strength * green_fun
        return pred
    
    def forward_model_7(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.05, 1.00, -np.pi]):
        """ Forward model for forward prediction
        
        Valid for Model 7 - NLR single layer with unknown thickness - P(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p, |S|, \angle(S)].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        # unknown thickness
        t_p = model_par[4]
        # Source strength
        source_strength = model_par[4]*np.exp(1j*model_par[5])
        
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, rho_p = rho_p, 
                                                      t_p = t_p)
        pred = source_strength * green_fun
        return pred
    
    def nested_sampling_single_freq(self, jf = 0, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0, dlogz=0.01):
        """ Run single freq inference
        
        Parameters
        ----------
        jf : int
            Freq index to run
        n_live : int
            The number of live points in your initial and final population.
            It will contain your initial population, and be updated. As the iterations
            go on, the likelihood of such set of points will increase until termination.
        max_iter : int
            The maximum number of iterations allowed.
        tol_evidence_increase : float
            The tolerance of evidence increase - used for iteration stop. 
            If the maximum log-likelihood multiplied by the current mass 
            does not help to increase the evidence*tol_evidence_increase,
            then the iterations can stop.
        seed : int
            seed for random number generator.
        max_up_attempts : int
            number of attempts to update the parameter set of lowest log-likelihood
            to a parameter set with higher likelihood.
        """
        self.current_k0 = self.controls.k0[jf]
        ba = BayesianSampler(measured_coords = self.receivers.coord[self.id_z_list[0],:], 
                              measured_data = self.pres_s[:, jf],
                              parameters_names = self.parameters_names,
                              num_model_par = self.num_model_par,
                              sampling_scheme = self.sampling_scheme,
                              enlargement_factor = self.enlargement_factor)
        ba.set_model_fun(model_fun = self.model_fun)
        ba.set_uniform_prior_limits(lower_bounds = self.lower_bounds, 
                                    upper_bounds = self.upper_bounds)
        ba.nested_sampling(n_live = n_live, max_iter = max_iter, 
                           max_up_attempts = max_up_attempts, seed = seed,
                           dlogz = dlogz)
        # ba.ultranested_sampling(n_live = n_live, max_iter = max_iter)
        # ba.ultranested_sampling_react(n_live = n_live, max_iter = max_iter)
        ba.compute_statistics()
        return ba
    
    def pk_bayesian(self, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0):
        
        self.ba_list = []
        
        # bar = tqdm(total=len(self.controls.k0),
        #            desc='Calculating Bayesian inversion (dDCISM)...', ascii=False)
        
        # Freq loop
        for jf, self.current_k0 in enumerate(self.controls.k0):
            print("Baysian inversion for freq {} [Hz]. {} of {}".format(self.controls.freq[jf],
                                                                        jf+1, len(self.controls.k0)))
            ba = self.nested_sampling_single_freq(jf = jf, n_live = n_live, 
                                                  max_iter = max_iter,
                                                  max_up_attempts = max_up_attempts, 
                                                  seed = seed)
            self.ba_list.append(ba)
        #     bar.update(1)
        # bar.close()
        
        
    def get_kp(self, inf_value, k0):
        """ Get the complex wave-number inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        
        Returns
        ----------
        kp : complex
            complex-valued wave-number
        """
        return k0*(inf_value[0] + 1j*inf_value[1])

    def get_rhop(self, inf_value, rho0):
        """ Get the complex density inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        rho0 : float
            air density
        
        Returns
        ----------
        rhop : complex
            complex-valued density
        """
        return rho0*(inf_value[2] + 1j*inf_value[3])

    def get_cp(self, kp, omega):
        """ Get the complex sound speed inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        omega : float
            angular frequency
        
        Returns
        ----------
        cp : complex
            complex-valued sound speed
        """
        return omega/kp
    
    def get_Zp(self, kp, rhop, omega):
        """ Get the complex characteristic impedance inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        rhop : complex
            complex-valued density
        omega : float
            angular frequency
        
        Returns
        ----------
        Zp : complex
            complex-valued characteristic impedance
        """
        return rhop*omega/kp
    
    def get_theta_t(self, kp, k0, theta):
        """ Get the complex refraction angle inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        theta : float or array
            angle of incidence in radians
        
        Returns
        ----------
        theta_t : complex
            complex-valued refraction angle
        """
        return np.arcsin((k0/kp)*np.sin(theta))
        
    def recon_vp_nlr_1layer(self, k0, rho0, inf_value, theta):
        """ NLR 1 layer planar reflection coefficient reconstruction (single freq)
        
        Reconstruct the planar reflection coefficient and its absorption coefficient.
        Valid for a single layer of non-locally reacting porous material above rigid-backing.
        It can reconstruct for multiple angles at a single frequency.  
        
        Parameters
        ----------
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        rho0 : float
            air density
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        theta : float or array
            angle of incidence in radians
        Returns
        ----------
        vp : complex
            complex-valued reflection coefficient
        alpha : floar
            absorption coefficient
        """
        # Get the wave-number / density 
        kp = self.get_kp(inf_value, k0)
        rhop = self.get_rhop(inf_value, self.air.rho0)
        # Get thickness
        if self.num_model_par == 4:
            tp = self.t_p
        elif self.num_model_par == 5:
            tp = inf_value[4]
        # Angular wave-number in air
        k0z = k0*np.cos(theta)
        # Angular wave-number in the layer
        theta_t = self.get_theta_t(kp, k0, theta)
        k1z = kp*np.cos(theta_t)
        # reflection coefficient
        num = -1j*k0z/rho0 - (k1z/rhop)*np.tan(k1z*tp)
        den = -1j*k0z/rho0 + (k1z/rhop)*np.tan(k1z*tp)
        vp = num/den
        # absorption coefficient
        alpha = 1-np.abs(vp)**2
        return vp, alpha
    
    def recon_zs_nlr_1layer(self, z0, vp, theta):
        """ NLR 1 layer surface impedance reconstruction (single freq)
        
        Reconstruct the surface impedance.
        Valid for a single layer of non-locally reacting porous material above rigid-backing.
        It can reconstruct for multiple angles at a single frequency.  
        
        Parameters
        ----------
        z0 : float
            characteristic impedance of air (rho0*c0)
        vp : complex
            complex-valued reflection coefficient
        theta : float or array
            angle of incidence in radians
        Returns
        ----------
        zs : complex
            complex-valued surface impedance
        """
        return (z0/np.cos(theta))*((1+vp)/(1-vp))
      
    def get_beta(self, inf_value):
        """ Get the complex surface admittance inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        
        Returns
        ----------
        beta : complex
            complex-valued surface admittance
        """
        return inf_value[0] + 1j*inf_value[1]

    def recon_vp_lr(self, k0, inf_value, theta):
        """ LR planar reflection coefficient reconstruction (single freq)
        
        Reconstruct the planar reflection coefficient and its absorption coefficient.
        Valid for a locally reacting sample.
        It can reconstruct for multiple angles at a single frequency.  
        
        Parameters
        ----------
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        theta : float or array
            angle of incidence in radians
        Returns
        ----------
        vp : complex
            complex-valued reflection coefficient
        alpha : floar
            absorption coefficient
        """
        # get surface admittance
        beta = self.get_beta(inf_value)
        # Angular wave-number in air
        k0z = k0*np.cos(theta)
        # reflection coefficient
        vp = (k0z - k0*beta)/(k0z + k0*beta)
        # absorption coefficient
        alpha = 1-np.abs(vp)**2
        return vp, alpha
    
    def kp_rhop_range(self, resist = [3000, 60000], phi = [0.15, 1.0],
                      alpha_inf = [1.0, 3.00], Lam = [50e-6, 300e-6],
                      Lamlfac = [1.01, 3.0],
                      thickness = [2e-3, 20e-2], theta_deg = [0, 75],
                      n_samples = 20000):
        """ Run a JCA range study to determine useful ranges for kp and rhop
        """
        k_p_samples, rho_p_samples, beta_samples = mat.kp_rhop_range_study(
            resist = resist, phi = phi, alpha_inf = alpha_inf, Lam = Lam,
            Lamlfac = Lamlfac, thickness = thickness, theta_deg = theta_deg, 
            n_samples = n_samples, freq_vec = self.controls.freq)
        if self.chosen_model == 0:
            self.lb_mtx = mat.get_min_beta(beta_samples)
            self.ub_mtx = mat.get_max_beta(beta_samples)
        else:
            self.lb_mtx = mat.get_min_kp_rhop(k_p_samples, rho_p_samples)
            self.ub_mtx = mat.get_max_kp_rhop(k_p_samples, rho_p_samples)
    
    def plot_search_space(self,):
        """ plots the search space as a function of frequency (sanity check)
        """
        if self.chosen_model == 0 or self.chosen_model == 1:
            figshape = (1, 2)
            figsize = (8, 4)
            fig, ax = ut_is.give_me_an_ax(figshape, figsize)
            beta_l = self.lb_mtx[0,:] + 1j*self.lb_mtx[1,:]
            ax = ut_is.plot_spk_re_imag(self.controls.freq, beta_l, ax = ax, 
                                   xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                            ylims = None, 
                                            color = 'k', linewidth = 1.5, linestyle = '--',
                                            alpha = 1.0, label = None)
            beta_u = self.ub_mtx[0,:] + 1j*self.ub_mtx[1,:]
            ax = ut_is.plot_spk_re_imag(self.controls.freq, beta_u, ax = ax, 
                                        xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                        ylims = None, 
                                        color = 'k', linewidth = 1.5, linestyle = '--',
                                        alpha = 1.0, label = None)
            # Physical limits
            ax[0,0].plot(self.controls.freq, 0.001+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            # Fill
            ax[0,0].fill_between(self.controls.freq, np.real(beta_l), np.real(beta_u), 
                                 color='grey', alpha = 0.5)
            ax[0,1].fill_between(self.controls.freq, np.imag(beta_l), np.imag(beta_u), 
                                 color='grey', alpha = 0.5)
            # Labels
            ax[0,0].set_ylim((-0.1, np.amax(np.real(beta_u)))) 
            ax[0,1].set_ylim((np.amin(np.imag(beta_l)), np.amax(np.imag(beta_u))))
            ax[0,0].set_xlabel("Frequency [Hz]")
            ax[0,1].set_xlabel("Frequency [Hz]")
            ax[0,0].set_ylabel(self.parameters_names[0])
            ax[0,1].set_ylabel(self.parameters_names[1])
            plt.tight_layout()
        else:
            figshape = (2, 2)
            figsize = (8, 6)
            fig, ax = ut_is.give_me_an_ax(figshape, figsize)
            # plot kp
            kp_l = self.lb_mtx[0,:] + 1j*self.lb_mtx[1,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(kp_l), ax[0,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(kp_l), ax[0,1], 
                                     color = 'k', linestyle = '--', linx = False)
            kp_u = self.ub_mtx[0,:] + 1j*self.ub_mtx[1,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(kp_u), ax[0,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(kp_u), ax[0,1], 
                                     color = 'k', linestyle = '--', linx = False)
            
            # plot rhop
            rhop_l = self.lb_mtx[2,:] + 1j*self.lb_mtx[3,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(rhop_l), ax[1,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(rhop_l), ax[1,1], 
                                     color = 'k', linestyle = '--', linx = False)
            rhop_u = self.ub_mtx[2,:] + 1j*self.ub_mtx[3,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(rhop_u), ax[1,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(rhop_u), ax[1,1], 
                                     color = 'k', linestyle = '--', linx = False)
            
            # Physical limits
            ax[0,0].plot(self.controls.freq, np.ones(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[0,1].plot(self.controls.freq, -0.01+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[1,0].plot(self.controls.freq, 0.9*np.ones(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[1,1].plot(self.controls.freq, -0.01+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            # Fill
            ax[0,0].fill_between(self.controls.freq, np.real(kp_l), np.real(kp_u), 
                                 color='grey', alpha = 0.5)
            ax[0,1].fill_between(self.controls.freq, np.imag(kp_l), np.imag(kp_u), 
                                 color='grey', alpha = 0.5)
            ax[1,0].fill_between(self.controls.freq, np.real(rhop_l), np.real(rhop_u), 
                                 color='grey', alpha = 0.5)
            ax[1,1].fill_between(self.controls.freq, np.imag(rhop_l), np.imag(rhop_u), 
                                 color='grey', alpha = 0.5)
            # labels
            ax[0,0].set_ylim((0.5, np.amax(np.real(kp_u)))) 
            ax[0,1].set_ylim((np.amin(np.imag(kp_l)), 0.1))
            ax[1,0].set_ylim((0.9, np.amax(np.real(rhop_u)))) 
            ax[1,1].set_ylim((np.amin(np.imag(rhop_l)), 0.1))
            ax[1,0].set_xlabel("Frequency [Hz]")
            ax[1,1].set_xlabel("Frequency [Hz]")
            ax[0,0].set_ylabel(self.parameters_names[0] + r'$/k_0$')
            ax[0,1].set_ylabel(self.parameters_names[1] + r'$/k_0$')
            ax[1,0].set_ylabel(self.parameters_names[2] + r'$/\rho_0$')
            ax[1,1].set_ylabel(self.parameters_names[3] + r'$/\rho_0$')
            plt.tight_layout()
            
    
    # def get_kp_rhop(self, ba, mode = "mean"):
    #     if mode == "mean":
    #         kp = self.current_k0*(ba.mean_values[0] + 1j*ba.mean_values[1])
    #         rhop = self.air.rho0*(ba.mean_values[2] + 1j*ba.mean_values[3])
    #     elif mode == "lower_ci":
    #         kp = self.current_k0*(ba.ci[0,0] + 1j*ba.ci[0,1])
    #         rhop = self.air.rho0*(ba.ci[0,2] + 1j*ba.ci[0,3])
    #     elif mode == "upper_ci":
    #         kp = self.current_k0*(ba.ci[1,0] + 1j*ba.ci[1,1])
    #         rhop = self.air.rho0*(ba.ci[1,2] + 1j*ba.ci[1,3])
    #     else:
    #         kp = self.current_k0*(ba.mean_values[0] + 1j*ba.mean_values[1])
    #         rhop = self.air.rho0*(ba.mean_values[2] + 1j*ba.mean_values[3])
    #     return kp, rhop
    
    # def recon_mat_props_nlr(self, ba, id_f, theta, mode = "mean"):
    #     """ From infereed kp and rhop, reconstruct Zp, cp
        
    #     Single frequency estimation (can be multiple angle)
        
    #     Parameters
    #     ----------
    #     ba : object
    #         Bayesian inference object
    #     id_f : int
    #         freq index
    #     theta : float or numpy1dArray
    #         angle of incidence in radians
    #     """
    #     # Chose thickness according to model
    #     if self.chosen_model == 5:
    #         tp = ba.mean_values[4]
    #     else:
    #         tp = self.t_p
    #     # Wave-number and density from mean values
    #     kp, rhop = self.get_kp_rhop(ba, mode = mode)
    #     # Characteristic and surface impedance
    #     Zp, Zs = self.imp_recon_nlr(w = self.controls.w[id_f], kp = kp,
    #                                 rhop = rhop, theta = theta, tp = tp)
    #     # Reflection and absorption coefficient
    #     Vp, alpha = self.ref_abs_coeff(Zs, theta)
    #     return kp, rhop, Zp, Zs, Vp, alpha
    
    # def imp_recon_nlr(self, w, kp, rhop, theta, tp):
    #     """ Reconstruct Zp and Zs
        
    #     Parameters
    #     ----------
    #     w : float
    #         angular frequency in rad/s
    #     kp : float complex
    #         Wave-number in porous media. It can come from mean or confidence interval
    #     rhop : float complex
    #         Density in porous media. It can come from mean or confidence interval
    #     """
    #     # Characteristic impedance
    #     Zp = rhop * w/kp
    #     # k0
    #     k0 = w/self.air.c0
    #     # refraction index
    #     n_index = kp/k0
    #     theta_t = np.arcsin(np.sin(theta)/n_index)
    #     kzp = kp * np.cos(theta_t)
    #     Zs = -1j * Zp * (kp/kzp) *(1/np.tan(kzp*tp))
    #     return Zp, Zs
    
    # def ref_abs_coeff(self, Zs, theta):
    #     """ From infereed Zs, reconstruct Vp, alpha
        
    #     Single frequency estimation (can be multiple angle)
        
    #     Parameters
    #     ----------
    #     ba : object
    #         Bayesian inference object
    #     theta : float or numpy1dArray
    #         angle of incidence in radians
    #     """
    #     # Reflection and absorption coefficient
    #     Vp = (Zs*np.cos(theta) - (self.air.rho0 * self.air.c0)) /\
    #         (Zs*np.cos(theta) + (self.air.rho0 * self.air.c0))
    #     alpha = 1-(np.abs(Vp))**2
    #     return Vp, alpha