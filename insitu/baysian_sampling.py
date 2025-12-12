# -*- coding: utf-8 -*-
"""
Created on Sep  2025

@author: Eric Brandão - Bayesian sampler class
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils_insitu as ut_is
try:
    import ultranest
    from ultranest.popstepsampler import PopulationSimpleSliceSampler, PopulationRandomWalkSampler
    from ultranest.mlfriends import (AffineLayer, LocalAffineLayer, MLFriends,
                            RobustEllipsoidRegion, ScalingLayer, WrappingEllipsoid,
                            find_nearby)
except:
    print("Not possible to use Ultranest in this environment")
        

class BayesianSampler(object):
    """ Bayesian sampling analysis of inverse problems
    """
    
    def __init__(self, measured_coords = None, measured_data = None, 
                 num_model_par = 1, parameters_names = None,
                 likelihood = None, sampling_scheme = 'slice'):
        """ Bayesian sampling schemes
        
        Parameters
        ----------
        measured_coords : numpy1dArray
            Array containing the measured coordinates 
        measured_data : numpy1dArray
            Array containing the measured data
        num_model_par : int
            The number of model parameters
        parameters_names: list
            List containing strings with the parameters' names.
        model_fun : function callable
            Function containing your model. Will be called multiple times during inference
            Its input are the measured coordinates and the
        likelihood : str or None
            likelihood type function
        sampling_scheme : str
            Desired sampling scheme for nested sampling. The possibilites are:
                "slice" (default - slice sampling), "random walk" (random walk)
                "constrained" (for random parameter update). Given a choice, 
                the update function is selected.
        """
        # Make sure there is measured data
        if measured_coords is None or measured_data is None:
            raise ValueError("It is necessary to input your measured coordinates and data. Otherwise, I can not do anything!")
        elif measured_coords.shape[0] != len(measured_data):
            raise ValueError("Measured coordinates length and data length must be the same!")
        # Make sure parameters name has the same size of num_par
        elif len(parameters_names) != num_model_par:
            raise ValueError("The parameters_names list must be a list with same size of num_par")
        else:
            self.measured_coords = measured_coords
            self.measured_data = measured_data
            self.num_of_meas = len(measured_data)
            self.num_model_par = num_model_par
            self.parameters_names = parameters_names
            
        if likelihood == "Student-t" or likelihood == None:
            self.likelihood_fun = self.log_t
        elif likelihood == "Gaussian1D":
            self.likelihood_fun = self.log_normal_1d
            self.set_log_normal_1d_std()
        else:
            self.likelihood_fun = self.log_t
        
        if sampling_scheme == 'slice':
            self.sample_update_fun = self.constrained_resample_slice
        elif sampling_scheme == 'random walk':
            self.sample_update_fun = self.constrained_resample_rw
        elif sampling_scheme == 'constrained':
            self.sample_update_fun = self.constrained_resample
        else:
            self.sample_update_fun = self.constrained_resample_slice
            # self.seed = seed
            # self.rng = np.random.default_rng(seed)
        # Make sure there is model function and get number of parameters to estimate
        # if not callable(model_fun):    
        #     raise ValueError("model_fun must be a callable function!")
        # else:
        #     self.model_fun = model_fun
        #     self.num_par = self.model_fun.__code__.co_argcount - 1
            
    def set_model_fun(self, model_fun):
        """ Set the model function which will be called multiple times at inference
        
        This function is called to evaluate the likelihood for a given set of model parameters.
        The function itself should have a structured input such as
        
        "def line_model(x_meas, model_par = [1, 0])"
        
        where x_meas stands for the measurement coordinates and model_par is a vector containing
        the guesses for the values in the parameter space. *kwargs may also apply depending
        on the model. The output should return a y_pred (prediction at x_coord)
        
        Parameters
        ----------
        model_fun : function callable
            Function containing your model. Will be called multiple times during inference
            Its input are the measured coordinates and the
        """
        if not callable(model_fun):    
            raise ValueError("model_fun must be a callable function!")
        else:
            self.model_fun = model_fun
            
    def set_uniform_prior_limits(self, lower_bounds = [0], upper_bounds = [1]):
        """ Set the uniform prior limits
        
        Parameters
        ----------
        lower_bonds : numpy1dArray
            Array containing the lower bonds of the parameter space
        upper_bonds : numpy1dArray
            Array containing the upper bonds of the parameter space
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(lower_bounds) != self.num_model_par or len(upper_bounds) != self.num_model_par:
            raise ValueError("Lower and Upper bonds must be a vector with the same size of num_par")
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
    def set_convergence_tolerance(self, convergence_tol = [0.01]):
        """ Set convergence tolerance for nested sampling.
        
        This can be used to stop the nested sampling algorithm. When the spread
        of the current live point population becomes smaller than the convergence_tol,
        then, the algorithm has converged.
        
        Parameters
        ----------
        convergence_tol : numpy1dArray
            Array containing the tolerance values for each of the model's parameters
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(convergence_tol) != self.num_model_par:
            raise ValueError("convergence_tol must be a vector with the same size of num_par")
        self.convergence_tol = convergence_tol
    
    def sample_uniform_prior(self, num_samples = 1):
        """ Sample uniform prior - all that is available for now
        
        Draw a number of samples from a uniform prior volume.
        
        Parameters
        ----------
        num_samples : int
            number of sampling points
        
        Returns
        ----------
        prior_samples : numpyndArray
            samples of the uniform prior distribution with dimensions (num_pts, num_model_par)
        """
        # prior_samples = np.random.uniform(low = self.lower_bounds, 
        #                                   high = self.upper_bounds,
        #                                   size = (num_samples, self.num_model_par))
        
        prior_cube = np.random.uniform(low = np.zeros(self.num_model_par), 
                                          high = np.ones(self.num_model_par),
                                          size = (num_samples, self.num_model_par))
        prior_samples = np.zeros(prior_cube.shape)
        for jp in range(self.num_model_par):
            prior_samples[:, jp] = prior_cube[:, jp] *\
                (self.upper_bounds[jp]-self.lower_bounds[jp]) + self.lower_bounds[jp]
        return prior_samples, prior_cube
    
    def set_log_normal_1d_std(self, sigma = 1):
        """ Set std for 1D normal distribution
        """
        self.sigma = sigma
    
    def log_normal_1d(self, model_par):
        """ Computes log-like Normal likelihood (1D distribution)
        
        Parameters
        ----------
        model_par : numpy1dArray
            parameters for a giving tried solution vector
            
        Returns
        ----------
        logp : float
            Value of log - like Student-t distribution
        
        """
        # mu = np.array([1.0, -1.0])
        # d = model_par - mu
    
        # Sigma = np.array([[1.0, 0.8],
        #           [0.8, 1.5]])
        # invSigma = np.linalg.inv(Sigma)
        # return -0.5 * d @ invSigma @ d
        y_pred = self.model_fun(x_meas = self.measured_coords, 
                                model_par = model_par)
        # error norm
        error_norm = np.linalg.norm(self.measured_data - y_pred)
        logp = - 0.5 * self.num_of_meas * np.log(2*np.pi*self.sigma*2) -\
            error_norm**2/(2*self.sigma**2)
        # logp = -0.5*self.num_of_meas * np.log(2*np.pi) -\
        #     self.num_of_meas * np.log(self.sigma) -\
        #     0.5*error_norm**2/(self.sigma**2)

        return logp
        
    def log_t(self, model_par):
        """ Computes log-like Student-t likelihood
        
        Parameters
        ----------
        model_par : numpy1dArray
            parameters for a giving tried solution vector
            
        Returns
        ----------
        logp : float
            Value of log - like Student-t distribution
        
        """
        # prediction
        y_pred = self.model_fun(x_meas = self.measured_coords, 
                                model_par = model_par)
        # error norm
        error_norm = np.linalg.norm(self.measured_data - y_pred)
        logp = -(self.num_of_meas/2)*np.log((error_norm**2)/2)
        return logp
    
    def brute_force_sampling_bar(self, num_samples = 1000):
        """ Brute force sample the parameter space.
        
        This method samples the uniform prior and then computes the associated
        likelihhod for each sample. It returns the prior samples, normalized weights, 
        and the log-likelihood. It runs a bar with the progress of the computation.
        It is supposed to be used mainly for demonstration of early stages of Bayesian
        inference.
        
        Parameters
        ----------
        num_samples : int
            total number of samples to draw
        Returns
        ---------
        prior_samples : numpyndArray
            Array containing the samples of your prior. Its dimensions will be
            num_model_par x num_samples.
        weights : numpy1dArray
            Normalized weights (linear scale).
        logp
            log-likelihood values.
        """
        self.prior_samples, self.prior_cube = self.sample_uniform_prior(num_samples = num_samples)
        # Initialize logp
        logp = np.zeros(num_samples)
        # bar
        bar = tqdm(total = num_samples,
                   desc='Brute force sampling...', ascii=False)
        # Compute log-likelihood
        for jp, pr_sam in enumerate(self.prior_samples):
            logp[jp] = self.likelihood_fun(model_par = pr_sam)
            bar.update(1)
        bar.close()
        # Compute likelihood normalized weights
        self.weights = np.exp(logp - np.max(logp))
        self.weights /= np.sum(self.weights)
        return self.prior_samples, self.prior_cube, self.weights, logp      
    
    def brute_force_sampling(self, num_samples = 1000):
        """ Brute force sampling of the parameter space.
        
        This method samples the uniform prior and then computes the associated
        likelihhod for each sample. It returns the prior samples, normalized weights, 
        and the log-likelihood.
        
        Parameters
        ----------
        num_samples : int
            total number of samples to draw
        Returns
        ---------
        prior_samples : numpyndArray
            Array containing the samples of your prior. Its dimensions will be
            num_model_par x num_samples.
        weights : numpy1dArray
            Normalized weights (linear scale).
        logp
            log-likelihood values.
        """
        # sample values from prior.
        self.prior_samples, self.prior_cube = self.sample_uniform_prior(num_samples = num_samples)
        # Initialize logp
        logp = np.zeros(num_samples)
        # Compute log-likelihood
        for jp, pr_sam in enumerate(self.prior_samples):
            logp[jp] = self.likelihood_fun(model_par = pr_sam)
        # Compute likelihood normalized weights
        self.weights = np.exp(logp - np.max(logp))
        self.weights /= np.sum(self.weights)
        return self.prior_samples, self.prior_cube, self.weights, logp 
    
    def update_values(self, random_par, adjust_suggested, current_worst_par_val,
                      type_of_update = "adding"):
        """ Update a given parameter value and logp from a set of parameter vector 
        """
        
        # Update (Adding)
        if type_of_update == "adding":
            updated_sample_par = current_worst_par_val[random_par] + adjust_suggested
        else:
            updated_sample_par = current_worst_par_val[random_par] - adjust_suggested
        updated_model_par = current_worst_par_val
        updated_model_par[random_par] = updated_sample_par
        updated_logp = self.likelihood_fun(updated_model_par)
        return updated_logp, updated_model_par       
    
    
    def get_worst_logp(self, it_num = 0):
        """ Find the worst log-likelihood value in the current live pts population.
        
        Finds the worst log-likelihood value in the current live pts population, 
        then fill this log-likelihood value and associated parameters to the 
        correct place in the dead points population.
        
        This is Step 1 (Beaton and Xiang - Sec. IV.B.1)
        
        Parameters
        ----------
        it_num : int
            iteration number
        """
        # Find index of worst log-likelihood and update dead_pts and logp_dead
        self.worst_logp_index = np.argmin(self.logp_live)
        self.logp_dead[it_num] = self.logp_live[self.worst_logp_index]
        self.dead_pts[it_num, :] = self.live_pts[self.worst_logp_index, :]

    def compute_spread(self, pop_pts):
        """ Compute the spread (max-min) of a population of sampled values.
        
        It allows one to evaluate the shrinking of the live points population.
        
        Parameters
        ----------
        pop_pts : numpyndArray
            A population of points in parameter space
        """
        spread = np.amax(pop_pts, axis = 0) - np.amin(pop_pts, axis = 0)
        return spread
    
    def evaluate_move(self, logp_new, parameter_vec_new):
        """ Evaluate if the proposed move increases likelihood
        
        Evaluate if the proposed move generated a higher likelihood. If so,
        then substitute the new log-likelihood to the worst one. Do the same
        with the new parameter vector. Also updates the variable
        self.update_sucessful to True to indicate that the worst log-likelihood
        and its parameter vector were updated - allowing the algorithm to move on.
        
        Parameters
        ----------
        logp_new : float
            The value of the log-likelihhod for the proposed parameter vector.
        parameter_vec_new : numpy1dArray
            The proposed parameter vector.
        """
        if logp_new > self.logp_live[self.worst_logp_index]:
            self.logp_live[self.worst_logp_index] = logp_new
            self.live_pts[self.worst_logp_index, :] = np.copy(parameter_vec_new)
            self.update_successful = True
            self.num_evals_sucessful += 1
    
    def suggest_random_adjustment(self, spread):
        """ Suggest a random adjustment for a single random parameter
        
        These are Steps 2-3 (Beaton and Xiang - Sec. IV.B.1)
        
        1. Select a random parameter in the reference point.
        2. Use a uniform distribution to generate a random adjustment
        value for the parameter. The uniform distribution
        will have limits from zero to a maximum value dependent
        on the type of parameter selected. The MAX-MIN (spread/2) of the
        current live_pts population is used as suggestion for this update.
        
        Parameters
        ----------
        spread : numpy1dArray
            The Array containing the spread of the current population
        
        Returns
        ----------
        random_par_id : int
            index of the parameter to update. It will be between 0 and num_model_par-1
        suggested_adjustment : float
            value of ajustment.
        """
        # Select a random parameter index
        random_par_id = np.random.randint(low = 0, high = self.num_model_par, size = 1)[0]
        # Generate a random adjustment value for the parameter (uniform dist)
        suggested_adjustment = np.random.uniform(low = 0,
                                                 high = spread[random_par_id]/2, 
                                                 size = 1)[0]
        return random_par_id, suggested_adjustment
    
    def update_parameter(self, parameter_vec, 
                         random_par_id = 0, suggested_adjustment = 0.0):
        """ Updates the choosen parameter.
        
        We need to decide if we add or subtract the random suggestion. To do so, 
        we will calculate which is the closest border of the parameter we want to alter.
        If the parameter is closer to the lower bound we add. If the parameter is closer
        to the upper bound we subtract.
        
        This is Step 4 (Beaton and Xiang - Sec. IV.B.1)
        
        Parameters
        ----------
        parameter_vec : numpy1dArray
            Parameter vector to adjust
        random_par_id : int
            Parameter to update
        suggested_adjustment : float
            Value to add or subtract from current parameter.
        """
        # Get parameter vector to ajust (this is a guessed solution)
        suggested_parameter_vec = np.copy(parameter_vec)
        # compute distances from upper and lower bounds
        lower_bound_distance = parameter_vec[random_par_id] - self.lower_bounds[random_par_id]
        upper_bound_distance = self.upper_bounds[random_par_id] - parameter_vec[random_par_id]
        # Summing or subtracting
        if lower_bound_distance <= upper_bound_distance:
            suggested_parameter_vec[random_par_id] += np.abs(suggested_adjustment)
        else:
            suggested_parameter_vec[random_par_id] -= np.abs(suggested_adjustment)
        return suggested_parameter_vec
    
    def update_from_lowest_likelihood(self, current_spread, max_up_attempts):
        """ Update the parameter vector of lowest likelihood.
        
        Updates the parameter of lowest likelihood and computes the new log likelihood.
        It uses a random suggestion approach. A random index of the parameter vector is
        selected and a random number is also generated serving as an update suggestion, 
        which will be either summed or subtracted from the parameter vector value.
        The suggestion is evaluated. If the move is sucessful (higher likelihood), it stops.
        If not, a new suggestion is performed.
        
        Parameters
        ----------
        current_spread : numpy1dArray
            The Array containing the spread of the current population
        max_up_attempts : int
            Maximum number of update attempts.
        """
        attempt_num = 0 # number of attempts made
        while attempt_num <= max_up_attempts and not self.update_successful:
            # Get a random adjustment suggestion        
            random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
            self.random_par_id.append(random_par_id)
            # Get a parameter vec suggestion
            parameter_vec_new =\
                self.update_parameter(parameter_vec = self.live_pts[self.worst_logp_index],
                                      random_par_id = random_par_id, 
                                      suggested_adjustment = suggested_adjustment)
            # Evaluate logp of suggested parameter vector
            logp_new = self.likelihood_fun(model_par = parameter_vec_new)
            # Evaluate if the move increased likelihood
            self.evaluate_move(logp_new, parameter_vec_new)
            attempt_num += 1
            
    def update_from_random_par_vec(self, current_spread):
        """ Update from a random parameter vector from the remaining live-pts population.
        
        Try to update a random parameter vector from the the remaining live-pts population,
        excluding the one with the lowest likelihood. This is only tried after trying to 
        update the parameter vector with lowest likelihood for a number of trials.
        It uses a random suggestion approach as in "update_from_lowest_likelihood"
        
        1 - Exclude the lowest likelihood parameter from the pool.
        2 - Shuffle remaining index (for a random ordering)
        3 - Try to update until sucessful.
        
        Parameters
        ----------
        current_spread : numpy1dArray
            The Array containing the spread of the current population
        """
        # Get all indexes from the live_pts population
        all_indexes = np.arange(0, self.live_pts.shape[0])
        # Delete index with the lowest likelihood
        remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
        # suffle remaining indexes
        np.random.shuffle(remaining_indexes)
        # Start loop
        i = 0
        while i < len(remaining_indexes) and not self.update_successful:
            # Get a adjustment suggestion
            random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
            self.random_par_id.append(random_par_id)
            # Get a parameter suggestion
            parameter_vec_new =\
                self.update_parameter(parameter_vec = self.live_pts[remaining_indexes[i]],
                                      random_par_id = random_par_id, 
                                      suggested_adjustment = suggested_adjustment)
            # Evaluate logp of suggested parameter vector
            logp_new = self.likelihood_fun(model_par = parameter_vec_new)
            self.evaluate_move(logp_new, parameter_vec_new)
            # increment i
            i += 1
    
    def constrained_resample(self, i, max_up_attempts = 5):
        """ Sample movement in the parameter space.
        
        1 - Computes the spread in the current live-pts population
        2 - Try to update from lowest likelihood (using random parameter suggestion)
        3 - Try to update from random parameter vector (using random parameter suggestion)
        
        Parameters
        ----------
        i : int
            iteration value
        max_up_attempts : int
            Maximum number of update attempts.
        """
        # Compute current spread in the given live_pts population
        self.spread[i,:] = self.compute_spread(self.live_pts)
        # initialize while loop variables
        self.update_successful = False # success of update is set to False first
        # Try updating the parameter vector with lowest likelihood
        self.update_from_lowest_likelihood(self.spread[i,:], max_up_attempts)
        # If still not sucessful after max_up_attempts, then try updating a random parameter.
        if not self.update_successful:
            self.update_from_random_par_vec(self.spread[i,:])
                
    def update_rw(self, pt_of_origin, lower_bounds, upper_bounds,
                  max_up_attempts):
        """ Update the parameter vector of lowest likelihood with a Random walk.
        
        Updates the parameter of lowest likelihood and computes the new log likelihood.
        It uses a random walk approach. It walks the point of origin at a random direction
        and random distance, then evaluates the move. If the move is sucessful, it stops.
        If not, move again starting from the previous point of destination.
        
        Parameters
        ----------
        pt_of_origin : numpy1dArray
            The parameter vector where you start your random walk. This can either be
            the parameter vector with lowest likelihood or a random parameter vector
            from the current live point population.
        lower_bounds : numpy1dArray
            current lower bounds vector
        upper_bounds : numpy1dArray
            current upper bounds vector
        max_up_attempts : int
            Maximum number of update attempts. 
        Returns
        ----------
        logp_new : float
            New log-likelihood value
        parameter_vec_new : numpy 1dArray
            New samples value
        """
        # init attempt_num
        attempt_num = 0 # number of attempts made
        # Origin of random walk
        parameter_vec_new = np.copy(pt_of_origin)
        while attempt_num <= max_up_attempts and not self.update_successful:
            # Get a random adjustment by random walking the original point
            parameter_vec_new, _ =\
                ut_is.random_move(origin = parameter_vec_new, 
                                  lower_bounds = lower_bounds, 
                                  upper_bounds = upper_bounds)
            # Evaluate logp of suggested parameter vector
            logp_new = self.likelihood_fun(model_par = parameter_vec_new)
            self.num_evals += 1
            # Evaluate if the move increased likelihood
            self.evaluate_move(logp_new, parameter_vec_new)
            # increase attempt_num
            attempt_num += 1
            
    def constrained_resample_rw(self, i, max_up_attempts = 5):
        """ Sample movement in the parameter space with random walk.
        
        1 - Computes the spread in the current live-pts population
        2 - Try to update from lowest likelihood (using random parameter suggestion)
        3 - Try to update from random parameter vector (using random parameter suggestion)
        
        Parameters
        ----------
        i : int
            iteration value
        max_up_attempts : int
            Maximum number of update attempts.
        """
        # Compute current spread in the given live_pts population
        self.spread[i,:] = self.compute_spread(self.live_pts)
        # Compute bounds given the current live points population
        lower_bounds = 1.00*np.amin(self.live_pts, axis = 0)
        upper_bounds = 1.00*np.amax(self.live_pts, axis = 0)
        # initialize while loop variables
        self.update_successful = False # success of update is set to False first
        # Try updating the point with lowest likelihood.
        self.update_rw(pt_of_origin = self.live_pts[self.worst_logp_index], 
                       lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                       max_up_attempts = max_up_attempts)
        # If not sucessful - try updating starting from a random parameter vector
        if not self.update_successful:
            # Get remaining indexes - excluding the one with lowest likelihood.
            all_indexes = np.arange(0, self.live_pts.shape[0])
            remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
            # Sample a random point in the current pop of live pts
            pt_of_origin_id = np.random.randint(low = 0, high = len(remaining_indexes),
                                                size = 1)[0]
            pt_of_origin = self.live_pts[remaining_indexes[pt_of_origin_id],:]
            # Try random walk
            self.update_rw(pt_of_origin = pt_of_origin, 
                           lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                           max_up_attempts = max_up_attempts)
            
    def constrained_resample_rw_mcmc(self, i, max_up_attempts = 100):
        """ Sample movement in the parameter space with random walk MCMC.
        
        1 - Computes the spread in the current live-pts population
        2 - Try to update from lowest likelihood (using random parameter suggestion)
        3 - Try to update from random parameter vector (using random parameter suggestion)
        
        Parameters
        ----------
        i : int
            iteration value
        max_up_attempts : int
            Maximum number of update attempts.
        """
        # Compute current spread in the given live_pts population
        self.spread[i,:] = self.compute_spread(self.live_pts)
        # Compute bounds given the current live points population
        lower_bounds = np.amin(self.live_pts, axis = 0)
        upper_bounds = np.amax(self.live_pts, axis = 0)
        # initialize while loop variables
        self.update_successful = False # success of update is set to False first
        attempt_num = 0
        while not self.update_successful and attempt_num <= self.live_pts.shape[0]-1:
            # Get remaining indexes - excluding the one with lowest likelihood.
            all_indexes = np.arange(0, self.live_pts.shape[0])
            remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
            # Sample a random point in the current pop of live pts
            pt_of_origin_id = np.random.randint(low = 0, high = len(remaining_indexes),
                                                size = 1)[0]
            pt_of_origin = self.live_pts[remaining_indexes[pt_of_origin_id],:]
            # Try random walk
            self.update_rw_mcmc(pt_of_origin = pt_of_origin, 
                           lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                           num_of_movs = 20, max_up_attempts = max_up_attempts)
            attempt_num += 1
            
    def update_rw_mcmc(self, pt_of_origin, lower_bounds, upper_bounds,
                       num_of_movs = 20, max_up_attempts = 100):
        """ Update the parameter vector of lowest likelihood with a Random walk MCMC.
        
        Updates the parameter of lowest likelihood and computes the new log likelihood.
        It uses a random walk approach. It walks the point of origin at a random direction
        and random distance, then evaluates the move. If the move is sucessful, it stops.
        If not, move again starting from the previous point of destination.
        
        Parameters
        ----------
        pt_of_origin : numpy1dArray
            The parameter vector where you start your random walk. This can either be
            the parameter vector with lowest likelihood or a random parameter vector
            from the current live point population.
        lower_bounds : numpy1dArray
            current lower bounds vector
        upper_bounds : numpy1dArray
            current upper bounds vector
        num_of_movs : int
            Number of movements before acceptance (for decorrelation). 
        Returns
        ----------
        logp_new : float
            New log-likelihood value
        parameter_vec_new : numpy 1dArray
            New samples value
        """
        attempt_num = 0
        # init attempt_num
        move_num = 0 # number of attempts made
        # Origin of random walk
        parameter_vec_new = np.copy(pt_of_origin)
        while move_num <= num_of_movs or attempt_num <= max_up_attempts:
            # Get a random adjustment by random walking the original point
            parameter_vec_new, _ =\
                ut_is.random_move(origin = parameter_vec_new, 
                                  lower_bounds = lower_bounds, 
                                  upper_bounds = upper_bounds)
            # Evaluate logp of suggested parameter vector
            logp_new = self.likelihood_fun(model_par = parameter_vec_new)
            # # Evaluate if the move increased likelihood
            # self.evaluate_move(logp_new, parameter_vec_new)
            if logp_new > self.logp_live[self.worst_logp_index]:
                move_num += 1
                # self.update_successful = True
            # else:
            #     self.update_successful = False
            # increase attempt_num
            attempt_num += 1
            if move_num>=num_of_movs:
                self.num_evals += 1
        if logp_new > self.logp_live[self.worst_logp_index]:
            self.logp_live[self.worst_logp_index] = logp_new
            self.live_pts[self.worst_logp_index, :] = np.copy(parameter_vec_new)
            self.update_successful = True
            self.num_evals_sucessful += 1
            
    def slice_step(self, it_num, theta, win_size = 0.2, max_up_attempts = 200):
        """ Move a parameter vector according to slice sampling.
        
        1. Get a random direction in which to move the parameter vector
        2. Get a Min and Max values for a given window size
        3. Draw a random distance between -0.5*win_size and 0.5*win_size.
        4. Move the parameter vector by a random distance in the random direction.
        5. Evaluate if proposed parameter lies inside the bounds of the prior distribution.
            If it is outiside, shrink the maximum possible distance by changing either
            u_min or u_max and sampling/moving again.
        6. Evaluate the log-likelihood. If update is sucessful, we are done. If not - SHADY 
        
        Parameters
        ----------
        it_num : int
            iteration value
        win_size: float
            Size of the slice window.
        max_up_attempts : int
            Maximum number of update attempts.   
        """
        # Get the parameter vector
        theta = np.copy(theta)
        # Get a random direction in which to move the parameter vector.
        direction = ut_is.get_random_direction(num_dim = self.num_model_par)
        # Min and max values for a given window size
        u_min, u_max = -0.5*win_size, 0.5*win_size
        for _ in range(max_up_attempts):
            # Draw a random distance between u_min and u_max.
            rand_dist = np.random.uniform(low = u_min, high = u_max, size = 1)
            # Move the parameter vector by a random distance in the random direction.
            theta_trial = theta + rand_dist * direction
            # Evaluate if proposed parameter lies inside the bounds of the prior distribution
            if np.any(theta_trial < self.lower_bounds) or np.any(theta_trial > self.upper_bounds):
                # If not, set either the u_max or u_min to the random distance value. 
                if rand_dist > 0:
                    u_max = rand_dist
                else: 
                    u_min = rand_dist
                continue
            # Evaluate log-likelihood.
            new_logp = self.likelihood_fun(model_par = theta_trial)
            if new_logp > self.logp_dead[it_num]:
                return theta_trial
            else:
                if rand_dist > 0: 
                    u_max = rand_dist
                else: 
                    u_min = rand_dist
        return theta
    
    def constrained_resample_slice(self, it_num = 0, max_up_attempts = 200):
        """ Sample movement in the parameter space with slice sampling.
        
        1 - Computes the spread in the current live-pts population
        2 - Computes a window size as the maximum value of the current spread.
        3 - Select a random candidate to move from the current live point population
        4 - Update selected parameter according to slice-sampling.
        5- Evaluate the log-likelihood. If larger then the worst one, the move is complete
            and the loop stops. This is tried by a maximum number of attempts. If fails,
            sample unifom prior
        
        Parameters
        ----------
        it_num : int
            iteration value
        max_up_attempts : int
            Maximum number of update attempts.        
        """
        # Compute current spread
        self.spread[it_num,:] = self.compute_spread(self.live_pts)
        # Compute the window size as the max value in the spread
        win_size = np.amax(self.spread[it_num,:])
        # Initialize new parameter vec as None
        theta_new = None
        for _ in range(max_up_attempts):
            # Select a random candidate to move from the current live point population
            cand = self.live_pts[np.random.randint(self.live_pts.shape[0]),:]
            # Move the sample according to slice sampling
            theta_try = self.slice_step(it_num = it_num, theta = cand, win_size = win_size,
                                        max_up_attempts = max_up_attempts)
            # Evaluate log-likelihood
            new_logp = self.likelihood_fun(model_par = theta_try)
            # num_evals
            self.num_evals += 1
            # If evaluation is sucessful, we can stop moving the sample.
            if new_logp > self.logp_dead[it_num]:
                self.num_evals_sucessful += 1
                theta_new = theta_try
                break
        # If failed after a number of iterations, just resample the prior as a desperate attempot.
        if theta_new is None:
            theta_new, prior_cube = self.sample_uniform_prior(num_samples = 1)
        # The newly proposed sample substitutes the worst parameter vector (and likelihood)
        self.live_pts[self.worst_logp_index, :] = theta_new
        self.logp_live[self.worst_logp_index] = self.likelihood_fun(model_par = theta_new)
        
    def get_transformLayer(self):
        """ Get transform layer (Ultranest)
        """
        if self.num_model_par > 1:
            self.transformLayer = AffineLayer()
        else:
            self.transformLayer = ScalingLayer()
        self.transformLayer.optimize(self.prior_cube, self.prior_cube)
        
    def get_region(self, ):
        """ Get region from MLfriends (Ultranest)
        """
        self.region = MLFriends(self.prior_cube, self.transformLayer)
        self.volfactor = ultranest.utils.vol_prefactor(n = self.num_model_par)
        # return region
    
    def ml_mover(self, it_num = 0, max_up_attempts = 200):
        """ sampling new point according to ultranest MLfriends algorithm
        """
        if it_num == 0:
            nextregion = self.region
        else:
            nextTransformLayer = self.transformLayer.create_new(self.prior_cube, 
                                                                self.region.maxradiussq)
            nextregion = MLFriends(self.prior_cube, nextTransformLayer)
        r, f = ultranest.integrator._update_region_bootstrap(nextregion, nbootstraps = 30, 
                                                             minvol=0., comm=None, mpi_size=1)
        nextregion.maxradiussq = r
        nextregion.enlarge = f
        # force shrinkage of volume
        # this is to avoid re-connection of dying out nodes
        if nextregion.estimate_volume() < self.region.estimate_volume():
            self.region = nextregion
            self.transformLayer = self.region.transformLayer
        self.region.create_ellipsoid(minvol = np.exp(-it_num / self.n_live) * self.volfactor)
        u = self.region.sample(nsamples = 100)
        return u
    
    def ultranest_slice_sampler(self, it_num = 0, max_up_attempts = 20):
        """ ultranest slice sampler
        """
        # slice_un = PopulationSimpleSliceSampler(popsize = 1,
        #                                         nsteps = max_up_attempts, 
        #                                         generate_direction = ultranest.popstepsampler.generate_random_direction)
        slice_un = PopulationRandomWalkSampler(popsize = 1, 
                                               nsteps = max_up_attempts, 
                                               generate_direction = ultranest.popstepsampler.generate_random_direction, 
                                               scale = 1)
        self.get_transformLayer()
        self.get_region()
        u, p, L, nc = slice_un.__next__(region = self.region, 
                                        Lmin = self.logp_live[self.worst_logp_index], 
                                        us = self.prior_cube, 
                                        Ls = self.logp_live, 
                                        transform = self.prior_un2, 
                                        loglike = self.log_t2)
        self.prior_cube[self.worst_logp_index, :] = u
        
    
    def log_t2(self, model_par):
        """ Computes log-like Student-t likelihood
        
        Parameters
        ----------
        model_par : numpy1dArray
            parameters for a giving tried solution vector
            
        Returns
        ----------
        logp : float
            Value of log - like Student-t distribution
        
        """
        logp = np.zeros((1, 1))
        logp[0,0] = self.log_t(model_par)
        return logp
    
    def nested_sampling(self, n_live = 250, max_iter = 1000, 
                        tol_evidence_increase = 1e-3, seed = 42,
                        max_up_attempts = 50, dlogz=0.001):
        """ Nested sampling loop.
        
        Performs nested sampling for the problem. The steps are
            1. Sample the prior for n_live pts. This will give you your initial
            population, a set of live points and a set of log-likelihood values.
            2. Initialize your variables. This includes initializing a set of
            dead points and their associated log-likelihoods. At the iterations,
            these sets will be filled in ascending order of log-likelihood.
            3. Run iterative sampling procedure
                3.1. Get the worst log-likelihood value in the current live point
                population and the associated parameters. Then these values becomes
                the values of the logp_dead (log-likelihood of the dead points) and
                the dead_points parameter at the i-th iteration.
                3.2. Update evidence
                3.3. Propose an updade for the worst parameter set of parameters
                found. This is a constrained move in the parameter space, where 
                the updated parameters must have a higher log-likelihood than
                the one found in step 3.1. The proposed parameter set 
                will substitute the live point value with the worst 
                log-likelihood (found in step 1). The log-likelihood (live) will
                be updated accordingly. This way, two things happen: 
                    (a) The live_pts will shrink around the most likely spot in
                    the parameter space (that can explain the data);
                    (b) The dead points (and dead log-likelihood) will consist of
                    a set of increasing likelihood.
                    The update options for step 3.3 are:
                        (a) - constrained_resample: as in Beaton and Xiang's paper
                        (b) - constrained_resample_rw: random walk approach
                        (c) - slice sampling: current the best one.
                3.4. Test if iteration can be stopped. Essentially when the evidence
                stops increasing, it is time to stop exploring the parameter space.
            4. Sort the live points population and associated log-likelihood.
            5. Concatenate the log-likelihood of dead and sorted live-points (and the samples).
            6. Normalized weights calculation.
            7. Update evidence (final) value with the max(live log-likelihood) info.
            
        Parameters
        ----------
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
        dlogz : float
            Target evidence uncertainty
        """
        self.n_live = n_live
        # number of evaluations and efficiency metric
        self.num_evals = 0
        self.num_evals_sucessful = 0
        # Initialize seed
        np.random.seed(seed)
        # Sample the prior and compute the logp of initial population
        self.live_pts, self.prior_cube,_ , self.logp_live =\
            self.brute_force_sampling(num_samples = self.n_live)
        # The initial population is the set of samples from brute_force_sampling
        self.init_pop = np.copy(self.live_pts)
        # Initialize variables (dead_pts and log-likelihood of the dead set.)
        self.dead_pts = np.zeros((max_iter, self.num_model_par))
        self.logp_dead = np.zeros(max_iter)
        # Initialize logZ (evidence)
        self.logZ = -1e300 # very low log number 10^-300
        # Initialize prior mass (at the start the whole prior mass)
        self.ksi_prev = 1
        # Initialize information (H)
        self.info = 0.0 # information init
        # Initialize spread of live_pts - keep track of shrinking
        self.spread = np.zeros((max_iter, self.num_model_par))
        # Init bar and main loop
        bar = tqdm(total = max_iter, desc='Nested sampling loop...', ascii=False)
        for i in range(max_iter):
            # 1 - Find index of worst log-likelihood and update variables
            self.get_worst_logp(it_num = i)
            # 2 - Update evidence Z using log-sum-exp for stability
            fraction_remain = self.update_evid_and_info(it_num = i)
            # 3 - Update sample
            self.sample_update_fun(i, max_up_attempts)                   
            # 4 - Test if iteration can be stopped
            if fraction_remain < dlogz:
                bar.update(max_iter-i) # update bar and break loop
                break
            bar.update(1)
        bar.close()
        # delete non-used indexes
        self.delete_zeros(i, max_iter)
        # Concatenate the log-likelihood of dead and sorted live-points (and the samples)
        self.concatenate_live_and_dead()
        # Normalized weights calculation.
        self.weights = np.exp(self.logp_concat - np.max(self.logp_concat))
        self.weights /= np.sum(self.weights)
        # Update evidence and info (final - live pts).
        self.final_update_evid_and_info(it_num = i)
        # Estimate uncertainty in evidence      
        self.logZ_err = np.sqrt(self.info/self.n_live)
        # Efficiency
        self.efficiency = 100*self.num_evals_sucessful/self.num_evals

    def update_evid_and_info(self, it_num):
        """ Updates the log-evidence and information
        """
        # log of current prior mass
        log_ksi_k = -(it_num + 1)/self.n_live
        # delta of prior mass (lin & log scale) - integration goes from 1 to 0
        delta_ksi = self.ksi_prev - np.exp(log_ksi_k)
        log_delta_ksi = np.log(delta_ksi)
        # Update prior mass (lin scale)
        self.ksi_prev = np.exp(log_ksi_k)
        # Update log-evidence and Information (H)
        logZ_new = np.logaddexp(self.logZ, self.logp_dead[it_num]+log_delta_ksi)            
        self.info = np.exp(self.logp_dead[it_num] +\
                           log_delta_ksi -logZ_new) * self.logp_dead[it_num] +\
            np.exp(self.logZ-logZ_new)*(self.info+self.logZ) - logZ_new
        self.logZ = logZ_new
        # Calculate stopping criteria
        logz_remain = np.max(self.logp_live) - it_num / self.n_live
        fraction_remain = np.logaddexp(self.logZ, logz_remain) - self.logZ
        return fraction_remain
    
    def final_update_evid_and_info(self, it_num):
        """ Final update the log-evidence and information (from live points)
        """
        # Average remaining prior log-volume
        logvol = -(it_num+1) / self.n_live - np.log(self.n_live)
        for jl in range(self.n_live):
            # log(likelihood*dX)
            logwt = logvol + self.logp_live[jl]
            #logZ and info update
            logz_new = np.logaddexp(self.logZ, logwt)
            self.info = (np.exp(logwt - logz_new) * self.logp_live[jl] +\
                 np.exp(self.logZ - logz_new) * (self.info + self.logZ) - logz_new)
            self.logZ = logz_new
        
    def delete_zeros(self, it_num, max_iter):
        """ Delete zeros from variables
        
        Happens because inference converged earlier then max_iter
        """
        ranger = range(it_num+1, max_iter)
        self.logp_dead = np.delete(self.logp_dead, ranger)
        self.dead_pts = np.delete(self.dead_pts, ranger, axis = 0)
        self.spread = np.delete(self.spread, ranger, axis = 0)

    def concatenate_live_and_dead(self,):
        """ Concatenates live and dead points and the live and dead likelihoods
        """
        # Sort the live points population and associated log-likelihood.
        sorted_logp_live, sorted_live_pts = self.sort_livepop()
        # Concatenate the log-likelihood of dead and sorted live-points (and the samples)
        self.logp_concat = np.concatenate((self.logp_dead, sorted_logp_live))
        self.prior_samples = np.concatenate((self.dead_pts, sorted_live_pts))

    def sort_livepop(self, ):
        """ Computes sorted version of logp_live and live_pts.
        """
        sorted_indices = np.argsort(self.logp_live)
        sorted_logp_live = self.logp_live[sorted_indices]
        sorted_live_pts = self.live_pts[sorted_indices, :]
        return sorted_logp_live, sorted_live_pts
    
    def prior_un(self, cube):
        prior_cube = cube.copy()
        # transform location parameter: uniform prior
        prior_samples = np.zeros(prior_cube.shape)
        for jp in range(self.num_model_par):
            prior_samples[jp] = prior_cube[jp] *\
                (self.upper_bounds[jp]-self.lower_bounds[jp]) + self.lower_bounds[jp]
        return prior_samples
    
    def prior_un2(self, cube):
        prior_cube = cube.copy()
        # transform location parameter: uniform prior
        prior_samples = np.zeros(prior_cube.shape)
        for row in range(prior_cube.shape[0]):
            for jp in range(self.num_model_par):
                prior_samples[row, jp] = prior_cube[row, jp] *\
                    (self.upper_bounds[jp]-self.lower_bounds[jp]) + self.lower_bounds[jp]
        return prior_samples
    
    def ultranested_sampling(self, n_live = 250, max_iter = 1000):
        """ Run via ultranest
        """       
        sampler = ultranest.NestedSampler(param_names = self.parameters_names,
                                          loglike = self.log_t, 
                                          transform = self.prior_un, num_live_points = n_live)
        # sampler.log = False
        result = sampler.run(max_iters = max_iter)
        self.logp_concat = sampler.results['weighted_samples']['logl']
        self.logZ = sampler.results['logz']
        self.prior_samples = sampler.results['weighted_samples']['points']
        self.weights = sampler.results['weighted_samples']['weights']
        return sampler
    
    def ultranested_sampling_react(self, n_live = 250, max_iter = 1000):
        """ Run via ultranest
        """       
        sampler = ultranest.ReactiveNestedSampler(param_names = self.parameters_names,
                                                  loglike = self.log_t,
                                                  transform = self.prior_un)
        # sampler.log = False
        result = sampler.run(max_iters = max_iter, show_status=False, 
                             viz_callback=False, frac_remain=0.01,
                             min_num_live_points = n_live)
        self.logp_concat = sampler.results['weighted_samples']['logl']
        self.logZ = sampler.results['logz']
        self.prior_samples = sampler.results['weighted_samples']['points']
        self.weights = sampler.results['weighted_samples']['weights']
        return sampler
        
    def weighted_mean(self, ):
        """ Weighted mean
        """
        self.mean_values = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            self.mean_values[jdim] = np.sum(self.weights * self.prior_samples[:, jdim])
    
    def weighted_median(self, ):
        """ Weighted median
        """
        self.median_values = self.weighted_quantile(quantile = 0.5)
    
    def weighted_var(self, ):
        """ Weighted variance and Standard deviation
        """
        self.var = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            deviation = self.prior_samples[:, jdim] - self.mean_values[jdim]
            self.var[jdim] = np.sum(self.weights * deviation**2)
        self.std = np.sqrt(self.var)
    
    def ess_and_stderror(self, ):
        self.ess =  1.0 / np.sum(self.weights**2)
        self.standard_error = np.sqrt(self.var / self.ess)
        
    def confidence_interval(self, ci_percent = 95):
        """ Confidence interval with a given percentage.
        
        Uses weighted_quantile to compute the confidence interval.
        
        Parameters
        ----------
        ci_percent : float
            Percentage of the confidence interval.
        """
        self.ci_percent = ci_percent
        lower_quantile = 0.5*(100-self.ci_percent)/100
        upper_quantile = (self.ci_percent + 0.5*(100-self.ci_percent))/100
        lower_bonds_ci = self.weighted_quantile(quantile = lower_quantile)
        upper_bonds_ci = self.weighted_quantile(quantile = upper_quantile)
        self.ci = np.vstack((lower_bonds_ci, upper_bonds_ci))
        
    def weighted_quantile(self, quantile = 0.5):
        """ Weighted quantile
        """
        posterior_quantile = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            # Samples
            samples = self.prior_samples[:, jdim]
            # Sorted samples and weights
            idx = np.argsort(samples)
            x_sorted = samples[idx]
            w_sorted = self.weights[idx]
            # Cumulative distribution
            cdf = np.cumsum(w_sorted)
            # Interpolate cdf
            x_sorted_interp = np.linspace(x_sorted[0], x_sorted[-1], 
                                          num = 2*len(samples), endpoint=True)
            cdf_interpolated = np.interp(x_sorted_interp, x_sorted, cdf)
            # Find smallest j with cdf_j>= 𝑞uantile
            id_quantile = np.where(cdf_interpolated >= quantile)[0][0]
            posterior_quantile[jdim] = x_sorted_interp[id_quantile]
        return posterior_quantile
               
    def compute_statistics(self, ci_percent = 95):
        """ Compute statistical inferences
        """
        self.weighted_mean()
        self.weighted_median()
        self.weighted_var()
        self.ess_and_stderror()
        self.confidence_interval(ci_percent = ci_percent)
        
    def reconstruct_mean(self, x_meas):
        """ Reconstruction from mean values
        
        Parameters
        ----------
        x_meas : numpyndArray
            Coordinates where to reconstruct.
        """
        # prediction - from mean
        y_recon = self.model_fun(x_meas = x_meas,
                                 model_par = self.mean_values)
        return y_recon
    
    def reconstruct_ci(self, x_meas):
        """ Reconstruction from confidence interval values
        
        Parameters
        ----------
        x_meas : numpyndArray
            Coordinates where to reconstruct.
        """
        y_recon = np.zeros((2, len(x_meas)))
        for jci in range(2):
            y_recon[jci, :] = self.model_fun(x_meas = x_meas,
                                             model_par = self.ci[jci, :])
        return y_recon
    
    def plot_histograms(self, ):
        """ Plot histograms
        """
        plt.figure(figsize = (7,3))
        for jdim in range(self.prior_samples.shape[1]):
            plt.hist(self.prior_samples[:, jdim], bins = 100, weights = self.weights, 
                      density = True, label = r"$\theta_{}$".format(jdim), alpha = 0.7)
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlabel(r"Value")
        plt.ylabel(r"$p(\Theta)$")
        plt.tight_layout();
        
    def plot_smooth_marginal_posterior(self, figshape = None,
                                       figsize = None):
        """ Plot smooth posteriors using scipy KDE
        """
        if figshape is None:
            raise ValueError("I need a shape for the figure.")
        if figsize is None:
            figsize = (figshape[0]*4, figshape[1]*2)
           
        _, ax = ut_is.give_me_an_ax(figshape = figshape, figsize = figsize)
        
        jdim = 0
        for row in range(ax.shape[0]):
            for col in range(ax.shape[1]):
                if jdim < self.prior_samples.shape[1]:
                    kde = scipy.stats.gaussian_kde(self.prior_samples[:, jdim], 
                                                   weights = self.weights)
                    x_grid = np.linspace(self.prior_samples[:, jdim].min(), 
                                         self.prior_samples[:, jdim].max(), 500)
                    pdf = kde(x_grid)
                    ax[row, col].plot(x_grid, pdf, linewidth = 1.5, 
                                      label = self.parameters_names[jdim], 
                                      alpha = 0.7, color = 'mediumpurple')
                    # ax[row, col].legend()
                    ax[row, col].grid(linestyle = '--')
                    ax[row, col].set_xlim((self.lower_bounds[jdim], self.upper_bounds[jdim]))
                    ax[row, col].set_xlabel(self.parameters_names[jdim])
                    ax[row, col].set_ylabel(r"$p(\theta)$")
                    jdim += 1
        plt.tight_layout();
        return ax
    
    def plot_histogram(self, ax = None, par_id = 0):
        """ Plot histograms
        """
        if ax is None:
            _, ax = ut_is.give_me_an_ax(figshape = (1,1), figsize = (6,3))
            ax = ax[0,0]
        
        counts, _, _ = ax.hist(self.prior_samples[:, par_id], bins = 100, weights = self.weights, 
                  density = True, label = self.parameters_names[par_id], alpha = 0.8)
        # print(counts.shape)
        ax.axvline(x = self.mean_values[par_id], linestyle = '--', 
                   color = 'crimson', alpha = 0.8)
        ax.axvline(x = self.median_values[par_id], linestyle = '--', 
                   color = 'palevioletred', alpha = 0.8)
        ax.axvline(x = self.ci[0, par_id], linestyle = '--', 
                   color = 'lightpink', alpha = 0.8)
        ax.axvline(x = self.ci[1, par_id], linestyle = '--', 
                   color = 'lightpink', alpha = 0.8)
        # ax.legend()
        ax.set_xlim((self.mean_values[par_id]-0.25*self.ci[0, par_id],
                     self.mean_values[par_id]+0.25*self.ci[1, par_id]))
        ax.grid(linestyle = '--', alpha = 0.5)
        ax.set_xlabel(r"Value")
        ax.set_ylabel(r"$p(\Theta)$")
    
    def plot_samples(self, ax = None, par_id = 0):
        """ Plot a prior samples of a given parameter
        """
        if ax is None:
            _, ax = ut_is.give_me_an_ax(figshape = (1,1), figsize = (6,3))
            ax = ax[0,0]
            
        ax.scatter(np.arange(stop = self.prior_samples.shape[0]),
                   self.prior_samples[:, par_id], alpha = 0.4, s = 5)
        ax.set_ylim((self.lower_bounds[par_id]-np.abs(0.2*self.lower_bounds[par_id]), 
                     self.upper_bounds[par_id]+0.2*np.abs(self.upper_bounds[par_id])))
        ax.set_xlim((0, self.prior_samples.shape[0]))
        ax.grid(linestyle = '--')
        ax.set_xlabel("index [-]")
        ax.set_ylabel(self.parameters_names[par_id])
    
    def plot_full_trace(self):
        """ plot full trace of all parameters
        """
        _, ax = ut_is.give_me_an_ax(figshape = (self.num_model_par, 2), 
                                    figsize = (6, self.num_model_par*1.5))
        
        for jp in range(self.num_model_par):
            self.plot_samples(ax = ax[jp, 0], par_id = jp)
            self.plot_histogram(ax = ax[jp, 1], par_id = jp)
        plt.tight_layout()
    
    def plot_loglike_vs_mass(self, ax = None):
        """ Plots the evolution of the log-likelihood vs. mass
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = ut_is.give_me_an_ax()
        
        ax[0,0].plot(np.exp(self.delta_mu), self.logp_dead, '-k', linewidth = 1.5, alpha = 0.85)
        ax[0,0].grid(linestyle = '--')
        # ax.set_xlabel(r"iteration [-]")
        ax[0,0].set_xlabel(r"Prior mass [-]")
        ax[0,0].set_ylabel(r"$\mathcal{L}(\theta)$ [Np]")
        ax[0,0].set_xlim((0, 1))
        plt.tight_layout();
        
    def plot_loglike(self, ax = None):
        """ Plots the evolution of the log-likelihood as a function of iteration number
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = ut_is.give_me_an_ax()        
        # ax[0,0].plot(self.logp_dead, '-k', linewidth = 1.5, alpha = 0.85)
        ax[0,0].plot(self.logp_concat, '-k', linewidth = 1.5, alpha = 0.85)
        ax[0,0].grid(linestyle = '--')
        ax[0,0].set_xlabel(r"Iteration Number [-]")
        # ax[0,0].set_ylabel(r"$\mathrm{log}(\mathcal{L}(\theta))$ [Np]")
        ax[0,0].set_ylabel(r"log($L(\theta$)) [Np]")
        ax[0,0].set_xlim((0, len(self.logp_concat)))
        plt.tight_layout(); 
    
    def plot_spread(self, ax = None):
        """ Plots the spread as a function of iteration number
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = ut_is.give_me_an_ax()
            
        for i in range(self.spread.shape[1]):
            ax[0,0].plot(self.spread[:, i], linewidth = 1, 
                    label = self.parameters_names[i] + \
                        r": ({}, {}). $s_o = {:.2f}$".format(self.lower_bounds[i], 
                                                             self.upper_bounds[i],
                                                             self.upper_bounds[i]-self.lower_bounds[i]), 
                    alpha = 0.85)
        ax[0,0].legend()
        ax[0,0].grid(linestyle = '--')
        ax[0,0].set_xlabel(r"iteration [-]")
        ax[0,0].set_ylabel(r"spread")
        ax[0,0].set_xlim((0, self.spread.shape[0]))
        # ax.set_ylim((np.amin(self.lower_bounds), np.amax(self.upper_bounds)))
        plt.tight_layout();
        
    # def plot_marginal_posterior(self, prior_samples, weights):
    #     """ Plot smooth posteriors
    #     """
    #     plt.figure(figsize = (7,3))
    #     for jdim in range(prior_samples.shape[1]):
    #         samples = prior_samples[:, jdim]
    #         idx = np.argsort(samples)
            
    #         plt.plot(samples[idx], weights[idx], linewidth = 2, label = r"$\theta_{}$".format(jdim), 
    #                  alpha = 0.7)
    #     plt.legend()
    #     plt.grid(linestyle = '--')
    #     plt.xlabel(r"$\theta$")
    #     plt.ylabel(r"$p(\Theta)$")
    #     plt.tight_layout();
        
    
        
        
# returned_tuple =\
#     self.constrained_resample2(
#         current_worst_samplevec = self.dead_pts[i, :],
#         current_spread = current_spread,
#         current_worst_logp = self.logp_dead[i],
#         max_attempts = max_attempts)
# self.logp_live[self.worst_logp_index] = returned_tuple[0] 
# self.live_pts[self.worst_logp_index, :] = returned_tuple[1] 
# update_unsuccessful = returned_tuple[2]
# # 2 - If unsuccessful - select a random parameter as base and try again
# if update_unsuccessful:
#     all_indexes = np.arange(0, n_live)
#     remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
#     random_index_id = np.random.randint(low = 0, 
#                                         high = len(remaining_indexes), 
#                                         size = 1)[0]
#     choosen_index = remaining_indexes[random_index_id]
#     returned_tuple =\
#         self.constrained_resample(
#             current_worst_samplevec = self.live_pts[choosen_index, :],
#             current_spread = current_spread,
#             current_worst_logp = self.logp_dead[i],
#             max_attempts = max_attempts)
#     self.logp_live[self.worst_logp_index] = returned_tuple[0] 
#     self.live_pts[self.worst_logp_index, :] = returned_tuple[1] 
#     update_unsuccessful = returned_tuple[2]
# # 3 - If unsuccessful - resample prior using the spread
# if update_unsuccessful:
#     mean = np.mean(self.live_pts, axis = 0)
#     prior_sample = np.random.uniform(low = mean-current_spread/2, 
#                                       high = mean+current_spread/2,
#                                       size = (1, self.num_model_par))[0]
#     # print(prior_sample)
#     logp_new = self.log_t(model_par = prior_sample)
#     if logp_new > self.logp_dead[i]:
#         self.logp_live[self.worst_logp_index] = logp_new 
#         self.live_pts[self.worst_logp_index, :] = np.copy(prior_sample) 
        #update_unsuccessful = False
    # else:
    #     logp_new = np.copy(current_worst_logp)
    #     suggested_parameter_vec = np.copy(current_worst_samplevec)        
    

# def nested_sampling2(self, n_live=200, max_iter=2000):
    
#     # thetas = self.sample_uniform_prior(num_samples = n_live) #sample_prior(n_live)
#     thetas, _, logLs = self.brute_force_sampling(num_samples = n_live) #logLs = np.array([loglike(th) for th in thetas])
#     dead_points, dead_logLs, dead_logw = [], [], []
#     logX_prev = 0.0 # Initialize logX
#     Z = 0 # Initialize evidence
#     bar = tqdm(total = max_iter,
#                desc='Nested sampling loop...', ascii=False)
#     for i in range(max_iter):
#         worst = np.argmin(logLs) # index of worst likelihood
#         logL_star = logLs[worst] # worst likelihood value
#         theta_star = thetas[worst] # worst theta value
#         logX = -(i+1)/n_live # Current log X
#         logw = np.log(np.exp(logX_prev)-np.exp(logX)) # Log of the weights (simple)
#         dead_points.append(theta_star) # Append worst theta value
#         dead_logLs.append(logL_star) # Append worst likelihood value
#         dead_logw.append(logw) # Append weight value
#         # Make the move of worst point
#         seed = thetas[self.rng.integers(n_live)]
#         new_theta, new_logL = self.constrained_move(seed, logL_star)
#         # new_theta, new_logL = self.move_towards(theta_star, logL_star, 
#         #                                         np.delete(thetas, worst, axis=0),
#         #                                         dist_factor = 0.05)
#         thetas[worst], logLs[worst] = new_theta, new_logL
#         logX_prev = logX
#         # Evidence update (Check it)
#         # Z += np.exp(logX) * np.exp(logw) # Update evidence
#         Z_dead = np.exp(logX) * np.exp(logw)
#         Xm = np.exp(-i/n_live)
#         Z_live = (Xm/n_live) * np.sum(np.exp(logLs))
#         Z = Z + Z_dead + Z_live
#         # Bar update
#         bar.update(1)
#     bar.close()
#     # Posterior weights
#     lw = np.array(dead_logLs) + np.array(dead_logw)
#     # self.weights = np.exp(lw - np.max(lw))
#     # self.weights /= np.sum(self.weights)
#     # self.prior_samples = np.array(dead_points)
#     log_dead_weights = lw
#     log_live_weights = logLs
#     log_weights_concat = np.concatenate((log_dead_weights, log_live_weights))
#     self.weights = np.exp(log_weights_concat - np.max(log_weights_concat))
#     self.weights /= np.sum(self.weights)
#     self.prior_samples = np.concatenate((np.array(dead_points), thetas))
#     return self.prior_samples, self.weights, log_weights_concat


# def constrained_resample2(self, current_worst_samplevec, current_spread,
#                           current_worst_logp = -1e6, max_attempts = 5):
#     """ Sample movement in the parameter space.
    
#     Parameters
#     ----------
#     """
#     # initialize while loop variables
#     attempt_num = 0 # number of attempts made
#     update_unsuccessful = True # success of update is set to False first
          
#     # worst_logp = current_worst_logp-1
#     while attempt_num <= max_attempts and update_unsuccessful:
#         # Get a adjustment suggestion
#         random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
#         self.random_par_id.append(random_par_id)
#         # Get a parameter suggestion
#         suggested_parameter_vec =\
#             self.update_parameter(parameter_vec = current_worst_samplevec,
#                                   random_par_id = random_par_id, 
#                                   suggested_adjustment = suggested_adjustment)
#         # Evaluate logp of suggested parameter vector
#         logp_new = self.log_t(model_par = suggested_parameter_vec)
#         # print(attempt_num)
#         if logp_new > current_worst_logp:
#             # current_worst_logp =  np.copy(logp_new)
#             # current_worst_samplevec = np.copy(suggested_parameter_vec)
#             update_unsuccessful = False
#         else:
#             logp_new = np.copy(current_worst_logp)
#             suggested_parameter_vec = np.copy(current_worst_samplevec)
                    
#         attempt_num += 1
#         return logp_new, suggested_parameter_vec, update_unsuccessful


   # def get_destination(self, theta_start, theta_cluster, dist_factor = 0.2):
   #     # Mean of cluster
   #     cluster_mean = np.mean(theta_cluster, axis = 0)
   #     # Direction vector
   #     dir_vec = cluster_mean - theta_start
   #     # Compute total distance to mean
   #     distance = np.linalg.norm(dir_vec)
   #     # Make the move
   #     theta_dest = theta_start + dist_factor * distance * dir_vec/np.linalg.norm(dir_vec)
   #     return theta_dest
   
   # def move_towards(self, theta_current, logL_current, theta_cluster,
   #                  dist_factor = 0.2, num_trials = 20):
   #     trial_num = 0
   #     logL_prop = np.log(np.finfo(float).eps)
   #     while trial_num <= num_trials or logL_prop < logL_current:
   #         theta_dest = self.get_destination(theta_current, theta_cluster,
   #                                           dist_factor = dist_factor)
   #         logL_prop = self.log_t(theta_dest)
   #         if logL_prop > logL_current:
   #             theta, logL = theta_dest, logL_prop
   #         trial_num += 1
   #     return theta, logL
   
   # def constrained_move(self, theta0, logL_thresh, step=0.05, n_steps=50):
   #     theta = theta0.copy()
   #     logL = self.log_t(theta)
   #     for _ in range(n_steps):
   #         prop = theta + self.rng.normal(scale = step, size = self.num_model_par)
   #         # prop = self.sample_uniform_prior(num_samples = 1)[0,:]
   #         # reflect to stay inside bounds
   #         for j in range(self.num_model_par):
   #             if prop[j] < self.lower_bounds[j]:
   #                 prop[j] = self.lower_bounds[j] #2*self.lower_bounds[j] - prop[j]
   #             if prop[j] > self.upper_bounds[j]:
   #                 prop[j] = self.upper_bounds[j] #2*self.upper_bounds[j] - prop[j]
   #         logL_prop = self.log_t(prop)
   #         if logL_prop > logL_thresh:
   #             theta, logL = prop, logL_prop
   #     return theta, logL
   
   # def constrained_move2(self, theta0, logL_thresh, num_trials = 20):
   #     theta = theta0.copy()
   #     logL = self.log_t(theta)
   #     trial_num = 0
   #     logL_prop = np.log(np.finfo(float).eps)
   #     while trial_num <= num_trials or logL_prop < logL_thresh:
   #         prop = prop = theta + self.rng.normal(scale = 0.05, size = self.num_model_par)#self.sample_uniform_prior(num_samples = 1)[0,:]
   #         for j in range(self.num_model_par):
   #             if prop[j] < self.lower_bounds[j]:
   #                 prop[j] = self.lower_bounds[j] + 0.0001 #2*self.lower_bounds[j] - prop[j]
   #             if prop[j] > self.upper_bounds[j]:
   #                 prop[j] = self.upper_bounds[j] - 0.0001 #2*self.upper_bounds[j] - prop[j]
   #         logL_prop = self.log_t(prop)
   #         if logL_prop > logL_thresh:
   #             theta, logL = prop, logL_prop
   #         trial_num += 1
   #     return theta, logL