# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:16:56 2025

@author: Win11
"""

import numpy as np
import utils_insitu as ut_is

#%%
origin = np.array([0, 0])
num_walks = 1000
seed = 42
lower_bounds = np.array([-2, -5])
upper_bounds = np.array([7, 5])
new_coords, directions = ut_is.radom_walk(origin = origin, 
                                          num_walks = num_walks, 
                                          lower_bounds = lower_bounds,
                                          upper_bounds = upper_bounds,
                                          seed = seed)

ut_is.plot_random_walk_2D(new_coords, directions, plot_arrows = False)

#%%
origin = np.array([0, 0, 0, 0])
num_walks = 10000
seed = 42
lower_bounds = np.array([-2, -5, -1.7, -2.2])
upper_bounds = np.array([2.5, 5.3, 5, 5])
new_coords, directions = ut_is.radom_walk(origin = origin, 
                                          num_walks = num_walks, 
                                          lower_bounds = lower_bounds,
                                          upper_bounds = upper_bounds,
                                          seed = seed)

# print(new_coords)
print(np.amin(new_coords, axis=0))
print(np.amax(new_coords, axis=0))