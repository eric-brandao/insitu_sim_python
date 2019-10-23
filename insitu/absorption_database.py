import toml
from insitu.controlsair import load_cfg
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio


def load_matdata_from_mat(config_file):
    '''
    This function is used to load the database from a .mat file
    '''
    config = load_cfg(config_file)
    mat = spio.loadmat(config['material']['mat_database'],
            struct_as_record = True)
            
    alpha_list = mat['material']['alpha'][0][0]
    return alpha_list
    # name  = mat['material']['description'][0][0]

def get_alpha_s(mat_file, alpha_list):
    config = load_cfg(mat_file)
    alpha = []
    s = []
    for m in config['material']:
        id = m['id']
        alpha.append(alpha_list[id-1])
        s.append(m['s'])
    return alpha, s