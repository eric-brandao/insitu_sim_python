import os
import sys
from pprint import pprint

os.chdir(os.path.dirname(os.path.realpath(__file__)))
#print('Working dir: {}'.format(os.getcwd()))
#print('sys.path:')
#pprint(sys.path)


from insitu.locreac_inf_sphe_pu import locre_infsample_sph_pu


locre_infsample_sph_pu()

