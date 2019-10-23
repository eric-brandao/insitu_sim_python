import os
import sys
from pprint import pprint

import pytest

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print('Working dir: {}'.format(os.getcwd()))
print('sys.path:')
pprint(sys.path)

pytest.main()
