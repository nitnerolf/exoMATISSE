###########################################################################
# Low frequency filling... in python!
# Test script
# Coded by Florentin Millour
###########################################################################

import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import os
from astropy.io import fits
from lff import *

def test_loadallv2():
    file = ['LFF/l_Pup_spec1_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits']
    loadallv2(file)
    print('Test loadallv2 passed')
    
test_loadallv2()