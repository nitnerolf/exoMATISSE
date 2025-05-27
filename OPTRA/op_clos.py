#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Functions to extract visibilities from the CF data
# Author: fmillour
# Date: 18/11/2024
# Project: OPTRA
#
################################################################################


from   astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt
from op_instruments import *
from itertools import combinations

def op_extract_clos(cfdata, verbose=True, plot=False):
    print('Extracting Closure phase and bispectrum amplitude')
    
    ntel = cfdata['ntel'] if 'ntel' in cfdata else 6  # Default to 4 if not specified
    triangles = [list(tri) for tri in combinations(range(ntel), 3)]
    print('Triangles:', triangles)
    
    return cfdata

op_extract_clos([])
print("toto")