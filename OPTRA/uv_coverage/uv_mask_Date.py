#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:56:57 2025

@author: nsaucourt
"""

########################## IMPORT ###############################


import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    


import numpy as np

from op_rawdata import op_compute_uv,op_MATISSE_L



sys.path.pop(0)
from module_uv_coverage import *


########################## PARAMETERS ###############################

wlen = np.arange(2.9,4.8,0.1) * 1e-6
wlen_ref = 3.5e-6

# UV Computation variables
RA = 167.01379572155
DEC = -77.65485854929
DATE = generate_date('2024-03-09T00:30:00.000', 9)
station_list = ['A0','B2','C1','D0']
# plot variable
plot = True

########################## GENERATE UV COMPUTATION ##############################

instrument = op_MATISSE_L
uCoord = []; vCoord = []
for date in DATE:
    
    hdr = create_header(RA,DEC,date,station_list)

    
    cfdata = {'hdr':hdr,
              'OI_WAVELENGTH':{'EFF_WAVE':wlen , 
                               'EFF_REF':wlen_ref},
              'OI_BASELINES':{}}
    
    if date == DATE[0]:
        cfdata = op_compute_uv(cfdata, True , instrument = instrument)
    else:
        cfdata = op_compute_uv(cfdata, False , instrument = instrument)
    
    uCoord.append(cfdata['OI_BASELINES']['UCOORD'])
    vCoord.append(cfdata['OI_BASELINES']['VCOORD'])

# RESHAPE THE uv Coordinate
uCoord = np.array(uCoord)
uCoord = uCoord.swapaxes(0,1).reshape(6,60)
vCoord = np.array(vCoord)
vCoord = vCoord.swapaxes(0,1).reshape(6,60)


########################## UV COVERAGE AND MASK ##############################
mask = mask_uv_coverage(uCoord,vCoord,wlen, plot = plot)

########################## FFT OF THE MASK ##########################
fft = fft_mask(mask, plot = plot)

