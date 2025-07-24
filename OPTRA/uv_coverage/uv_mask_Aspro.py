#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:48:09 2025

@author: nsaucourt
"""

# COMPUTE the mask on the uv coverage to see the real coverage
# Data's are from Aspro (from precomputed tracks)
# 
# COMPUTE fft of the mask




########################## IMPORT ###############################

import os
import numpy as np
from astropy.io import fits
from module_uv_coverage import mask_uv_coverage,fft_mask

########################## PARAMETERS ###############################

wlen_ref = 3.5e-6    
plot = True

# File parameters
data_path = os.path.dirname(__file__)
filename = 'Aspro2_HD_97048_VLTI_MATISSE_LM_2.86542-4.90177-94ch_A0-B2-C1-D0_2024-03-09.fits'

########################## FILE OPENING ##############################

with fits.open(os.path.join(data_path,filename)) as aspro:
    wlen = sorted(np.asarray(aspro[3].data.EFF_WAVE),reverse=True)
    # Raw UV coords
    uv_raw = {
        'u': aspro[4].data.UCOORD,
        'v': aspro[4].data.VCOORD
    }

# Reshape into (6 baselines x 9 frames )
uCoord = np.zeros((6,9))
vCoord = np.zeros((6,9))
for i in range(6):   
    for j in range(9):
        uCoord[i,j] = uv_raw['u'][j*6 + i]
        vCoord[i,j] = uv_raw['v'][j*6 + i]



mask = mask_uv_coverage(uCoord,vCoord,wlen, plot = plot)


########################## FFT OF THE MASK ##########################

fft = fft_mask(mask, plot = plot)

