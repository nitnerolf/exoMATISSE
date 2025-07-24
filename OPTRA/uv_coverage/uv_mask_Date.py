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
from mask_uv_coverage import mask_uv_coverage, fft_mask
from op_rawdata import op_compute_uv,op_MATISSE_L
from astropy.time import Time,TimeDelta


sys.path.pop(0)


########################### FUNCTIONS #########################

#Compute the UTC of 10 points from the start_utc until start_utc + interval 
def generate_date(start_utc, interval):
    start = Time(start_utc)
    end = start + TimeDelta(interval * 3600, format='sec')  # 'interval' hours later

    # 10 points equally spaced between start and end
    times = start + np.linspace(0, (end - start).sec, 10) * TimeDelta(1, format='sec')
    
    return [t.isot for t in times]


########################## PARAMETERS ###############################

wlen = np.arange(2.9,4.8,0.1) * 1e-6
wlen_ref = 3.5e-6

# UV Computation variables
RA = 167.01379572155
DEC = -77.65485854929
DATE = generate_date('2024-03-09T00:30:00.000', 9)

# plot variable
plot = True

########################## GENERATE UV COMPUTATION ##############################

instrument = op_MATISSE_L
uCoord = []; vCoord = []
for date in DATE:
    
    hdr = {'ORIGIN'                        :'Paranal',
           'RA'                            : RA ,
           'DEC'                           : DEC ,
           'DATE-OBS'                      : date,
           'HIERARCH ESO DET NDIT'         : 6,
           'HIERARCH ESO DET SEQ1 DIT'     : 10,
           'HIERARCH ESO ISS CONF NTEL'    : instrument['ntel'],
           'HIERARCH ESO ISS GEOLON'       : -70.40479659 ,
           'HIERARCH ESO ISS GEOLAT'       : -24.62794830 ,
           'HIERARCH ESO ISS GEOELEV'      :  2635,
           'LST'                           : Time(date).sidereal_time('apparent',longitude=-70.40479659).hour*3600,
           'MJD-OBS'                       : Time(date).mjd,
           'HIERARCH ESO ISS CONF STATION1':"A0",
           'HIERARCH ESO ISS CONF T1NAME'  :'AT1',

           'HIERARCH ESO ISS CONF STATION2':"B2",
           'HIERARCH ESO ISS CONF T2NAME'  :'AT2',

           'HIERARCH ESO ISS CONF STATION3':"C1",
           'HIERARCH ESO ISS CONF T3NAME'  :'AT3',

           'HIERARCH ESO ISS CONF STATION4':"D0",
           'HIERARCH ESO ISS CONF T4NAME'  :'AT4',}

    
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

