#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:25:37 2025

@author: nsaucourt
"""

########################### IMPORT #########################

import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    

from op_rawdata import _op_get_location , op_compute_uv,op_uv_coverage,op_MATISSE_L
sys.path.pop(0)
from astropy.time import Time,TimeDelta
import numpy as np
from module_uv_coverage import *





########################### VARIABLES #########################
DATE = generate_date('2024-05-25T05:05:05.555', 3)
RA = 271.220315
DEC = -24.44458
station_list = ["U1","U2","U3","U4",'U5_LACOUR', 'U6_SAUCOURT','U7_SAUCOURT']

wlen = np.arange(2.9,4.8,0.1) * 1e-6
wlen_ref = 3.5e-6


# Data of the possible different kilometric stations 
# See DOI: 10.1117/12.3020269
Kilometric_stations = {'U_VISTA':       { 'lon':-70.3975  ,'lat':-24.615833,'E':737.56 ,'N':1347.16 ,'ALT':2520},
                          'U_V0':       { 'lon':-70.399803,'lat':-24.621146,'E':504.77 ,'N':756.38  ,'ALT':2500},
                          'U_SPECULOOS':{ 'lon':-70.391083,'lat':-24.615861,'E':1386.23,'N':1344.05 ,'ALT':2450},
                          'U_V1':       { 'lon':-70.381639,'lat':-24.622111,'E':2340.82,'N':649.08  ,'ALT':2430},
                          'U_V2':       { 'lon':-70.394806,'lat':-24.628056,'E':1009.85,'N':-11.98  ,'ALT':2490},
                          'U_V3':       { 'lon':-70.375389,'lat':-24.631250,'E':2972.48,'N':-367.13 ,'ALT':2400},
                          'U_V4':       { 'lon':-70.385472,'lat':-24.632389,'E':1953.29,'N':-493.78 ,'ALT':2430},
                          'U_V5':       { 'lon':-70.388694,'lat':-24.632944,'E':1627.61,'N':-555.5  ,'ALT':2450},
                          'U_V6':       { 'lon':-70.376250,'lat':-24.643306,'E':2885.31,'N':-1707.7 ,'ALT':2420},
                          'U_V7':       { 'lon':-70.392333,'lat':-24.633778,'E':1259.79,'N':-648.23 ,'ALT':2440},
                          'U_V8':       { 'lon':-70.384556,'lat':-24.645306,'E':2045.77,'N':-1930.09,'ALT':2450},
                         }


proposed_stations = {'U5_LACOUR':{'E': 144 ,'N':- 131 }, # SEE DOI: 10.1051/0004-6361/202451060 with U3-U5 ~ 200m
                     'U6_SAUCOURT':{'E': 72.59 ,'N':- 51.16 },# UNDER THE LAB
                     'U7_SAUCOURT':{'E': 56.32 ,'N': -3.88 }}# OVER THE LAB


########################### UV Computations #########################
uCoord = []; vCoord = []
ntel = len(station_list)
instrument = create_instru(ntel)
new_coords = Kilometric_stations | proposed_stations

for date in DATE:
    hdr = create_header(RA,DEC,date,station_list,instrument = instrument, new_coords =  new_coords)
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


########################### UV COVERAGE #########################
op_uv_coverage(uCoord,vCoord, cfdata,instrument = instrument)
