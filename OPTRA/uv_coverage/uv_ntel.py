#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:25:37 2025

@author: nsaucourt
"""

####### HOW TO USE THIS CODE  ######


# 1. ADAPT ntel TO THE NUMBER OF TELESCOPES YOU WANT 
#
# 2. ADD ntel STATIONS AND NAMES OF TELESCOPES BY ADDING 
#        'HIERARCH ESO ISS CONF STATION{i}' AND 'HIERARCH ESO ISS CONF T{i}NAME' 
#       
# 3. IF THE STATIONS DOESNT ALREADY EXIST, PLEASE ADD ALSO THE E,N COORDINATES BY ADDING
#        'HIERARCH ESO ISS CONF T{i}X' AND 'HIERARCH ESO ISS CONF T{i}Y'
#         





########################### IMPORT #########################

import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import op_rawdata
from op_rawdata import _op_get_location , op_compute_uv,op_uv_coverage,op_MATISSE_L
from astropy.time import Time,TimeDelta
import numpy as np

sys.path.pop(0)


########################### FUNCTIONS #########################

#Compute the UTC of 10 points from the start_utc until start_utc + interval 
def generate_date(start_utc, interval):
    start = Time(start_utc)
    end = start + TimeDelta(interval * 3600, format='sec')  # 'interval' hours later

    # 10 points equally spaced between start and end
    times = start + np.linspace(0, (end - start).sec, 10) * TimeDelta(1, format='sec')
    
    return [t.isot for t in times]

###############################################################
#Create a fake VLTI instrument with a scrambling associated
def create_instru(ntel):
    if ntel == 4:
        return op_MATISSE_L
    else:
        return  {  'name': f'fake_MATISSE_{ntel}tel',
                    'ntel': ntel,
                    'scrP': np.arange(1,ntel+1),
                    'scrB':[[i,j] for i in range(0,ntel-1) for j in range(i+1,ntel)] }
        
                 

###############################################################
# GPS to E,N coordinate converter
def gps_to_E_N( lat, lon):
    """
    Convert lthe GPS coordinates (lat, lon) in meters (E,N)
    based on the origine of the VLTI (lat0, lon0).
    """
    R = 6371000  # Mean Radius of Earth in meters
    lat0 = -24.62794830
    lon0 = -70.40479659

    lat0_rad = np.radians(lat0)
    lat_rad  = np.radians(lat)
    dlat = lat_rad - lat0_rad
    dlon = np.radians(lon - lon0)

    # Mean to correct the latitude
    latm = (lat_rad + lat0_rad) / 2

    dx = R * dlon * np.cos(latm)  # towards East
    dy = R * dlat                 # towards North

    return round(dx,2), round(dy,2)


########################### VARIABLES #########################
DATE = generate_date('2024-05-25T05:05:05.555', 3)
ntel = 6
RA = 271.220315
DEC = -24.44458

# Data of the possible different kilometric stations 
# See DOI: 10.1117/12.3020269
Kilometric_stations = {'U_VISTA':       { 'lon':-70.3975  ,'lat':-24.615833,'E':737.56 ,'N':1347.16 ,'ALT':2520},
                          'U_V0':       { 'lon':-70.399803,'lat':-24.621146,'E':504.77 ,'N':756.38  ,'ALT':2500},
                          'U_SPECULOOS':{ 'lon':-70.391083,'lat':-24.615861,'E':1386.23,'N':1344.05 ,'ALT':2450},
                          'U_V1':       { 'lon':-70.381639,'lat':-24.622111,'E':2340.82,'N':649.08  ,'ALT':2430},
                          'U_V2':       { 'lon':-70.394806,'lat':-24.628056,'E':1009.85,'N':-11.98  ,'ALT':2490},
                          # 'U_V3':       { 'lon':-70.375389,'lat':-24.631250,'E':2972.48,'N':-367.13 ,'ALT':2400},
                          'U_V4':       { 'lon':-70.385472,'lat':-24.632389,'E':1953.29,'N':-493.78 ,'ALT':2430},
                          'U_V5':       { 'lon':-70.388694,'lat':-24.632944,'E':1627.61,'N':-555.5  ,'ALT':2450},
                          # 'U_V6':       { 'lon':-70.376250,'lat':-24.643306,'E':2885.31,'N':-1707.7 ,'ALT':2420},
                          'U_V7':       { 'lon':-70.392333,'lat':-24.633778,'E':1259.79,'N':-648.23 ,'ALT':2440},
                          # 'U_V8':       { 'lon':-70.384556,'lat':-24.645306,'E':2045.77,'N':-1930.09,'ALT':2450},
                         }

# SEE DOI: 10.1051/0004-6361/202451060 with U3-U5 ~ 200m
proposed_stations = {'U5_LACOUR':{'E': 144 ,'N':- 131 },
                     'U6_SAUCOURT':{'E': 72.59 ,'N':- 51.16 },# AU DESSOUS DU LAB
                     'U7_SAUCOURT':{'E': 56.32 ,'N': -3.88 }}# AU DESSUS DU LAB


########################### UV Computations #########################
uCoord = []; vCoord = []
instrument = create_instru(ntel)
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
           'HIERARCH ESO ISS CONF STATION1':"U1",
           'HIERARCH ESO ISS CONF T1NAME'  :'UT1',
           # 'HIERARCH ESO ISS CONF T1X'     : 9.925, 
           # 'HIERARCH ESO ISS CONF T1Y'     : 20.335,

           'HIERARCH ESO ISS CONF STATION2':"U2",
           'HIERARCH ESO ISS CONF T2NAME'  :'UT2',
           # 'HIERARCH ESO ISS CONF T2X'     : -14.887,   
           # 'HIERARCH ESO ISS CONF T2Y'     : -30.502,

           'HIERARCH ESO ISS CONF STATION3':"U3",
           'HIERARCH ESO ISS CONF T3NAME'  :'UT3',
           # 'HIERARCH ESO ISS CONF T3X'     : -44.915,   
           # 'HIERARCH ESO ISS CONF T3Y'     : -66.183,

           'HIERARCH ESO ISS CONF STATION4':"U4",
           'HIERARCH ESO ISS CONF T4NAME'  :'UT4',
           # 'HIERARCH ESO ISS CONF T4X'     : -103.306,   
           # 'HIERARCH ESO ISS CONF T4Y'     : -43.999,
      
           'HIERARCH ESO ISS CONF STATION5':"U5",
           'HIERARCH ESO ISS CONF T5NAME'  : 'UT5',
           'HIERARCH ESO ISS CONF T5X'     : 144,  
           'HIERARCH ESO ISS CONF T5Y'     : -131,
           
           'HIERARCH ESO ISS CONF STATION6':"U6",
           'HIERARCH ESO ISS CONF T6NAME'  : 'UT6',
           'HIERARCH ESO ISS CONF T6X'     : 56.32,  
           'HIERARCH ESO ISS CONF T6Y'     : -3.88,
           
           
        }
    
    ##### IF YOU WANT TO USE ALL KILOMETRIC STATIONS
    # for i,(k,v) in enumerate(Kilometric_stations.items()):
        # hdr[f'HIERARCH ESO ISS CONF STATION{ntel - i}'] = k
        # hdr[f'HIERARCH ESO ISS CONF T{ntel-i}NAME'] = f'UT{ntel- i}'
        # hdr[f'HIERARCH ESO ISS CONF T{ntel - i}X'] = v['E']
        # hdr[f'HIERARCH ESO ISS CONF T{ntel - i}Y'] = v['N']
            
            
    
    
    wlen = np.arange(2.9,4.8,0.1) * 1e-6
    wlen_ref = 3.5e-6
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
