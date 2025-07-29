#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:40:28 2023

@author: mhoulle
"""

import numpy as np
import astropy.io.fits as fits
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import EarthLocation

###################
### Conversions ###
###################

def mas2deg(angle):
    return angle/1000/3600
def deg2mas(angle):
    return angle*1000*3600
def mas2rad(angle):
    return np.deg2rad(mas2deg(angle))
def rad2mas(angle):
    return deg2mas(np.rad2deg(angle))
def wrap(angle):
    return np.angle(np.exp(1j*angle))


###############################
### Convolution & Smoothing ###
###############################

def get_DIT(hdu):
    DIT = hdu['OI_VIS'].data['INT_TIME']
    return DIT 


def convolve_gaussian(x, axis=-1, fwhm=5):
    """Convolve an array of vectors with a Gaussian kernel along a given axis."""
    
    print(f'Smoothing by a Gaussian kernel of FWHM = {fwhm} pix')
    
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    
    kernel_size = x.shape[axis]

    x_kernel = np.tile(np.arange(kernel_size).reshape(-1, 1), kernel_size)
    x_kernel -= np.arange(kernel_size)
    
    kernel = np.exp(-0.5 * (x_kernel/sigma)**2)
    kernel /= np.sum(kernel, axis=0)
    
    conv = x @ kernel
    
    return conv

def convolve_square(x, axis=-1, width=5):
    """Convolve an array of vectors with a rectangular kernel along a given axis."""
    
    if width % 2 == 0:
        raise ValueError('Kernel width must be an odd number.')
    else:
        radius = (width-1) // 2
        print(f'Smoothing by a rectangular kernel of width = {width} pix')
    
    kernel_size = x.shape[axis]

    x_kernel = np.tile(np.arange(kernel_size).reshape(-1, 1), kernel_size)
    x_kernel -= np.arange(kernel_size)
    
    kernel = np.zeros_like(x_kernel, dtype=float)
    kernel[(x_kernel >= -radius) & (x_kernel <= radius)] = 1
    kernel /= np.sum(kernel, axis=0)
    
    conv = x @ kernel
    
    return conv


######################
### OIFITS tidying ###
######################

def reorder_exposures_by_mjd(hdu):
    """Reorder the OI_VIS, OI_VIS2 and OI_T3 tables according to their respective MJDs."""
    hdu_reorder = hdu.copy()
    n_base_dict = {'OI_VIS': 6, 'OI_VIS2': 6, 'OI_T3': 4}  # Number of baselines or triangles per exposure
    n_exp = hdu['OI_VIS'].data.size // 6  # Number of exposures in the OIFITS
    for table in ['OI_VIS', 'OI_VIS2', 'OI_T3']:
        n_base = n_base_dict[table]
        # Get MJD
        mjd  = hdu[table].data['MJD'][::n_base]
        # Sort
        mjd_order  = np.argsort(mjd)
        # Initialize
        table_reorder = hdu[table].copy()
        # Reorder
        for i_exp in range(n_exp):
            i_exp_new = mjd_order[i_exp]
            table_reorder.data[i_exp*n_base:(i_exp+1)*n_base] = hdu[table].data[i_exp_new*n_base:(i_exp_new+1)*n_base]
        # Save in new HDU
        hdu_reorder[table] = table_reorder
    return hdu_reorder

def reorder_baselines(hdu):
    """Reorder the OI_VIS and OI_VIS2 tables according to baselines in order (12, 13, 14, 23, 24, 34)
       and the OI_T3 table in order (123, 124, 134, 234)."""
    # Baseline & triangle definitions
    base_idx = [set([32, 33]), set([32, 34]), set([32, 35]), set([33, 34]), set([33, 35]), set([34, 35])]
    triangle_idx = [set([32, 33, 34]), set([32, 33, 35]), set([32, 34, 35]), set([33, 34, 35])]
    base_dict = {'OI_VIS': base_idx, 'OI_VIS2': base_idx, 'OI_T3': triangle_idx}
    # Reorder
    hdu_reorder = hdu.copy()
    n_exp = hdu['OI_VIS'].data.size // 6  # Number of exposures in the OIFITS
    for table in ['OI_VIS', 'OI_VIS2', 'OI_T3']:
        if table in hdu:
            n_base = len(base_dict[table])  # Number of baselines or triangles per exposure (6 or 4)
            # Initialize
            table_reorder = hdu[table].copy()
            # Reorder
            for i_exp in range(n_exp):
                for i_base in range(n_base):
                    i_base_new = base_dict[table].index(set(hdu[table].data['STA_INDEX'][i_exp*n_base + i_base]))
                    table_reorder.data[i_exp*n_base + i_base_new] = hdu[table].data[i_exp*n_base + i_base]
            # Save in new HDU
            hdu_reorder[table] = table_reorder
    return hdu_reorder

def get_baseline_name(sta_index):
    """Give the name of the baseline associated to a given station index."""
    base_name = ['U1-U2', 'U1-U3', 'U1-U4', 'U2-U3', 'U2-U4', 'U3-U4']
    base_idx = [set([32, 33]), set([32, 34]), set([32, 35]), set([33, 34]), set([33, 35]), set([34, 35])]
    return base_name[base_idx.index(set(sta_index))]

def get_all_baseline_names():
    """Give the list of baseline names in the same order as returned by reorder_baselines."""
    return ['U1-U2', 'U1-U3', 'U1-U4', 'U2-U3', 'U2-U4', 'U3-U4']

def compute_UV_coords(ra, dec, lat, long, mjd, sta_xyz, base_order):
    """ Compute the UV coordinates at a given time and position.
        Inputs:
        - ra, dec: right ascension and declination of target (deg),
        - lat, long: geographic coordinates of the observatory (deg),
        - mjd: modified Julian date at the time of the observation,
        - sta_xyz: (x, y, z) coordinates of the telescopes relative to the (lat, long) reference point (m),
        - base_order: order of the baselines (list of tuples), ex: ((1,2), (1,3), (2,3)).
        Outputs:
        - U, V coordinates of the observations (m).
    """
    #ra *= 24/360  # deg -> h
    # cos_d = np.cos(np.deg2rad(dec))
    # sin_d = np.sin(np.deg2rad(dec))
    # sin_b = np.sin(np.deg2rad(lat))
    # cos_b = np.cos(np.deg2rad(lat))
    cos_d = np.cos(dec*np.pi/180)
    sin_d = np.sin(dec*np.pi/180)
    sin_b = np.sin(lat*np.pi/180)
    cos_b = np.cos(lat*np.pi/180)

    # Compute the hour angle
    t = Time(mjd, scale='utc', format='mjd', location=EarthLocation(lon=long, lat=lat))
    lmst = t.sidereal_time('apparent')
    #hour_angle = lmst.hms.h + lmst.hms.m/60 + lmst.hms.s/3600 - ra
    hour_angle = lmst.deg - ra
    #print(lmst.hms, lmst.hms.h, lmst.hms.m, lmst.hms.s)
    cos_ha = np.cos(hour_angle * np.pi/180)
    sin_ha = np.sin(hour_angle * np.pi/180)
    #cos_ha = np.cos(hour_angle * np.pi/12)
    #sin_ha = np.sin(hour_angle * np.pi/12)

    # Compute relative coordinates of telescope pairs
    base_order = np.array(base_order) - 1
    dx, dy, dz = np.zeros(len(base_order)), np.zeros(len(base_order)), np.zeros(len(base_order))
    for i_base, base in enumerate(base_order):
        dx[i_base], dy[i_base], dz[i_base] = sta_xyz[base[0]] - sta_xyz[base[1]]

    # Compute UV coordinates of baselines
    U = dx * cos_ha - dy * sin_b * sin_ha + dz * cos_b * sin_ha
    V = dx * sin_d * sin_ha + dy * (sin_b * sin_d * cos_ha + cos_b * cos_d) - dz * (cos_b * sin_d * cos_ha - sin_b * cos_d)
    return U, V

def compute_UV_coords_bis(ra, dec, lat, long, mjd, sta_xyz, base_order):
    """ Compute the UV coordinates at a given time and position.
        Inputs:
        - ra, dec: right ascension and declination of target (deg),
        - lat, long: geographic coordinates of the observatory (deg),
        - mjd: modified Julian date at the time of the observation,
        - sta_xyz: (x, y, z) coordinates of the telescopes relative to the (lat, long) reference point (m),
        - base_order: order of the baselines (list of tuples), ex: ((1,2), (1,3), (2,3)).
        Outputs:
        - U, V coordinates of the observations (m).
    """
    ra *= 24/360  # deg -> h
    cos_d = np.cos(np.deg2rad(dec))
    sin_d = np.sin(np.deg2rad(dec))
    sin_b = np.sin(np.deg2rad(lat))
    cos_b = np.cos(np.deg2rad(lat))

    # Compute the hour angle
    c = (280.46061837, 360.98564736629, 0.000387933, 38710000.0)
    mjd2000 = 51544.5
    t0 = mjd - mjd2000
    t = t0 / 36535
    theta = c[0] + c[1] * t0 + t**2 * (c[2] - t/c[3])
    lst = (theta + long) / 15
    if lst < 0:
        decimal = lst - int(lst)
        lst = 24 + int(lst) % 24 + decimal
    decimal = lst - int(lst)
    lst = int(lst) % 24 + decimal
    hour_angle = lst - ra
    cos_ha = np.cos(hour_angle * np.pi/12)
    sin_ha = np.sin(hour_angle * np.pi/12)

    # Compute relative coordinates of telescope pairs
    base_order = np.array(base_order) - 1
    dx, dy, dz = np.zeros(len(base_order)), np.zeros(len(base_order)), np.zeros(len(base_order))
    for i_base, base in enumerate(base_order):
        dx[i_base], dy[i_base], dz[i_base] = sta_xyz[base[0]] - sta_xyz[base[1]]

    # Compute UV coordinates of baselines
    U = - dx * cos_ha + dy * sin_b * sin_ha - dz * cos_b * sin_ha
    V = - dx * sin_d * sin_ha - dy * (sin_b * sin_d * cos_ha + cos_b * cos_d) + dz * (cos_b * sin_d * cos_ha - sin_b * cos_d)
    return U, V

def compute_altaz(ra, dec, lat, long, mjd):
    """ Compute the alt-az coordinates at a given time and position.
        Inputs:
        - ra, dec: right ascension and declination of target (deg),
        - lat, long: geographic coordinates of the observatory (deg),
        - mjd: modified Julian date at the time of the observation,
        Outputs:
        - altitude and azimuth of target (deg).
    """
    ra *= 24/360  # deg -> h
    cos_d = np.cos(np.deg2rad(dec))
    sin_d = np.sin(np.deg2rad(dec))
    sin_b = np.sin(np.deg2rad(lat))
    cos_b = np.cos(np.deg2rad(lat))

    # Compute the hour angle
    t = Time(mjd, scale='utc', format='mjd', location=(long, lat))
    lmst = t.sidereal_time('mean')
    hour_angle = lmst.hms.h + lmst.hms.m/60 + lmst.hms.s/3600 - ra
    cos_ha = np.cos(hour_angle * np.pi/12)

    alt = np.arcsin(sin_d * sin_b + cos_d * cos_b * cos_ha)
    az  = np.arccos((sin_d - np.sin(alt) * sin_b) / (np.cos(alt) * cos_b))
    return np.rad2deg(alt), np.rad2deg(az)

def compute_hour_angle(ra, lat, long, mjd):
    """ Compute the alt-az coordinates at a given time and position.
        Inputs:
        - ra: right ascension of target (deg),
        - lat, long: geographic coordinates of the observatory (deg),
        - mjd: modified Julian date at the time of the observation,
        Outputs:
        - hour angle (hour).
    """
    ra *= 24/360  # deg -> h
    t = Time(mjd, scale='utc', format='mjd', location=(long, lat))
    lmst = t.sidereal_time('mean')
    hour_angle = lmst.hms.h + lmst.hms.m/60 + lmst.hms.s/3600 - ra
    return hour_angle


##################
### Atmosphere ###
##################
    
def refractive_index(wl, T=15, P=1013.25, h=0.1, N_CO2=423, bands='all'):
    """ Compute the refractive index at a given temperature, pressure and relative humidity
        as a function of wavelength, using Equation (5) of Voronin & Zheltikov (2017).
        
        Reference: Voronin, A. A. and Zheltikov, A. M. The generalized Sellmeier equation for air. 
        Sci. Rep. 7, 46111; doi: 10.1038/srep46111 (2017).
        
        Inputs:
        - wl: array of wavelengths in microns,
        - T: temperature in °C,
        - P: pressure in hPa,
        - h: relative humidity,
        - N_CO2: CO2 concentration in dry air in ppm,
        - bands: list of absorption bands to include according to the numbering in Table 1 of
            Voronin & Zheltikov (2017). Default: 'all' (includes all 15 bands).
        Output:
        - n_air: array of refractive indices at each value of wl.
    """
    ## Characteristic wavelengths (microns)
    wl_abs1 = 1e-3 * np.array([15131, 4290.9, 2684.9, 2011.3, 47862, 6719.0, 2775.6, 1835.6,
               1417.6, 1145.3, 947.73, 85, 127, 87, 128])
    wl_abs2 = 1e-3 * np.array([14218, 4223.1, 2769.1, 1964.6, 16603, 5729.9, 2598.5, 1904.8, 
               1364.7, 1123.2, 935.09, 24.546, 29.469, 22.645, 34.924])
    
    ## Absorption coefficients
    A_abs1 = np.array([4.051e-6, 2.897e-5, 8.573e-7, 1.550e-8, 2.945e-5, 3.273e-6, 
              1.862e-6, 2.544e-7, 1.126e-7, 6.856e-9, 1.985e-9, 1.2029482,
              0.26507582, 0.93132145, 0.25787285])
    A_abs2 = np.array([1.010e-6, 2.728e-5, 6.620e-7, 5.532e-9, 6.583e-8, 3.094e-6,
              2.788e-6, 2.181e-7, 2.336e-7, 9.479e-9, 2.882e-9, 5.796725,
              7.734925, 7.217322, 4.742131])
    
    ## Conversion
    T += 273.15 # °C -> K
    P *= 1e2    # hPa -> Pa
    
    ## Water vapor
    T_cr = 647.096 #  critical-point temperature of water (K)
    P_cr = 22.064e6 # critical-point pressure of water (Pa)
    tau = T_cr / T
    th = 1 - T / T_cr
    a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502
    # Saturated water vapor pressure (Pa)
    P_sat = P_cr * np.exp(tau*(a1*th + a2*th**1.5 + a3*th**3 + a4*th**3.5 + a5*th**4 + a6*th**7.5))
    # Water vapor density (m-3)
    N_H2O = h * P_sat / (const.k_B.value * T)

    ## Dry air pressure (Pa)
    P_dry = P - h * P_sat

    ## Gas concentrations (m-3) (mixing ratio in ppm of dry air * concentration of dry air)
    N_air = P_dry / (const.k_B.value * T)
    N_N2 = (780840 * 1e-6) * N_air
    N_O2 = (209460 * 1e-6) * N_air
    N_Ar = (9340 * 1e-6) * N_air
    N_CO2 = (N_CO2 * 1e-6) * N_air
    
    N_gas = np.zeros_like(wl_abs1)
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[14] = N_H2O
    
    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wl*1e-6))**2
   
    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(15)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 1
    for i_band in bands:
        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**2 / (wl**2 - wl_abs1[i_band]**2)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**2 / (wl**2 - wl_abs2[i_band]**2)
        n_air += dn_air1 + dn_air2
    
    return n_air
   
    
   
    
   
    
   




