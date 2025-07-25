#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:52:32 2025

@author: nsaucourt
"""

import os
import sys

# --- Add parent folder temporarily for import op_corrflux ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from op_corrflux import op_air_index, op_compute_nco2
from scipy import interpolate
import math
import astropy.constants as const
sys.path.pop(0)


#########################################################
# Compute the piston by fft 
def op_get_piston_fft(amp,phi,wlen ):
    
    #Linear interpolation in wavenumber sigma
    sigma     = 1.0/wlen
    cf = (amp*np.exp(1j*phi))
    dsigma = np.diff(sigma)
    # Make sigma increasing
    if np.mean(dsigma) < 0:
        sigma     = sigma[::-1]
        cf        = cf[::-1]
    
    step      = np.min(np.abs(dsigma))
    sigma_lin = np.arange(np.min(sigma), np.max(sigma), step)

    #Interpolation of correlated flux 
    f = interpolate.interp1d(sigma, np.real(cf))
    cf_real_interp = f(sigma_lin)
    f = interpolate.interp1d(sigma, np.imag(cf))
    cf_imag_interp = f(sigma_lin)
    cf_interp = cf_real_interp + 1j * cf_imag_interp
    
    log_base_2 = int(math.log2(cf_interp.size)) 
    new_size = int(2**(log_base_2+4))
    cf_interp = np.pad(cf_interp, (new_size//2 - cf_interp.shape[0]//2), mode='constant', constant_values=0)

    fft_cf = np.fft.fftshift(np.fft.fft(cf_interp))
    OPDs   = np.fft.fftshift(np.fft.fftfreq(cf_interp.shape[0], step))

    dsp = np.abs(fft_cf)
    mx = np.argmax(dsp)
    #OPD determination
    OPD0 = OPDs[mx]
    OPDp1 = OPDs[mx+1]
    OPDm1 = OPDs[mx-1]
    
    peak0 = dsp[mx]
    peakp1 = dsp[mx+1]
    peakm1 = dsp[mx-1]
    
    OPD = (OPD0 * peak0 + OPDp1 * peakp1 + OPDm1 * peakm1)/(peak0 + peakp1 + peakm1)
    
    return cf,OPD


#########################################################
# Correct the piston
def op_corr_piston(cf,wlen,piston):
    corr =  np.exp(1j * 2 * np.pi * piston / wlen)
    cf_corr = cf * np.conj(corr)
    return cf_corr

#########################################################
# Compute Mathar's model
def n_air_mathar(cond_atm, nu):
    
    x0 = np.array([
        [ 0.200049e-3,  0.588431e-1, -3.13579,     -0.108142e-7,  0.586812e-12,
          0.266900e-8,  0.608860e-17,  0.517962e-4,  0.778638e-6,  -0.217243e-15],
        [ 0.145221e-9, -0.825182e-7,  0.694124e-3,   0.230102e-11, 0.312198e-16,
          0.168162e-14, 0.461560e-22, -0.112149e-7,  0.446396e-12,  0.104747e-20],
        [ 0.250951e-12, 0.137982e-9, -0.500604e-6,  -0.154652e-14, -0.197792e-19,
          0.353075e-17, 0.184282e-24,  0.776507e-11, 0.784600e-15, -0.523689e-23],
        [-0.745834e-15, 0.352420e-13, -0.116668e-8, -0.323014e-17, -0.461945e-22,
         -0.963455e-20, -0.524471e-27, 0.172569e-13, -0.195151e-17, 0.817386e-26],
        [-0.161432e-17, -0.730651e-15, 0.209644e-11, 0.630616e-20, 0.788398e-25,
         -0.223079e-22, -0.121299e-29, -0.320582e-16, -0.542083e-20, 0.309913e-28],
        [ 0.352780e-20, -0.167911e-18, 0.591037e-14, 0.173880e-22, 0.245580e-27,
          0.453166e-25, 0.246512e-32, -0.899435e-19, 0.103530e-22, -0.363491e-31],
    ])
    
    hum, pres, temp = cond_atm

    temp_ref = 273.15 + 17.5 # K
    hum_ref  = 10 # 10%
    pres_ref = 750*1e2  #pa
    nu_ref   = 1e4/3.4 # cm-1
    n_1 = 0

    for i in range(6):
        # Mathar's Coefficient
        c_ref, c_T, c_TT, c_H, c_HH, c_p, c_pp, c_TH, c_Tp, c_Hp = x0[i]     
        c_i  = (c_ref 
            + c_T * (1/temp - 1/temp_ref) + c_TT * (1/temp - 1/temp_ref)**2 
            + c_H * (hum - hum_ref) + c_HH * (hum - hum_ref)**2 
            + c_p * (pres - pres_ref) + c_pp * (pres - pres_ref)**2 
            + c_TH * (1/temp - 1/temp_ref) * (hum - hum_ref)  
            + c_Tp * (1/temp - 1/temp_ref) * (pres - pres_ref) 
            + c_Hp * (hum - hum_ref) * (pres - pres_ref) )
               
        n_1 += c_i * (nu - nu_ref)**i     
    
    #FIXME : COMPRENDRE POURQUOI RETIRER LA MOYENNE EST NECESSAIRE
    n_1 -=np.mean(n_1)
    return n_1

################################################################
# Compute the upgraded Voronin Air index based on the new_coef method
def n_air_newcoef(params, wlen, T, P, h, N_CO2=421, bands='all',wlen_ref = 3500):
    wl_abs1_1, wl_abs1_2, wl_abs1_6, wl_abs2_1, wl_abs2_2, wl_abs2_6, A_abs1_1, A_abs1_2, A_abs1_6, A_abs2_1, A_abs2_2, A_abs2_6= params
    wl = wlen * 1e6 # m -> µm

    ## Characteristic wavelengths (microns)
    wl_abs1 = 1e-3 * np.array([15131, wl_abs1_1*wlen_ref, wl_abs1_2*wlen_ref, 2011.3, 47862, 6719.0, wl_abs1_6*wlen_ref, 1835.6,
               1417.6, 1145.3, 947.73, 85, 127, 87, 128])
    wl_abs2 = 1e-3 * np.array([14218, wl_abs2_1*wlen_ref, wl_abs2_2*wlen_ref, 1964.6, 16603, 5729.9, wl_abs2_6*wlen_ref, 1904.8, 
               1364.7, 1123.2, 935.09, 24.546, 29.469, 22.645, 34.924])
    
    ## Absorption coefficients
    A_abs1 = np.array([4.051e-6, A_abs1_1, A_abs1_2, 1.550e-8, 2.945e-5, 3.273e-6, 
              A_abs1_6, 2.544e-7, 1.126e-7, 6.856e-9, 1.985e-9, 1.2029482,
              0.26507582, 0.93132145, 0.25787285])
    A_abs2 = np.array([1.010e-6,A_abs2_1, A_abs2_2, 5.532e-9, 6.583e-8, 3.094e-6,
              A_abs2_6, 2.181e-7, 2.336e-7, 9.479e-9, 2.882e-9, 5.796725,
              7.734925, 7.217322, 4.742131])
    
    ## Conversion
    # T += 273.15 # °C -> K
    # P *= 1e2    # hPa -> Pa
    h/=100
    ## Water vapor
    T_cr = 647.096 # critical-point temperature of water (K)
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
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[14:] = N_H2O
    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wlen))**2

    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(15)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 0
    for i_band in bands:

        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**2 / (wl**2 - wl_abs1[i_band]**2)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**2 / (wl**2 - wl_abs2[i_band]**2)
        n_air += dn_air1 + dn_air2
    
    #FIXME : COMPRENDRE POURQUOI RETIRER LA MOYENNE EST NECESSAIRE
    n_air -=np.mean(n_air)
    return n_air

################################################################
# Compute the upgraded Voronin Air index based on the newcoef_H2O method
def n_air_newcoef_H2O(params, wlen, T, P, h, N_CO2=421, bands='all',wlen_ref = 3500):
    wl_abs1_1, wl_abs1_2, wl_abs1_6,wl_abs1_15, wl_abs2_1, wl_abs2_2, wl_abs2_6,wl_abs2_15, A_abs1_1, A_abs1_2, A_abs1_6, A_abs1_15, A_abs2_1, A_abs2_2, A_abs2_6,A_abs2_15= params
    wl = wlen * 1e6 # m -> µm

    ## Characteristic wavelengths (microns)
    wl_abs1 = 1e-3 * np.array([15131, wl_abs1_1*wlen_ref, wl_abs1_2*wlen_ref, 2011.3, 47862, 6719.0, wl_abs1_6*wlen_ref, 1835.6,
               1417.6, 1145.3, 947.73, 85, 127, 87, 128,wl_abs1_15*wlen_ref])
    wl_abs2 = 1e-3 * np.array([14218, wl_abs2_1*wlen_ref, wl_abs2_2*wlen_ref, 1964.6, 16603, 5729.9, wl_abs2_6*wlen_ref, 1904.8, 
               1364.7, 1123.2, 935.09, 24.546, 29.469, 22.645, 34.924,wl_abs2_15*wlen_ref])
    
    ## Absorption coefficients
    A_abs1 = np.array([4.051e-6, A_abs1_1, A_abs1_2, 1.550e-8, 2.945e-5, 3.273e-6, 
              A_abs1_6, 2.544e-7, 1.126e-7, 6.856e-9, 1.985e-9, 1.2029482,
              0.26507582, 0.93132145, 0.25787285,A_abs1_15])
    A_abs2 = np.array([1.010e-6,A_abs2_1, A_abs2_2, 5.532e-9, 6.583e-8, 3.094e-6,
              A_abs2_6, 2.181e-7, 2.336e-7, 9.479e-9, 2.882e-9, 5.796725,
              7.734925, 7.217322, 4.742131, A_abs2_15])
    
    ## Conversion
    # T += 273.15 # °C -> K
    # P *= 1e2    # hPa -> Pa
    h/=100
    ## Water vapor
    T_cr = 647.096 # critical-point temperature of water (K)
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
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[14:] = N_H2O
    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wlen))**2

    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(16)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 0
    for i_band in bands:

        deg = 2
        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**deg / (wl**deg - wl_abs1[i_band]**deg)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**deg / (wl**deg - wl_abs2[i_band]**deg)
        n_air += dn_air1 + dn_air2
        
    #FIXME : COMPRENDRE POURQUOI RETIRER LA MOYENNE EST NECESSAIRE
    n_air -=np.mean(n_air)
    return n_air


################################################################
# Compute the upgraded voronin equations for air index using ab method
def n_air_ab_method(params, wlen, T, P, h, N_CO2=421, bands='all'):
    par_a1,par_a2,par_a6,par_b1,par_b2,par_b6 = params
    wl = wlen * 1e6 # m -> µm

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
    # T += 273.15 # °C -> K
    # P *= 1e2    # hPa -> Pa
    h/=100
    ## Water vapor
    T_cr = 647.096 # critical-point temperature of water (K)
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
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[14] = N_H2O
    
    N_gas[1] = np.float64(par_a1*N_CO2 + par_b1*N_H2O)
    N_gas[2] = np.float64(par_a2*N_CO2 + par_b2*N_H2O)
    N_gas[6] = np.float64(par_a6*N_CO2 + par_b6*N_H2O)

    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wlen))**2

    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(15)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 0
    for i_band in bands:
            
        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**2 / (wl**2 - wl_abs1[i_band]**2)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**2 / (wl**2 - wl_abs2[i_band]**2)
        n_air += dn_air1 + dn_air2
        
    #FIXME : COMPRENDRE POURQUOI RETIRER LA MOYENNE EST NECESSAIRE
    n_air -= np.mean(n_air)
    return n_air


################################################################
# Compute the global reduced chi2 of a residual
def compute_chi2(resid,weight,params,nbases,nfiles):
    return np.sum((resid)**2*weight)/(len(resid)*nbases*nfiles-len(params))


################################################################
# Compute the local reduced chi2 of a residual
def compute_chi2_local(resid,weight,params):
    return np.sum((resid)**2*weight)/(len(resid) - len(params))