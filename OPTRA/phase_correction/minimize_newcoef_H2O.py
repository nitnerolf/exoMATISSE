#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:07:55 2025

@author: nsaucourt
"""

############################### IMPORT ###############################

import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from scipy.optimize import minimize
import astropy.constants as const
from op_corrflux import op_air_index, op_compute_nco2
from scipy import interpolate
import math
import json
sys.path.pop(0)

from module_phase_correction import *

############################### GLOBAL VARIABLES ###############################
plot = True

# Put CAL if you want to work on the Calibrators, bet_pic if you want to work on Beta Pic
obj  = 'CAL'
# obj  = "bet_pic"



############################### FUNCTIONS ###############################

################################################################
# Compute the upgraded Voronin Air index based on the newcoef_H2O method
def n_air_newcoef_H2O(params, wlen, T, P, h, N_CO2=421, bands='all'):
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
    n_air -=np.mean(n_air)
    return n_air



################################################################
# Compute the global reduced chi2 of the corrected air index
def chi2_global_newcoef_H2O(params,wlen,TEMP, PRES, HUM,PATH,PHASE,WEIGHT,AMP, N_CO2=421):
    chi2 = 0

    for ibase in range(nbases):
        for ifile in range(nfiles):
            # Atmospheric Conditions
            hum = HUM[ibase,ifile]
            pres = PRES[ibase,ifile]
            temp = TEMP[ibase,ifile] 
            
            # variables 
            path = PATH[ibase,ifile] 
            amp = AMP[ibase,ifile]
            weight = WEIGHT[ibase,ifile]
            phi = PHASE[ibase,ifile]
            exp_obs = np.exp(1j * phi)
            
            # Compute model
            n_model = n_air_newcoef_H2O(params,wlen,temp,pres,hum,N_CO2)
            phi_model = (2 * np.pi * path) * (n_model) / wlen
            exp_model = np.exp(1j * phi_model)
            
            # Compute residuals
            resid = np.angle(exp_obs * np.conj(exp_model))

            #Piston Correction
            cf,OPD = op_get_piston_fft(amp,resid,wlen)
            cf_corr = op_corr_piston(cf,wlen,OPD)
            totvis = np.sum(cf_corr)
            cvis = cf_corr * np.exp(-1j * np.angle(totvis[...,None]))
            resid = np.angle(cvis)

            #CHI2 Computation
            chi2 += compute_chi2(resid,weight,params,nbases,nfiles)
    return chi2




###############################  MAIN PROGRAM ###############################

if __name__ == '__main__': 

    ############ FILE OPENER ############
    # Files' name 
    jsonfile = os.path.expanduser(f'~/Documents/exoMATISSE/OPTRA/phase_correction/parameters_min_{obj}.json')
    filename = os.path.expanduser(f'~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_{obj}.fits')
    outdir   = os.path.expanduser('~/Documents/Result_phase_correction/')
    
    # Fits' extracion
    with fits.open(filename) as fh:
        hdr = fh[0].header
        nfiles = fh['OI_VIS'].header['NFILES']
        nbases = 6
        wlen = fh['OI_VIS'].data['WLEN'][0][0]
        nwlen = len(wlen)
        phase = fh['OI_VIS'].data['PHASE'].reshape(nbases,nfiles,nwlen)
        PATH = fh['OI_VIS'].data['PATH']
        HUM  = fh['OI_VIS'].data['HUM'] * 100
        PRES = fh['OI_VIS'].data['PRES'] * 1e2
        TEMP = fh['OI_VIS'].data['TEMP'] + 273.15
        ERR = fh['OI_VIS'].data['ERR'].reshape(nbases,nfiles,nwlen)
        AMP = fh['OI_VIS'].data['AMP'].reshape(nbases,nfiles,nwlen)
        N_CO2 = op_compute_nco2({'hdr':hdr})
    
    #json's extraction
    try: 
        with open(jsonfile, 'r') as f:
            all_params = json.load(f)
    except:
        print("The JSON file was not in the good format: reinitialization of the JSON file")
        all_params = {"new_coef_H2O":{"chi2_min":1000000,"params_min":[]}}
        
    try:
        
        new_coef_H2O = all_params['new_coef_H2O'] 
    except:
        print("The JSON file did not have the new_coef_H2O method: initialization of the parameters ")
        all_params['new_coef_H2O']={"chi2_min":1000000,"params_min":[]}
        
    new_coef_H2O = all_params['new_coef_H2O'] 
    
    
    
    ############################### VARIABLES ###############################
    
    wlen_cm =1e2 * wlen
    nu = 1/wlen_cm
    wlen_ref = 3500 # in nm
    
    #mask
    nu_mask = (nu < 3548) & (nu > 2389)
    
    # masked variables 
    nu_masked    = nu[nu_mask]
    wlen_masked  = wlen[nu_mask]
    phase_masked = phase[...,nu_mask]
    ERR_masked   = ERR[...,nu_mask]
    AMP_masked   = AMP[...,nu_mask]
    
    # Weight for chi2
    WEIGHT = (1/ERR_masked)**2 
    
    # CHI2 initialization
    chi2_vor = 0
    chi2_new_coef_H2O = 0
    
    ############################### MINIMIZATION ###############################
    
    # initial conditions
    vor0 = [4290.9/wlen_ref, 2684.9/wlen_ref,2775.6/wlen_ref,4290.9/wlen_ref,4223.1/wlen_ref, 2769.1/wlen_ref,2598.5/wlen_ref,4223.1/wlen_ref,2.897e-5, 8.573e-7,1.862e-6,0,2.728e-5, 6.620e-7,2.788e-6,0]
    # vor0 =  [ 1.22597108e+00,  7.67114233e-01,  7.93028500e-01,  1.22597142e+00, 1.20659845e+00,  7.91171366e-01,  7.42428519e-01,  1.20659999e+00, 4.43338085e-05, -1.72039559e-06,  3.76655977e-07, -7.12464612e-07, 2.34019660e-05, -5.73826882e-06,  2.93219876e-06,  1.90389813e-07]
    # vor0 = [ 1.22597111e+00,  7.67114254e-01,  7.93028470e-01,  1.22597145e+00, 1.20659848e+00,  7.91171419e-01,  7.42428494e-01,  1.20660002e+00, 4.29608641e-05,  6.50105229e-07,  2.03709837e-07, -7.38255587e-07, 2.18340391e-05, -2.28027564e-06,  2.89129652e-06,  2.84686902e-07]
    eps_vec = [1e-2]*8 + [1e-10]*8
    bounds = [(0.5,1.5)]*8 + [(-10,10)]*8 
    
    for i in range(5):
        for j in range(5):
            eps_vec = [1e-0*10**(-i)]*8 + [1e-8*10**(-j)]*8
            result = minimize(chi2_global_newcoef_H2O, vor0, args = (wlen_masked,TEMP, PRES, HUM,PATH,phase_masked,WEIGHT,AMP_masked,N_CO2),method='l-bfgs-b',bounds = bounds,jac = False,options  = {'eps'     : eps_vec,    'maxls'   : 1000,       'ftol'    : 1e-30,    'gtol'   : 1e-15,    'maxiter' : 1000, 'iprint':1 })
            print(result.x)
            print(result.success)
            print(result.nit)
            print(result.message)
            print(result.fun)
            params_min_new_coef_H2O = result.x
            chi2_min_new_coef_H2O   = result.fun
            vor0 = params_min_new_coef_H2O

    # SAVE IN JSON
    if chi2_min_new_coef_H2O < new_coef_H2O["chi2_min"]:
        new_coef_H2O["chi2_min"] = chi2_min_new_coef_H2O
        new_coef_H2O["params_min"] = params_min_new_coef_H2O.tolist()
        all_params['new_coef_H2O'] = new_coef_H2O
        print("CHI2 is better -> dumped in parameters_min.json")
        with open(jsonfile, 'w') as f:
            json.dump(all_params,f)
    
    
    ############################### COMPUTE MODELS ###############################
    if plot:
        fig1,ax1  = plt.subplots(nbases,nfiles,sharex=True,sharey=True,figsize = (nfiles * 6.5,nbases * 3 ))
        
        for ibase in range(nbases):
            for ifile in range(nfiles):
                # Atmospheric Conditions
                hum = HUM[ibase,ifile]
                pres = PRES[ibase,ifile]
                temp = TEMP[ibase,ifile] 
                
                # variables 
                path = PATH[ibase,ifile] 
                amp = AMP_masked[ibase,ifile] 
                weight = WEIGHT[ibase,ifile]
                phi = phase_masked[ibase,ifile]        
                exp_obs = np.exp(1j * phi)
                
                
                ################## model new_coef_H2O ##################
                
                n_new_coef_H2O   = n_air_newcoef_H2O(params_min_new_coef_H2O,wlen_masked,temp,pres,hum,N_CO2)
                phi_new_coef_H2O = (2 * np.pi * path) * (n_new_coef_H2O) / wlen_masked
                exp_new_coef_H2O = np.exp(1j * phi_new_coef_H2O)
                
                # Compute residuals
                resid_new_coef_H2O = np.angle(exp_obs * np.conj(exp_new_coef_H2O))
                
                #Piston Correction
                cf_new_coef_H2O, OPD = op_get_piston_fft(amp,resid_new_coef_H2O,wlen_masked)
                cf_corr_new_coef_H2O = op_corr_piston(cf_new_coef_H2O,wlen_masked,OPD)
                totvis_new_coef_H2O  = np.sum(cf_corr_new_coef_H2O)
                cvis_new_coef_H2O    = cf_corr_new_coef_H2O * np.exp(-1j * np.angle(totvis_new_coef_H2O[...,None]))
                resid_new_coef_H2O   = np.angle(cvis_new_coef_H2O)
                
                #CHI2 Computation
                cost_new_coef_H2O  = compute_chi2_local(resid_new_coef_H2O, weight,params_min_new_coef_H2O)
                chi2_new_coef_H2O += compute_chi2(resid_new_coef_H2O, weight,params_min_new_coef_H2O)
                
                
                #Convertion in degrees
                resid_new_coef_H2O = np.degrees(resid_new_coef_H2O)
               
                
               
                ################## model_VOR ##################
                
                n_vor = op_air_index(wlen_masked,temp-273.15,pres/100,hum/100,N_CO2) - 1
                n_vor -= np.mean(n_vor)
                phi_vor = (2 * np.pi * path) * (n_vor) / wlen_masked
                exp_vor = np.exp(1j * phi_vor)
                
                # Compute residuals
                resid_vor =np.angle(exp_obs*np.conj(exp_vor))
                
                #Piston Correction
                cf_vor, OPD = op_get_piston_fft(amp,resid_vor,wlen_masked)
                cf_corr_vor = op_corr_piston(cf_vor,wlen_masked,OPD)
                totvis_vor  = np.sum(cf_corr_vor)
                cvis_vor    = cf_corr_vor * np.exp(-1j * np.angle(totvis_vor[...,None]))
                resid_vor = np.angle(cvis_vor)
                
                #CHI2 Computation
                cost_vor = compute_chi2_local(resid_vor, weight,[])
                chi2_vor+=(compute_chi2(resid_vor, weight,[],nbases,nfiles))
                
                #Convertion in degrees
                resid_vor = np.degrees(resid_vor )            
                
                
                
                ############################### PLOT ###############################
                
                # new_coef_H2O
                ax1[ibase,ifile].text(3.5e-6,-43,f'chi2 = {cost_new_coef_H2O:.2f}',c = 'black', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_new_coef_H2O ,c="black", label = 'resid_sau_wl ')
                
                # voronin's method
                ax1[ibase,ifile].text(3.5e-6,-31,f'chi2 = {cost_vor:.2f}',c = 'b', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_vor ,'b', label = 'resid_vor')
                
                
                # plot's component
                ax1[ibase,ifile].set_title(f'path = {path:.1f},base ={ibase}', fontsize = 13)
                ax1[ibase,ifile].set_xlabel('wlen (μm)',fontsize=12)
                ax1[ibase,ifile].set_ylabel('phases residuals (°)',fontsize=12)
                ax1[ibase,ifile].axhline(-10,c = 'r',ls = '--')
                ax1[ibase,ifile].axhline(10,c = 'r',ls = '--')
                ax1[ibase,ifile].axhline(0,c='g',ls='-.')
                ax1[ibase,ifile].tick_params(labelbottom = True, labelleft = True)
        plt.ylim(-50,50)
        
        plt.suptitle("phase correction: voronin's new coeff + H2O\n\n", fontsize = 22)
        handles, labels = ax1[0,0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig1.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5.0)
        
        # plt.savefig(os.path.expanduser('~/Documents/Result_phase_multideg_wl/global_rms.png'))
        
    
        
        print('chi2_new_coef_H2O = ',chi2_new_coef_H2O)
        print('chi2_vor          = ',chi2_vor)