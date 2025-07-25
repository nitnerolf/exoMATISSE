#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 18:50:06 2025

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
obj = 'CAL'
# obj = "bet_pic"


############################### FUNCTIONS ###############################

################################################################
# Compute the global rms of the corrected air index

def chi2_global_newcoef(params,wlen,TEMP, PRES, HUM,PATH,PHASE,WEIGHT,AMP, N_CO2=421):
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
            n_model = n_air_newcoef(params,wlen,temp,pres,hum,N_CO2)
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
        all_params = {"new_coef":{"chi2_min":1000000,"params_min":[]}}
        
    try:
        new_coef = all_params['new_coef'] 
    except:
        print("The JSON file did not have the new_coef method: initialization of the parameters ")
        all_params['new_coef']={"chi2_min":1000000,"params_min":[]}
        
    new_coef = all_params['new_coef'] 
    
    
    
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
    chi2_new_coef = 0
    
    
    ############################### MINIMIZATION ###############################
    # initial conditions
    vor0 = [4290.9/wlen_ref, 2684.9/wlen_ref,2775.6/wlen_ref,4223.1/wlen_ref, 2769.1/wlen_ref,2598.5/wlen_ref,2.897e-5, 8.573e-7,1.862e-6,2.728e-5, 6.620e-7,2.788e-6]
    # vor0 = np.array([1.2259709458541017, 0.7671140710489247, 0.7930282745651468, 1.2065993699982815, 0.7911710677055321, 0.7424280422630943, 3.138822238342934e-05, -6.706725912965262e-06, 8.965231151906084e-07, 2.7105138691498165e-05, -1.1914586067071067e-05, 3.249136289437916e-06])
    # vor0 = [ 1.22597093e+00,  7.67114049e-01,  7.93028318e-01,  1.20659936e+00,7.91171017e-01,  7.42428089e-01,  3.06025502e-05, -6.08629249e-06,3.08463088e-07,  2.61256033e-05, -1.13041175e-05,  4.55515338e-06]
    eps_vec = [1e-5]*6 + [1e-12]*6
    bounds = [(0.5,1.5)]*6 + [(-10,10)]*6
    # for i in range(3):
    #     for j in range(5):
            # eps_vec = [1e-4*10**(-i)]*6 + [1e-6*10**(-j)]*6
    
    result = minimize(chi2_global_newcoef, vor0, args = (wlen_masked,TEMP, PRES, HUM,PATH,phase_masked,WEIGHT,AMP_masked,N_CO2),method='l-bfgs-b',bounds = bounds,jac = False,options  = {'eps'     : eps_vec,    'maxls'   : 1000,       'ftol'    : 1e-30,    'gtol'   : 1e-15,    'maxiter' : 1000, 'iprint':1 })
    print(result.x)
    print(result.success)
    print(result.nit)
    print(result.message)
    print(result.fun)
    params_min_new_coef = result.x
    chi2_min_new_coef  = result.fun
    vor0 = params_min_new_coef

    # SAVE IN JSON
    if chi2_min_new_coef < new_coef["chi2_min"]:
        new_coef["chi2_min"] = chi2_min_new_coef
        new_coef["params_min"] = params_min_new_coef.tolist()
        new_coef["params_min"].append(0)
        new_coef["params_min"].insert(9,0)
        new_coef["params_min"].insert(6,1)
        new_coef["params_min"].insert(3,1)
        all_params['new_coef'] = new_coef
        print(f"CHI2 is better -> dumped in parameters_min_{obj}.json")
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
                
                ################## model new_coef ##################
                
                n_new_coef   = n_air_newcoef(params_min_new_coef,wlen_masked,temp,pres,hum,N_CO2)
                phi_new_coef = (2 * np.pi * path) * (n_new_coef) / wlen_masked
                exp_new_coef = np.exp(1j * phi_new_coef)
                
                # Compute residuals
                resid_new_coef = np.angle(exp_obs * np.conj(exp_new_coef))
                
                #Piston Correction
                cf_new_coef, OPD = op_get_piston_fft(amp,resid_new_coef,wlen_masked)
                cf_corr_new_coef = op_corr_piston(cf_new_coef,wlen_masked,OPD)
                totvis_new_coef  = np.sum(cf_corr_new_coef)
                cvis_new_coef    = cf_corr_new_coef * np.exp(-1j * np.angle(totvis_new_coef[...,None]))
                resid_new_coef   = np.angle(cvis_new_coef)
                
                #CHI2 Computation
                cost_new_coef  = compute_chi2_local(resid_new_coef, weight,params_min_new_coef)
                chi2_new_coef += compute_chi2(resid_new_coef, weight,params_min_new_coef,nbases,nfiles)
                
                
                #Convertion in degrees
                resid_new_coef = np.degrees(resid_new_coef)
                
                
                
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
    
                # new_coef
                ax1[ibase,ifile].text(3.5e-6,+41,f'chi2 = {cost_new_coef:.2f}',c = 'purple', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_new_coef ,c= 'purple',ls='--', label = 'resid_sau_wl_strict ')
                
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
        
        plt.suptitle("phase correction: voronin's new coeff \n\n", fontsize = 22)
        handles, labels = ax1[0,0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig1.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5.0)
        
        # plt.savefig(os.path.expanduser(f'~/Documents/Result_phase_multideg_wl/global_rms_{obj}.png'))
        
        
        
        print('chi2_new_coef =', chi2_new_coef)
        print('chi2_vor      =', chi2_vor)
        
        
    
            