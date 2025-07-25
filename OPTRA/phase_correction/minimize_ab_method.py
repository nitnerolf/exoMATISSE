#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:35:11 2025

@author: nsaucourt
"""

############################### IMPORT ###############################
import os
import sys

# --- Add parent folder temporarily for import ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import os 
import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from scipy.optimize import minimize

from op_corrflux import op_air_index, op_compute_nco2

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
# Compute the global reduced chi2 of the corrected air index
def chi2_global_ab_method(params,wlen,TEMP, PRES, HUM,PATH,PHASE,WEIGHT,AMP, N_CO2=421):
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
            n_model = n_air_ab_method(params,wlen,temp,pres,hum,N_CO2)
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
        all_params = {"ab_method":{"chi2_min":1000000,"params_min":[]}}
        
    try:
        ab_method = all_params['ab_method']
    except:
        print("The JSON file did not have the ab_method : initialization of the parameters ")
        all_params['ab_method']={"chi2_min":1000000,"params_min":[]}
        
    ab_method = all_params['ab_method']
    
    ############################### VARIABLES ###############################
    
    wlen_cm =1e2 * wlen
    nu = 1/wlen_cm
    
    
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
    chi2_ab_method = 0
    

    ############################### MINIMIZATION ###############################
    
    # initial conditions
    vor02 = [1,1,0,0,0,1]
    # vor02 =[ 1.07559753e+00, -5.40699897e-01, -3.55006966e+00, -3.00868296e-03, 2.92266862e+00, -6.10953298e-01]
    # vor02 = [ 1.08702931, -0.45605894, -4.07949542, -0.00448344,  1.53845261, -0.06568879]
    bounds = [(-100000,1000000)]*6 
    eps_vec = [1e-8]*6 
    
    for i in range(10):
        for j in range(3):
            eps_vec = [1e10*10**(-i)*10**(-j)]*6 
            result = minimize(chi2_global_ab_method, vor02, args = (wlen_masked,TEMP, PRES, HUM,PATH,phase_masked,WEIGHT,AMP_masked,N_CO2),bounds = bounds,method='l-bfgs-b',jac = False,options  = {'eps'     : eps_vec,    'maxls'   : 1000,       'ftol'    : 1e-30,    'gtol'   : 1e-15,    'maxiter' : 1000, 'iprint':1 })
            print(result.x)
            print(result.success)
            print(result.nit)
            print(result.message)
            print(result.fun)
            params_min_ab_method = result.x
            chi2_min_ab_method   = result.fun
            vor02 = params_min_ab_method
     
            
    # SAVE IN JSON
    if chi2_min_ab_method < ab_method["chi2_min"]:
        ab_method["chi2_min"] = chi2_min_ab_method
        ab_method["params_min"] = params_min_ab_method.tolist()
        all_params['ab_method'] = ab_method
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
                
                
                
                ################## model ab_method ##################
                
                n_ab_method   = n_air_ab_method(params_min_ab_method,wlen_masked,temp,pres,hum,N_CO2)
                phi_ab_method = (2 * np.pi * path) * (n_ab_method) / wlen_masked
                exp_ab_method = np.exp(1j * phi_ab_method)
                
                # Compute residuals
                resid_ab_method = np.angle(exp_obs * np.conj(exp_ab_method))
                
                #Piston Correction
                cf_ab_method, OPD = op_get_piston_fft(amp,resid_ab_method,wlen_masked)
                cf_corr_ab_method = op_corr_piston(cf_ab_method,wlen_masked,OPD)
                totvis_ab_method  = np.sum(cf_corr_ab_method)
                cvis_ab_method    = cf_corr_ab_method * np.exp(-1j * np.angle(totvis_ab_method[...,None]))
                resid_ab_method   = np.angle(cvis_ab_method)
                
                #CHI2 Computation
                cost_ab_method  = compute_chi2_local(resid_ab_method, weight,params_min_ab_method)
                chi2_ab_method += compute_chi2(resid_ab_method, weight,params_min_ab_method,nbases,nfiles)
                
                #Convertion in degrees
                resid_ab_method = np.degrees(resid_ab_method )
                
                
                
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
                
                # ab_method
                ax1[ibase,ifile].text(3.5e-6,+31,f'chi2 = {cost_ab_method:.2f}',c = 'orange', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_ab_method ,c ="orange", label = 'resid_sau_ab ')
                    
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
        plt.suptitle("phase correction: ab - method\n\n", fontsize = 25)
        handles, labels = ax1[0,0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig1.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5.0)
        
        # plt.savefig(os.path.expanduser(outdir + 'minimized_ab_method.png'))
        
        print('chi2_ab_method =',chi2_ab_method)
        print('chi2_vor       =',chi2_vor)
        