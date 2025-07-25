#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:09:26 2025

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
import astropy.constants as const
from op_corrflux import op_air_index, op_compute_nco2
from scipy import interpolate
import math
import json
sys.path.pop(0)

from module_phase_correction import *

############################### GLOBAL VARIABLES ###############################
#plot variables
plot_ab_method    = True
plot_new_coef     = True
plot_new_coef_H2O = True
plot_vor          = True

# Put CAL if you want to work on the Calibrators, bet_pic if you want to work on Beta Pic
obj = 'CAL'
obj = "bet_pic"



###############################  MAIN PROGRAM ###############################

if __name__ == '__main__': 

    ############ FILE OPENER ############
    # Files' name 
    jsonfile = os.path.expanduser(f'~/Documents/exoMATISSE/OPTRA/phase_correction/parameters_min_{obj}.json')
    filename = os.path.expanduser(f'~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_{obj}.fits')
    outdir   = os.path.expanduser('~/Documents/Result_phase_correction/')
    
    # Fits' extracion
    with fits.open(filename) as fh:
        hdr    = fh[0].header
        nfiles = fh['OI_VIS'].header['NFILES']
        nbases = 6
        wlen   = fh['OI_VIS'].data['WLEN'][0][0]
        nwlen  = len(wlen)
        phase  = fh['OI_VIS'].data['PHASE'].reshape(nbases,nfiles,nwlen)
        PATH   = fh['OI_VIS'].data['PATH']
        HUM    = fh['OI_VIS'].data['HUM'] * 100
        PRES   = fh['OI_VIS'].data['PRES'] * 1e2
        TEMP   = fh['OI_VIS'].data['TEMP'] + 273.15
        ERR    = fh['OI_VIS'].data['ERR'].reshape(nbases,nfiles,nwlen)
        AMP    = fh['OI_VIS'].data['AMP'].reshape(nbases,nfiles,nwlen)
        N_CO2  = op_compute_nco2({'hdr':hdr})
    
    #json's extraction
    with open(jsonfile, 'r') as f:
        all_params = json.load(f)
    
    ab_method     = all_params["ab_method"]
    new_coef      = all_params["new_coef"]
    new_coef_H2O  = all_params["new_coef_H2O"]
    
    params_min_ab_method    = ab_method['params_min']
    params_min_new_coef     = new_coef['params_min']
    params_min_new_coef_H2O = new_coef_H2O['params_min']      
    
    
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
    chi2_vor          = 0
    chi2_ab_method    = 0
    chi2_new_coef     = 0
    chi2_new_coef_H2O = 0
     

    
    fig1,ax1  = plt.subplots(nbases,nfiles,sharex=True,sharey=True,figsize = (nfiles * 6.5,nbases * 3 ))
    
    
    
    ############################### COMPUTE MODELS ###############################
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
            
            
            
            ################## model new_coef ##################
            
            n_new_coef   = n_air_newcoef(params_min_new_coef,wlen_masked,temp,pres,hum,N_CO2,wlen_ref = wlen_ref)
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
            
            
            
            ################## model new_coef_H2O ##################
            
            n_new_coef_H2O   = n_air_newcoef_H2O(params_min_new_coef_H2O,wlen_masked,temp,pres,hum,N_CO2,wlen_ref =wlen_ref)
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
            chi2_new_coef_H2O += compute_chi2(resid_new_coef_H2O, weight,params_min_new_coef_H2O,nbases,nfiles)
            
            
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
            
            
            
            # ab_method
            if plot_ab_method:
                ax1[ibase,ifile].text(3.5e-6,+31,f'chi2 = {cost_ab_method:.2f}',c = 'orange', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_ab_method ,c ="orange", label = 'resid_sau_ab ')
                
            # new_coef_H2O
            if plot_new_coef_H2O:
                ax1[ibase,ifile].text(3.5e-6,-43,f'chi2 = {cost_new_coef_H2O:.2f}',c = 'black', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_new_coef_H2O ,c="black", label = 'resid_sau_wl ')
            
            # new_coef
            if plot_new_coef:
                ax1[ibase,ifile].text(3.5e-6,+41,f'chi2 = {cost_new_coef:.2f}',c = 'purple', fontsize = 12)
                ax1[ibase,ifile].plot(wlen_masked, resid_new_coef ,c= 'purple',ls='--', label = 'resid_sau_wl_strict ')
            
            # voronin's method
            if plot_vor:
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
    plt.suptitle('comparison of phase correction\n\n', fontsize = 15)
    
    #legend 
    handles, labels = ax1[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.tight_layout()
    # plt.savefig(os.path.expanduser(outdir + f'phase_comparison_{obj}.png'))
    
    print('chi2_new_coef     = ',chi2_new_coef)
    print('chi2_ab_method    = ',chi2_ab_method)
    print('chi2_new_coef_H2O = ',chi2_new_coef_H2O)
    print('chi2_vor          = ',chi2_vor)