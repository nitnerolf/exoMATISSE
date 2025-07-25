#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 19:37:23 2025

@author: nsaucourt
"""

############################### IMPORT ###############################

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

sys.path.pop(0)
from module_phase_correction import *



############################### MAIN PROGRAM ###############################
if __name__ == '__main__': 

    ############ FILE OPENER ############
    
    filename = os.path.expanduser('~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_bet_pic.fits')
    # filename = os.path.expanduser('~/Documents/exoMATISSE/OPTRA/phase_correction/fits_phase_CAL.fits')
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
        
        
    ############ VARIABLES ############

    wlen_cm =1e2 * wlen
    nu = 1/wlen_cm

    # Mask
    nu_mask = (nu < 3548) & (nu > 2389)
    
    # masked variables 
    nu_masked = nu[nu_mask]
    wlen_masked = wlen[nu_mask]
    phase_masked = phase[...,nu_mask]
    ERR_masked   = ERR[...,nu_mask]
    AMP_masked   = AMP[...,nu_mask]
    
    # Weight for chi2
    WEIGHT = (1/ERR_masked)**2 
    
    # CHI2 initialization
    chi2_mathar = 0
    chi2_vor = 0
    
    fig1,ax1  = plt.subplots(nbases,nfiles,figsize = (nfiles * 6.5,nbases * 3))
    plt.suptitle(' phase correction : Mathar and Voronin \n',fontsize=15)
    
    ############################### COMPUTE MODELS ###############################
    for ibase in range(nbases):   
        for ifile in range(nfiles):
            # Atmospheric Conditions
            hum = HUM[ibase,ifile]
            pres = PRES[ibase,ifile]
            temp = TEMP[ibase,ifile] 
            cond_atm = (hum,pres, temp )
            
            # variables 
            path = PATH[ibase,ifile]
            amp = AMP_masked[ibase,ifile]
            weight = WEIGHT[ibase,ifile]
            phi = phase_masked[ibase,ifile]
            exp_obs = np.exp(1j * phi)
            
            
            
            ################## model MATHAR ##################
            
            n_1_mathar  = n_air_mathar(cond_atm,nu_masked)
            n_1_mathar -= np.mean(n_1_mathar)
            phi_mathar  = (2 * np.pi * path) * (n_1_mathar) / wlen_masked
            exp_mathar  = np.exp(1j * phi_mathar)

            # Compute residuals
            resid_mathar = np.angle(exp_obs * np.conj(exp_mathar))
            
            #Piston Correction
            cf_mathar, OPD = op_get_piston_fft(amp,resid_mathar,wlen_masked)
            cf_corr_mathar = op_corr_piston(cf_mathar,wlen_masked,OPD)
            totvis_mathar  = np.sum(cf_corr_mathar)
            cvis_mathar    = cf_corr_mathar * np.exp(-1j * np.angle(totvis_mathar[...,None]))
            resid_mathar   = np.angle(cvis_mathar)
            
            #CHI2 Computation
            cost_mathar  = compute_chi2_local(resid_mathar, weight,[])
            chi2_mathar += compute_chi2(resid_mathar, weight,[],nbases,nfiles)
            print("mathar cost (∑r²) =", cost_mathar)
            
            #Convertion in degrees
            resid_mathar =np.degrees(resid_mathar )
        
        
        
            ################## model VORONIN ##################
            
            n_vor   = op_air_index(wlen_masked,temp-273.15,pres/1e2,hum/100,N_CO2 = N_CO2) - 1
            n_vor  -= np.mean(n_vor)
            phi_vor = (2 * np.pi * path) * (n_vor) / wlen_masked
            exp_vor = np.exp(1j * phi_vor)
            
            # Compute residuals
            resid_vor = np.angle(exp_obs*np.conj(exp_vor))
            
            #Piston Correction
            cf_vor,OPD  = op_get_piston_fft(amp,resid_vor,wlen_masked)
            cf_corr_vor = op_corr_piston(cf_vor,wlen_masked,OPD)
            totvis_vor  = np.sum(cf_corr_vor)
            cvis_vor    = cf_corr_vor * np.exp(-1j * np.angle(totvis_vor[...,None]))
            resid_vor   = np.angle(cvis_vor)
            
            #CHI2 Computation
            cost_vor = compute_chi2_local(resid_vor, weight,[])
            chi2_vor+= compute_chi2(resid_vor, weight,[],nbases,nfiles)
            print("voronin cost (∑r²) =", cost_vor)
            
            #Convertion in degrees
            resid_vor = np.degrees(resid_vor)
            
            
            ############################### PLOT ###############################
            
            # MATHAR 
            ax1[ibase,ifile].plot(wlen_masked,resid_mathar,color ='black',label = 'mathar' )
            ax1[ibase,ifile].text(3.5e-6,+31,f'chi2 = {cost_mathar:.2f}',c = 'black', fontsize = 12)
            
            # VORONIN
            ax1[ibase,ifile].plot(wlen_masked,resid_vor,color ='blue',label = 'voronin' )
            ax1[ibase,ifile].text(3.5e-6,-31,f'chi2 = {cost_vor:.2f}',c = 'b', fontsize = 12)
            
            
            # plot's component
            ax1[ibase,ifile].set_title(f'path = {path:.1f},base ={ibase}',fontsize=13)
            ax1[ibase,ifile].set_xlabel('wlen (μm)',fontsize=12)
            ax1[ibase,ifile].set_ylabel('phases residuals (°)',fontsize=12)
            ax1[ibase,ifile].axhline(-10,c = 'r',ls = '--')
            ax1[ibase,ifile].axhline(10,c = 'r',ls = '--')
            ax1[ibase,ifile].set_ylim(-50,50)
    
    handles, labels = ax1[0,0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys(), loc='upper left')
    plt.tight_layout()