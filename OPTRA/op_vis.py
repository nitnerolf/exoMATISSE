#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
# Functions to extract visibilities from the CF data
# Author: fmillour
# Date: 18/11/2024
# Project: OPTRA
#
################################################################################

from   astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt
from op_instruments import *
from scipy         import stats
import os

################################################################################
def op_compute_vis_bias(cfdata, verbose=True, plot=False):
    print('Computing visibility bias')
    verbose=True
    # Put the sum of the PSD in the first row to the same scale as the other rows
    psd0     = np.abs(cfdata['FFT']['dsp'])
    nfreq  = np.shape(psd0)[2]
    nfreq2 = int(nfreq/2)
    psd    = psd0[...,0:nfreq2]
    
    zone     = cfdata['CF']['zone'] # mask to extract each peak
    if verbose:
        print('Shape of psd:',   psd.shape)
        print('Shape of zone:', zone.shape)
    
    avgPSD     = np.mean(psd, axis=0) # Average on the frames
    
    if verbose:
        print('Shape of avg psd:',   avgPSD.shape)
    if plot:
        plt.figure()
        plt.imshow(avgPSD, vmax=2e5)
        plt.title('Average PSD')
        plt.xlabel('Frequency')
        plt.ylabel('Wavelength index')
        plt.colorbar(label='PSD')
        plt.show()
        
    bkPSD  = avgPSD * (1-np.sum(zone, axis=0)) # Background PSD
    bkPSDm = np.ma.array(bkPSD, mask=(bkPSD==0)) # Masked background PSD
    
    if plot:
        plt.figure()
        plt.imshow(bkPSDm, vmax=2e5)
        plt.show()
    
    #background = np.mean(bkPSDm, axis=1) # Average on the PSD pixels
    background = np.ma.median(bkPSDm, axis=1) # Average on the PSD pixels
    #background = stats.trim_mean(bkPSDm, axis=1, proportiontocut=0.1)
    
    iwlen=560
    iwlen=400
    iwlen=60
    iwlen=1289
    
    if plot:
        plt.figure()
        plt.plot(avgPSD[iwlen,:])
        plt.axhline(background[iwlen],color='red')
        plt.yscale('log')
        plt.title(f'Average PSD at wavelength {iwlen}')
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.show()
    
    
    if verbose:
        print('Shape of background:', background.shape)
    
    
    cfdata['VIS2']['bias'] = background
    return cfdata
    
################################################################################
def op_extract_simplevis2(cfdata, verbose=True, plot=False):
    
    cfdata['VIS2'] = {} # Add a VIS2 key to the cfdata dictionary
    cfdata = op_compute_vis_bias(cfdata, verbose=verbose, plot=plot)
    
    zone     = cfdata['CF']['zone'] # mask to extract each peak
    nbases  = zone.shape[0]
    psd0     = np.abs(cfdata['FFT']['dsp'])
    nfreq0   = psd0.shape[2]
    psd    = psd0[...,0:nfreq0//2]
    background = cfdata['VIS2']['bias'] # Background PSD
    
    if verbose:
        print('Shape of psd0:',   psd0.shape)
        print('Number of frequencies0:', nfreq0)
        print('Shape of psd:',   psd.shape)
        print('Shape of zone:', zone.shape)
        print('Shape of background:', background.shape)
        
    nframes = psd.shape[0]
    nwlen   = psd.shape[1]
    nfreq   = psd.shape[2]
    nbases  = zone.shape[0]
    if verbose:
        print('Number of frames:', nframes)
        print('Number of bases:', nbases)
        print('Number of frequencies:', nfreq)
        print('Number of wavelengths:', nwlen)
    
    pkPSD  = psd[:,None,...] * zone[None,...]
    pkPSDm = np.ma.array(pkPSD, mask=(pkPSD==0)) # Masked peak PSD
    #pkPSDm[:,0,...] /= nbases * 2 # Renormalize the first zone to the same scale as the others
    if plot:
        plt.figure()
        plt.imshow(np.sum(pkPSDm,axis=(0,1)))#,vmax=2e5)
        plt.title('Sum of peak PSD')
        plt.show()
    if verbose:
        print('Shape of pkPSD:',     pkPSD.shape)
        print('Shape of pkPSDm:',     pkPSDm.shape)
   
    simpleCF2   = np.sum(pkPSDm - background[None,None,:,None], axis=-1)
    simpleflux2 = simpleCF2[:,0,...]
    simplevis2  = np.mean(simpleCF2, axis=0) / np.mean(simpleflux2, axis=0)[None,:]
    simplevis2[1:,...] *= nbases  # Renormalize visibilities
    if verbose:
        print('Shape of simpleCF2:', simpleCF2.shape)
        print('Shape of simpleflux2:', simpleflux2.shape)
        print('Shape of simplevis2:', simplevis2.shape)
    
    if plot:
        plt.figure()
        for i in np.arange(nbases):
            plt.plot(np.mean(simpleCF2[i,...],axis=0))
        plt.plot(np.mean(simpleflux2,axis=0), color='green')
        plt.plot(background, color='red')
        plt.title(f'Simple visibility for zone {i}')
        plt.xlabel('Wlength index')
        plt.ylabel('PSD')
        plt.yscale('log')
        plt.show()
    # Renormalize back zero frequency
    #simplevis2[0,:] /= 9**2 / 2**2 / 2
    
    if verbose:
        print('shape of simplevis:', simplevis2.shape)
    
    if plot:
        plt.figure()
        plt.title(f'Visibilities')
        for i in np.arange(7):    
            plt.plot(simplevis2[i,:])
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Wavelength index')
        plt.ylabel('Squared Visibility')
        plt.legend([f'V2_{i+1}' for i in range(7)])
        plt.grid()
        plt.show()
    
    cfdata['VIS2'] = {}
    cfdata['VIS2']['simplevis2'] = simplevis2
    cfdata['VIS2']['simpleCF2'] = simpleCF2
    cfdata['VIS2']['simpleflux2'] = simpleflux2
    
    return cfdata,simplevis2

################################################################################
def op_average_vis2(cfdata, verbose=True, plot=False):
    
    toto
    
################################################################################
def op_correct_balance_simplevis2(cfdata, verbose=True, plot=False):
    toto

################################################################################
def op_compute_vfactor(cfdata, verbose=True, plot=False):
    # Find FT file corresponding to the science file
    scifile = cfdata['hdr']['filename']
    basedir = os.path.dirname(scifile)
    ftfile = None
    


    vfac = np.zeros((n_valid_frames, n_base_ft))
    
    for iB in range(6):
        for it, t in enumerate(mjds_valid):
            timela=np.argmin(np.abs(time_ft - t))
            timeha=np.argmin(np.abs(time_ft - (t + exp_time / 3600 / 24)))

            cf_ft_real = np.real(cf_ft[timela:timeha, indexBasefromGV2MT[bcd][iB]])
            cf_ft_imag = np.imag(cf_ft[timela:timeha, indexBasefromGV2MT[bcd][iB]])
            cf_ft_real_var = np.nanvar(cf_ft_real, axis=0)
            cf_ft_imag_var = np.nanvar(cf_ft_real, axis=0)
            n_frame_ft_in_this_matisse_frame = cf_ft_real.shape[0]

            real_sum_1 = np.nansum(cf_ft_real)**2
            imag_sum_1 = np.nansum(cf_ft_imag)**2
            real_var_sum = cf_ft_real_var.sum() * n_frame_ft_in_this_matisse_frame
            imag_var_sum = cf_ft_imag_var.sum() * n_frame_ft_in_this_matisse_frame
            real_sum_2 = (np.nansum(cf_ft_real, axis=1)**2).sum()
            imag_sum_2 = (np.nansum(cf_ft_imag, axis=1)**2).sum()

            vfac[it, iB] = (real_sum_1 + imag_sum_1 - real_var_sum - imag_var_sum) \
                            / (real_sum_2 + imag_sum_2 - real_var_sum - imag_var_sum) * 1 / n_frame_ft_in_this_matisse_frame


################################################################################
def op_correct_vfactor(cfdata, verbose=True, plot=False):
    toto