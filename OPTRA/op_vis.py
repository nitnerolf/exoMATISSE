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

################################################################################
def op_extract_simplevis2(cfdata, verbose=True, plot=False):
    print('Extracting visibility')
    verbose=True
    # Put the sum of the PSD in the first row to the same scale as the other rows
    psd0     = np.abs(cfdata['FFT']['data'])**2
    nfreq  = np.shape(psd0)[2]
    nfreq2 = int(nfreq/2)
    psd    = psd0[:,:,0:nfreq2]
    
    zone     = cfdata['CF']['zone'] # mask to extract each peak
    if verbose:
        print('Shape of psd:',   psd.shape)
        print('Shape of zone:', zone.shape)
    
    avgPSD     = np.mean(psd, axis=0) # Sum on the frames?
    
    if plot:
        plt.figure()
        plt.imshow(avgPSD, vmax=2e5)
        plt.show()
        
    bkPSD  = avgPSD * (1-np.sum(zone, axis=0)) # Background PSD
    bkPSDm = np.ma.array(bkPSD, mask=(bkPSD==0)) # Masked background PSD
    
    if plot:
        plt.figure()
        plt.imshow(bkPSDm, vmax=2e5)
        plt.show()
    
    pkPSD  = avgPSD[None,:,:] * zone
    pkPSDm = np.ma.array(pkPSD, mask=(pkPSD==0)) # Masked peak PSD
    
    
    if plot:
        plt.figure()
        plt.imshow(np.sum(pkPSDm,axis=0),vmax=2e5)
        plt.show()
    
    background = np.mean(bkPSDm, axis=1) # Sum on the PSD pixels
    if verbose:
        print('Shape of pkPSD:',     pkPSD.shape)
        print('Shape of pkPSDm:',     pkPSDm.shape)
        print('Shape of background:', background.shape)

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
    
    simplevis2 = np.zeros((zone.shape[0], zone.shape[1]))
    for i in np.arange(zone.shape[0]):    
        simplevis2[i,:] = 9**2 / 2**2 / 2 * np.sum(pkPSDm[i,...] - background[:,None], axis=1) / np.sum(pkPSDm[0,...] - background[:,None], axis=1)
        
        if plot:
            plt.figure()
            plt.plot(np.sum(pkPSD[i,...], axis=1))
            plt.plot(np.sum(pkPSD[0,...], axis=1) / 1e1, color='green')
            plt.plot(background, color='red')
            plt.title(f'Simple visibility for zone {i}')
            plt.xlabel('Wlength index')
            plt.ylabel('PSD')
            plt.yscale('log')
            plt.show()
    # Renormalize back zero frequency
    simplevis2[0,:] /= 9**2 / 2**2 / 2
    
    if verbose:
        print('shape of simplevis:', simplevis2.shape)
    
    if plot:
        plt.figure()
        plt.title(f'Visibilities')
        for i in np.arange(7):    
            plt.plot(simplevis2[i,:])
        plt.show()
    
    cfdata['VIS2'] = {}
    cfdata['VIS2']['simplevis2'] = simplevis2
    
    return cfdata,simplevis2

################################################################################
def op_correct_balance_simplevis2(cfdata, verbose=True, plot=False):
    toto