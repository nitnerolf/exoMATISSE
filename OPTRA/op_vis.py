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

def op_extract_simplevis2(cfdata, verbose=True, plot=False):
    print('Extracting visibility')
    # Put the sum of the PSD in the first row to the same scale as the other rows
    psd     = np.abs(cfdata['CF']['data'])**2
    print('Shape of psd:', psd.shape)
    niz     = cfdata['CF']['CF_nbpx'] # Number of PSD pixels in the fringe peak
    if verbose:
        print('Shape of psd:', psd.shape)
    
    sumPSD     = np.sum(psd,    axis=1) # Sum on the frames?
    npx = np.sum((sumPSD!=0), axis=-1)
    #print('npx:', npx)
    
    if plot:
        plt.figure()
        plt.plot(npx.T)
        plt.plot(niz.T, linestyle=':')
        #plt.show()
        
    ssumPSD    = np.sum(sumPSD, axis=-1) # Sum on the PSD pixels
    background = np.abs(cfdata['CF']['bckg'])**2
    if verbose:
        print('Shape of sumPSD:',     sumPSD.shape)
        print('Shape of ssumPSD:',    ssumPSD.shape)
        print('Shape of background:', background.shape)
    sumBkg  = np.sum(background, axis=0)
    print('Shape of sumBkg:', sumBkg.shape)
    
    if plot:
        plt.figure()
        plt.imshow(sumBkg,vmin=0,vmax=5e6)
        plt.title('Background')
        #plt.show()
    
    filtBkg = np.ma.array(sumBkg, mask=(sumBkg==0))
    #for iwlen in np.arange(sumBkg.shape[0]):
    #    filtBkg[iwlen,:] = np.ma.array(sumBkg[iwlen,:], mask=(sumBkg[iwlen,:]==0))
    
    if plot:
        plt.figure()
        plt.imshow(filtBkg,vmin=0,vmax=8e6)
        plt.title('Filtered background')
        #plt.show()
    
    iwlen=560
    iwlen=400
    iwlen=60
    
    if plot:
        plt.figure()
        plt.plot(sumBkg[iwlen,:])
        plt.plot(filtBkg[iwlen,:],color='red')
        
    sbkg = sumBkg[iwlen,:]
    med = np.median(sbkg[np.where(sbkg!=0)])
    print('Median:', med)
    
    if plot:
        plt.axhline(med)
    mean = np.mean(sbkg[np.where(sbkg!=0)])
    print('Mean:', mean)
    
    if plot:
        plt.axhline(mean,linestyle=':')
        plt.title(f'Background at wavelength {iwlen}')
        for i in np.arange(7):
            plt.plot(sumPSD[i,iwlen,:])
        plt.ylim(-0.1*med, 2*med)
        #plt.show()
    
    avgBkg = np.zeros(sumBkg.shape[0])
    for iwlen in np.arange(sumBkg.shape[0]):
        sbkg = sumBkg[iwlen,:]
        #avgBkg[iwlen] = np.median(sbkg[np.where(sbkg!=0)])
        avgBkg[iwlen] = np.mean(sbkg[np.where(sbkg!=0)])
    
    
    if verbose:
        print('Shape of avgBkg:', avgBkg.shape)
    
    if plot:
        plt.figure()
        plt.title(f'Averaged background')
        plt.plot(avgBkg)
        #plt.show()
    
    simplevis = np.zeros((7, sumPSD.shape[1]))
    for i in np.arange(7):    
        if i == 0:
            # Factor is 6 baselines **2 divided by 4 telescopes
            simplevis[i,:] = 36 / 4 * (ssumPSD[0,:] / (npx[0,:]+0) - avgBkg) / (ssumPSD[0,:] / npx[0,:])
        else:
            # Factor 2 here because we have only half the pixels in the photometric peak
            simplevis[i,:] = 36 / 4 * (ssumPSD[i,:] / (npx[i,:]+0) - avgBkg) / (ssumPSD[0,:] / npx[0,:] - avgBkg) * npx[i,:] / npx[0,:]
    simplevis[0,:] /= 9
    
    clipvis = sigma_clip(simplevis[0,:], sigma=3, maxiters=5)
    mask    = simplevis[0,:] < 0.95 * np.median(clipvis)
    if verbose:
        print('median of simplevis:', np.mean(clipvis))
        print('mask:', mask)
    mask2      = np.repeat(mask[None,:], 7, axis=0)
    simplevis2 = np.ma.array(simplevis, mask=mask2)
    
    if verbose:
        print('Shape of simplevis2:', simplevis2.shape)
        print('Shape of mask:', mask2.shape)

    if plot:
        plt.figure()
        plt.title(f'Visibilities')
        plt.plot(mask)
        plt.plot(simplevis[0,:])
        plt.show()
    
    cfdata['VIS2'] = {}
    cfdata['VIS2']['simplevis2'] = simplevis2
    
    return cfdata,simplevis2, mask2

def op_correct_balance_simplevis2(cfdata, verbose=True, plot=False):
    toto