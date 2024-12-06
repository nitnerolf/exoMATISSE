"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Functions to extract visibilities from the CF data
Author: fmillour
Date: 18/11/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from   astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt
from op_instruments import *

def op_extract_simplevis2(cfdata, verbose=True, plot=False):
    print('Extracting visibility')
    # Put the sum of the PSD in the first row to the same scale as the other rows
    psd     = np.abs(cfdata['CF']['CF'])**2
    niz     = cfdata['CF']['CF_nbpx']
    if verbose:
        print('Shape of psd:', psd.shape)
    
    sumPSD     = np.sum(psd, axis=1)
    background = np.abs(cfdata['CF']['bckg'])**2
    if verbose:
        print('Shape of sumPSD:', sumPSD.shape)
        print('Shape of background:', background.shape)
    sumBkg  = np.sum(background, axis=0)
    #print('Shape of sumBkg:', sumBkg.shape)
    
    # plt.figure()
    # plt.imshow(sumBkg, cmap='viridis')
    # plt.show()
    
    avgBkg = []
    count = 0
    filtBkg = np.ma.array(sumBkg, mask=sumBkg==0)
    
    # plt.figure()
    # plt.imshow(filtBkg, cmap='viridis')
    # plt.show()
    
    for iwlen in np.arange(sumBkg.shape[0]):
        count+=1
        if count%100 == 0:
            print('iwlen:', iwlen)
        #filtBkg = sigma_clip(sumBkg, sigma=3, maxiters=1)
        vgbk    = np.mean(filtBkg[iwlen, :])
        cnt     = filtBkg[iwlen, :].count()
        origCnt = len(filtBkg[iwlen, :])
        
        #print('vgbk:', vgbk)
        avgBkg.append(vgbk / 5 * (origCnt / cnt)**2)
    
    avgBkg = np.array(avgBkg)
    
    simplevis = np.zeros((7, sumPSD.shape[1]))
    for i in np.arange(7):    
        simplevis[i,:] = 36 / 4 * (sumPSD[i,:] - niz[i,:] * avgBkg) / sumPSD[0,:][None, :]
    simplevis[0,:] /= 9
    
    clipvis = sigma_clip(simplevis[0,:], sigma=3, maxiters=5)
    mask    = simplevis[0,:] < 0.99 * np.median(clipvis)
    if verbose:
        print('median of simplevis:', np.mean(clipvis))
        print('mask:', mask)
    mask2      = np.repeat(mask[None,:], 7, axis=0)
    simplevis2 = np.ma.array(simplevis, mask=mask2)
    
    if verbose:
        print('Shape of simplevis2:', simplevis2.shape)
        print('Shape of mask:', mask2.shape)

    # plt.figure()
    # plt.plot(mask)
    # plt.plot(simplevis[0,:])
    # plt.show()
    
    return simplevis2, mask2

def op_correct_balance_simplevis2(cfdata, verbose=True, plot=False):
    toto