#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Flux handling methods
# Author: fmillour
# Date: 01/07/2024
# Project: OPTRA
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import inspect

def op_extract_beams(rawdata, verbose=True, plot=False):
    #---------------------------------------------------
    data = rawdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
    
    intfdata    = rawdata['INTERF']['data']
    intfshape   = np.mean(intfdata,axis=0)
    intfshape  /= np.max(intfshape)[None,...]
    intftime    = np.mean(intfdata,axis=(1,2))
    intfprofile = np.mean(intfdata,axis=(0,1))
    intfprofile /= np.max(intfprofile)[None,...]
    rawdata['INTERF']['shape'] = intfshape
    rawdata['INTERF']['time']  = intftime
    rawdata['INTERF']['time']  = intfprofile

    if plot:
        plt.figure(1)
        plt.plot(intftime, label = 'Interf vs time')
        
        plt.figure(2)
        plt.plot(intfprofile, label = 'Interf profile')
        
        fig, axs = plt.subplots(1, 5, figsize=(10, 8), sharey=True)
        axs = axs.flatten()
        
        
    for ikey,key in enumerate(rawdata['PHOT']):
        photi = rawdata['PHOT'][key]['data']
        # Do something with photi
        # In ESO files, the data is a 3D array with shape (N, M, P)
        # where N is the number of frames, P and P are the two detector dimensions.
        photishape    = np.mean(photi,axis=0)
        photishape   /= np.max(photishape)[None,...]
        photitime     = np.mean(photi,axis=(1,2))
        photiprofile  = np.mean(photi,axis=(0,1))
        photiprofile /= np.max(photiprofile)[None,...]
        if verbose>1:
            print('Photi:', np.shape(photi), 'Photishape:', np.shape(photishape), 'Photitime:', np.shape(photitime))
        
        rawdata['PHOT'][key]['shape'] = photishape
        rawdata['PHOT'][key]['time']  = photitime
        rawdata['PHOT'][key]['profile']  = photiprofile
    
    if plot:
        photi = []
        phots = []
        photp = []
        for key in rawdata['PHOT']:
            photi.append(rawdata['PHOT'][key]['time'])
            phots.append(rawdata['PHOT'][key]['shape'])
            photp.append(rawdata['PHOT'][key]['profile'])
        # Determine global vmin and vmax for all photometry shapes
        #vmin = 0
        #vmax = 1
        
        im = axs[0].imshow(intfshape)
        axs[0].set_title(f'Intf shape')
        
        for i,key in enumerate(rawdata['PHOT']):
            plt.figure(1)
            plt.plot(photi[i], label = f'Phot {i+1}')
            
            plt.figure(2)
            plt.plot(photp[i], label = f'Phot {i+1}')
            
            im = axs[i+1].imshow(phots[i])
            axs[i+1].set_title(f'Phot {i+1} shape')
        plt.title('Photometry vs time')
        
        #plt.tight_layout()
        fig.colorbar(im, ax=axs[:min(4, len(phots))], orientation='vertical', fraction=0.02)
            
    if plot:
        plt.show()
    return rawdata
        

def op_compute_kappa():
    here_do_domething

def op_apply_kappa():
    here_do_domething

def op_filter_beams():
    here_do_domething

def op_compute_flux_fac():
    here_do_domething
