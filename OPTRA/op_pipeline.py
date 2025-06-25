#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /Users/fmillour/exoMATISSE/.venv/bin/python -m  pip install matplotlib pandas tqdm psutil mplcursors astropy numpy astroquery scipy

################################################################################
#
# Pipeline functions for OPTRA
#       __           __           __           __           __           __   
#     .'  `.       .'  `.       .'  `.       .'  `.       .'  `.       .'  `. 
#    /      \     /      \     /      \     /      \     /      \     /      \ 
# _.'        `._.'        `._.'        `._.'        `._.'        `._.'        `
#
# Author: fmillour
# Create Time: 2024-12-13 16:05:00
#
################################################################################

from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
from op_vis        import *
from op_oifits     import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os
from scipy.ndimage import median_filter
from scipy         import *
from scipy         import stats
from tqdm import tqdm

########################################################

def op_sort_files(data_dir):
    print("Sorting files in directory:", data_dir)
    files     = os.listdir(data_dir)
    fitsfiles = [f for f in files if ".fits" in f]
    fitsfiles = sorted(fitsfiles)
    # select  fits files that correspond to observations
    data= {'data_dir': data_dir}
    data['obs']     = []
    data['obs_typ'] = []
    data['obs_MJD'] = []
    data['sky']     = []
    data['sky_MJD'] = []
    data['dark']    = []
    data['dark_MJD']= []

    for fi in tqdm(fitsfiles,desc='Sorting files...'):
        #print(fi)
        hdr = fits.getheader(data_dir+fi)
        catg = hdr['ESO DPR CATG']
        type = hdr['ESO DPR TYPE']
        mjd  = hdr['MJD-OBS']
        #print(fi, inst, catg, type, chip, dit, ndit)
        if catg == 'SCIENCE' and type == 'OBJECT':
            #print("science file!")
            data['obs'].append(fi)
            data['obs_typ'].append('SCI')
            data['obs_MJD'].append(mjd)
        if catg == 'CALIB' and type == 'STD':
            #print("calibrator file!")
            data['obs'].append(fi)
            data['obs_typ'].append('CAL')
            data['obs_MJD'].append(mjd)
        if catg == 'CALIB' and type == 'SKY' :
            #print("sky file!")
            data['sky'].append(fi)
            data['sky_MJD'].append(mjd)
            
    print("Done!")
    return data

########################################################

def op_assign_sky(data):
    """
    Assign sky files to each target
    """
    print("Assigning sky files to targets...")
    indir = data['data_dir']
    data['matched_sky']     = []
    keys_to_match = ['INSTRUME','ESO DET CHIP NAME','ESO DET SEQ1 DIT']
    
    skyfiles    = data['sky']
    all_sky_mjd = np.array(data['sky_MJD'])
    
    for ifile,obsfile in tqdm(enumerate(data['obs']),desc='Assigning skies...'):
        hdr = fits.getheader(indir+obsfile)
        obs_mjd = data['obs_MJD'][ifile]
        #print('obs_mjd:', obs_mjd)
        #print('sky_mjd:', all_sky_mjd)
        mjdiff = np.abs(all_sky_mjd - obs_mjd)
        
        jfile_best = 0
        dif = mjdiff[jfile_best]
        for jfile,skyfile in enumerate(skyfiles):
            hdr2 = fits.getheader(indir+skyfile)
            match = True
            for key in keys_to_match:
                if key not in hdr or key not in hdr2:
                    match = False
                    #data['matched_sky'].append('Nope!')
                    break
                if hdr[key] != hdr2[key]:
                    match = False
                    #data['matched_sky'].append('Nope!')
                    break
            if match:
                newdif = mjdiff[jfile]
                if newdif < dif:
                    dif = newdif
                    jfile_best = jfile
                
        #print(f"Matched {obsfile} ({obs_mjd}) with {skyfiles_sorted[jfile_best]} ({mjd_sorted[jfile_best]})")
        data['matched_sky'].append(skyfiles[jfile_best])
    
    print("Done!")
    return data

########################################################

def op_assign_bias():
    """
    Assign bias to a given target
    """
    pass

########################################################

def op_assign_flat():
    """
    Assign flat to a given target
    """
    pass

########################################################

def op_assign_bpm():
    """
    Assign bad pixel map to a given target
    """
    pass

########################################################

