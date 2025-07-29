#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python -m  pip install matplotlib pandas tqdm psutil mplcursors astropy numpy astroquery scipy

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
    files = os.listdir(data_dir)
    fitsfiles = [
        f for f in files
        if ".fits" in f and not f.startswith('_') and not f.startswith('.')
    ]
    fitsfiles = sorted(fitsfiles)
    # select  fits files that correspond to observations
    data_collection= {'data_dir': data_dir}
    data_collection['obs']     = []
    data_collection['obs_typ'] = []
    data_collection['obs_MJD'] = []
    data_collection['sky']     = []
    data_collection['sky_MJD'] = []
    data_collection['dark']    = []
    data_collection['dark_MJD']= []

    for fi in tqdm(fitsfiles,desc='Sorting files...'):
        #print(fi)
        hdr = fits.getheader(data_dir+fi)
        catg = hdr['ESO DPR CATG']
        type = hdr['ESO DPR TYPE']
        mjd  = hdr['MJD-OBS']
        #print(fi, inst, catg, type, chip, dit, ndit)
        if catg == 'CALIB' and type == 'STD':
            #print("calibrator file!")
            data_collection['obs'].append(fi)
            data_collection['obs_typ'].append('CAL')
            data_collection['obs_MJD'].append(mjd)
        if catg == 'SCIENCE' and type == 'OBJECT':
            #print("science file!")
            data_collection['obs'].append(fi)
            data_collection['obs_typ'].append('SCI')
            data_collection['obs_MJD'].append(mjd)
        if catg == 'CALIB' and type == 'SKY' :
            #print("sky file!")
            data_collection['sky'].append(fi)
            data_collection['sky_MJD'].append(mjd)
            
    print("Done!")
    return data_collection

########################################################

def op_assign_sky(data_collection):
    """
    Assign sky files to each target
    """
    print("Assigning sky files to targets...")
    indir = data_collection['data_dir']
    data_collection['matched_sky']     = []
    keys_to_match = ['INSTRUME','ESO DET CHIP NAME','ESO DET SEQ1 DIT', 'ESO INS BCD1 NAME', 'ESO INS BCD2 NAME']
    
    skyfiles    = data_collection['sky']
    all_sky_mjd = np.array(data_collection['sky_MJD'])
    
    for ifile,obsfile in tqdm(enumerate(data_collection['obs']),desc='Assigning skies...'):
        hdr = fits.getheader(indir+obsfile)
        obs_mjd = data_collection['obs_MJD'][ifile]
        #print('obs_mjd:', obs_mjd)
        #print('sky_mjd:', all_sky_mjd)
        mjdiff = np.abs(all_sky_mjd - obs_mjd)
        
        sorted_indices = np.argsort(mjdiff)
        sorted_skies = np.array(skyfiles)[sorted_indices]
        mjdiff_sorted = mjdiff[sorted_indices]
        
        jfile_best = 0
        dif = mjdiff_sorted[jfile_best]
        for jfile,skyfile in enumerate(sorted_skies):
            hdr2 = fits.getheader(indir+skyfile)
            match = True
            for i,key in enumerate(keys_to_match):
                if key not in hdr or key not in hdr2:
                    match = False
                    #print('Nope!')
                    #data_collection['matched_sky'].append('Nope!')
                    break
                if hdr[key] != hdr2[key]:
                    match = False
                    #print('Nope!')
                    #data_collection['matched_sky'].append('Nope!')
                    break
                
            if match:
                print(jfile)
                # Stop at first match
                jfile_best = jfile
                break
        if jfile == len(sorted_skies)-1:
            print('reached end of sky files list')
        if  match == False:
            print('No match!!!! ', obsfile, 'with', skyfile)
        
        stringObs = 'Obs: '
        stringSky = 'Sky: '
        for key in keys_to_match:
            stringObs += f"{key}: {hdr[key]} | "
            stringSky += f"{key}: {hdr2[key]} | "
        #print(stringObs)
        #print(stringSky)
            
        #print(f"Matched {obsfile} ({obs_mjd}) with {skyfiles_sorted[jfile_best]} ({mjd_sorted[jfile_best]})")
        data_collection['matched_sky'].append(sorted_skies[jfile_best])
    
    print("Done!")
    return data_collection

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

def op_compute_oifits(starfile, skyfile, badfile, flatfile, shiftfile, outdir=None, verbose=False, plot=False):
    ##########################################################
    # load raw data file
    data = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=verbose, plot=plot)
    data = op_extract_beams(data, verbose=verbose, plot=True)
    
    ##########################################################
    # Compute correlated flux
    data = op_get_corrflux(data, shiftfile, plot=plot, verbose=verbose)
    
    
    data = op_compute_uv(data,False)
    
    ##########################################################
    # Compute squared visibilities    
    data, vis2 = op_extract_simplevis2(data, verbose=verbose, plot=plot)
    op_correct_balance_simplevis2(data, verbose=True, plot=True)
    
    #########################################################
    # Compute output file name
    if outdir is None:
        outdir = os.path.dirname(starfile) + '/'
    basename = os.path.basename(starfile)
    basen    = os.path.splitext(basename)[0]
    directory = data['hdr']['DATE-OBS'].split('T')[0]+'_OIFITS/'
    if not os.path.exists(outdir+directory):
        os.makedirs(outdir+directory)
    chip = data['hdr']['ESO DET CHIP NAME']
    if 'HAWAII' in chip:
        band = 'L'
    elif 'AQUARIUS' in chip:
        band = 'N'
    basen = directory+data['hdr']['INSTRUME'][0:3]    + '_' +\
    data['hdr']['DATE-OBS'].replace(':','').replace('-','')          + '_' +\
    data['hdr']['ESO OBS TARG NAME'].replace(' ','_') + '_' +\
    data['hdr']['ESO DPR CATG'][0:3]                  + '_' +\
    band                                              + '_' +\
    data['hdr']['ESO INS DIL ID']                     + '_' +\
    data['hdr']['ESO INS BCD1 ID'][0]                          +\
    data['hdr']['ESO INS BCD2 ID'][0]
    
    #########################################################
    outfilename = os.path.expanduser(outdir+f'{basen}_oi.fits')
    hdr = data['hdr']
    oiwavelength = op_gen_oiwavelength(data, verbose=verbose, wlen_fin='EFF_WAVE')
    oitarget     = op_gen_oitarget(data, verbose=True, plot=False)
    oirray       = op_gen_oiarray(data, verbose=True, plot=False)
    oivis        = op_gen_oivis(data, cfin='CF_piston_corr', verbose=verbose, plot=False)
    oivis2        = op_gen_oivis2(data, v2in='simplevis2', verbose=verbose, plot=False)
    op_write_oifits(outfilename, hdr, oiwavelength, oirray, oitarget, oivis, oivis2, oit3=None)
    
    #########################################################
    

########################################################