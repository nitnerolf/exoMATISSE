#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Cosmetics for raw data
# Author: fmillour, nsaucourt
# Date: 01/07/2024
# Project: OPTRA
#
# This module provides functions to process and calibrate raw data from FITS 
# files. It includes functions to interpolate bad pixels, load and apply bad 
# pixel maps, load and apply flat field maps, subtract sky data, and display 
# the structure and header of FITS files.
#
################################################################################

from   astropy.io import fits
from   scipy.ndimage import median_filter
from   scipy import *
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
from   op_instruments import *
from   matplotlib.ticker import MultipleLocator
from   copy import deepcopy
import numpy.linalg as lin
from   op_parameters import *
from   astroquery.simbad import Simbad
from   astropy.coordinates import SkyCoord
from   astropy.time import Time
import astropy.units as u
import inspect
import os

##############################################
# Function to interpolate bad pixels
def op_interpolate_bad_pixels(data, bad_pixel_map, method='gaussian', add_bad=None, verbose=False, plot=False):
    #---------------------------------------------------
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    #---------------------------------------------------

    if np.any(add_bad):
        bad_pixel_map = bad_pixel_map.copy()  # Don't modify input bad pixel map, use a copy
        for j in range(len(add_bad)):
            bad_pixel_map[add_bad[j][1], add_bad[j][0]] = 1
    
    if plot:
        med = np.nanmedian(data.flatten())
        mad = np.nanmedian(np.abs(data.flatten() - med))
        vmin = med - 3 * mad
        vmax = med + 10 * mad
        fig, ax = plt.subplots(1, 3, figsize=(8, 6), sharex=True, sharey=True)
        ax = ax.flatten()
        ax[0].imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax[0].set_title('Original Data')
        ax[0].set_xlabel('X-axis')
        ax[0].set_ylabel('Y-axis')
        ax[1].imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax[1].set_title('Data with Bad Pixels')
        ax[1].set_xlabel('X-axis')
        ax[1].set_ylabel('Y-axis')
    
    data[bad_pixel_map] = np.nan
    
    if method == 'median':
        # Apply a median filter to the data
        filtered_data = median_filter(data, size=3)
        #plt.imshow(filtered_data, cmap='gray')
        # Replace bad pixels with the median filtered values
        data[bad_pixel_map] = filtered_data[bad_pixel_map]
    
    elif method == 'mean':
        # Iteratively fill NaNs (bad pixels) using the mean of surrounding pixels until all are filled
        mask = np.isnan(data)
        iteration = 0
        while np.any(mask):
            # Compute mean of 3x3 neighborhood, ignoring NaNs
            inds = np.argwhere(mask)
            for y, x in inds:
                y0, y1 = max(0, y-1), min(data.shape[0], y+2)
                x0, x1 = max(0, x-1), min(data.shape[1], x+2)
                
                # plt.figure()
                # plt.imshow(filtered_data[y0:y1, x0:x1], cmap='gray', origin='lower')
                # plt.show()
                window = data[y0:y1, x0:x1]
                val = np.nanmean(window)
                data[y, x] = val
            mask = np.isnan(data)
            iteration += 1
            if iteration > 15:  # Prevent infinite loops
                break
    
    elif method == 'astrofix':
        fixed_img,para,TS=astrofix.Fix_Image(data,"asnan",max_clip=1)
    
    elif method == 'gaussian':
        # Use Gaussian interpolation to fill NaNs, method taken from AMICAL software, MIT licence, author A. Soulain ()
        # See https://github.com/SAIL-Labs/AMICAL/tree/main
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        # Create a Gaussian kernel
        kernel = Gaussian2DKernel(3./2.355)
        # Interpolate NaNs using the Gaussian kernel
        data = interpolate_replace_nans(data, kernel)
    
    if plot:
        ax[2].imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax[2].set_title('Interpolated Data')
        ax[2].set_xlabel('X-axis')
        ax[2].set_ylabel('Y-axis')
        plt.show()
    
    return data

##############################################
# Load bad pixel map
def op_load_bpm(filename, verbose=False):
    #---------------------------------------------------
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    #---------------------------------------------------
    fh = fits.open(filename)
    bpm = fh[0].data.astype(bool)
    fh.close()
    return bpm

##############################################
# Apply bpm
def op_apply_bpm(rawdata, bpmap, verbose=False):
    #---------------------------------------------------
    data = rawdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
    
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf   = rawdata['INTERF']['data']
    nframe = np.shape(intf)[0]
    if verbose: print(f'Processing {nframe} frames')
    wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    for i in range(nframe):
        intf[i] = op_interpolate_bad_pixels(intf[i], wbpm)
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot   = rawdata['PHOT'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        for i in range(nframe):
            phot[i] = op_interpolate_bad_pixels(phot[i], wbpm)
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        for i in range(nframe):
            other[i] = op_interpolate_bad_pixels(other[i], wbpm)
        rawdata['OTHER'][key]['data'] = other
        
    return rawdata

##############################################
# Load flat field map
def op_load_ffm(filename, verbose=False):
    #---------------------------------------------------
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    #---------------------------------------------------
    
    fh = fits.open(filename)
    ffm = fh[0].data.astype(float)
    fh.close()
    return ffm

##############################################
# Apply flat field map
def op_apply_ffm(rawdata, ffmap, verbose=False):
    #---------------------------------------------------
    data = rawdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
        
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf   = rawdata['INTERF']['data']
    
    nframe = np.shape(intf)[0]
    if verbose: print(f'Processing {nframe} frames')
        
    wffm = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    
    intf /= wffm[None,...]
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot   = rawdata['PHOT'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        if verbose: print('phot:', np.shape(phot), 'wffm:', np.shape(wffm))
        phot /= wffm[None,...]
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # correct flat field
        other /= wffm[None,...]
        rawdata['OTHER'][key]['data'] = other
    
    return rawdata

##############################################
# correct_masked_pixels
def op_correct_masked_pixels(rawdata, stripes=64, sides=1024, verbose=False):
    return 0
    #---------------------------------------------------
    data = rawdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
        
    icorner = rawdata['INTERF']['corner']
    inaxis  = rawdata['INTERF']['naxis']
    intf    = rawdata['INTERF']['data']

    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        corny = []
        if icorner[0] == corner[0] :
            print("Found a corner in the OTHER data that matches the INTERF data")
            print('key:', key, 'corner:', corner, 'icorner:', icorner)
            print('size of intf:', np.shape(intf))
            print('size of other:', np.shape(rawdata['OTHER'][key]['data']))
            corny.append(np.mean(rawdata['OTHER'][key]['data'], axis=(0,1)))
        
        cornavg = np.mean(corny,axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(cornavg, label='Average corner data')
        plt.plot( np.mean(intf, axis=(0,1)), label='Average intf data')
        plt.plot( np.mean(intf-cornavg[None,None,:], axis=(0,1)), label='Average intf data')
        plt.show()

##############################################
# Subtract sky
def op_subtract_sky(rawdata, skydata, skydataOO=None, verbose=False, plot=False):
    #---------------------------------------------------
    data = rawdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
    data['hdr'][f'HIERARCH PROC{count} SKY'] = os.path.basename(skydata['filename'])
    
    if verbose: print('Interf shape:', np.shape(rawdata['INTERF']['data']), 'Sky shape:', np.shape(skydata['INTERF']['data']))
    # Compute robust average of sky
    skydata['INTERF']['data'] = np.mean(skydata['INTERF']['data'], axis=0)#stats.trim_mean(skydata['INTERF']['data'], 0.05, axis=0)
    for key in skydata['PHOT']:
        skydata['PHOT'][key]['data'] = np.mean(skydata['PHOT'][key]['data'], axis=0) #stats.trim_mean(skydata['PHOT'][key]['data'], 0.05, axis=0)
    for key in skydata['OTHER']:
        skydata['OTHER'][key]['data'] = np.mean(skydata['OTHER'][key]['data'], axis=0)
            
    rawdata['INTERF']['datamean'] = np.mean(rawdata['INTERF']['data'], axis=0) #stats.trim_mean(rawdata['INTERF']['data'], 0.05, axis=0)
    for key in rawdata['PHOT']:
        rawdata['PHOT'][key]['datamean'] = np.mean(rawdata['PHOT'][key]['data'], axis=0) #stats.trim_mean(rawdata['PHOT'][key]['data'], 0.05, axis=0)
    
    if plot:
        fringes = rawdata['INTERF']['datamean']
        med = np.nanmedian(fringes.flatten())
        mad = np.nanmedian(np.abs(fringes.flatten() - med))
        vmin = med - 3 * mad
        vmax = med + 10 * mad
        
        fig, axs = plt.subplots(3, 5, figsize=(8, 8), sharey=True)
        axs = axs.flatten()
        
        axs[0].imshow(rawdata['INTERF']['datamean'], vmin=vmin, vmax=vmax)
        axs[0].set_title(f'Intf shape')
        
        for i, key in enumerate(rawdata['PHOT']):
            axs[i+1].imshow(rawdata['PHOT'][key]['datamean'], label=key, vmin=vmin, vmax=vmax)
            axs[i+1].set_title(f'Phot {i+1} shape')
            
        axs[0+5].imshow(skydata['INTERF']['data'], vmin=vmin, vmax=vmax)
        axs[0+5].set_title(f'Intf shape')
        
        for i, key in enumerate(skydata['PHOT']):
            axs[i+1+5].imshow(skydata['PHOT'][key]['data'], label=key, vmin=vmin, vmax=vmax)
            axs[i+1+5].set_title(f'Phot {i+1} shape')
        
    # Subtract sky from rawdata
    rawdata['INTERF']['data'] -= skydata['INTERF']['data'][None,...]
    for key in rawdata['PHOT']:
        rawdata['PHOT'][key]['data'] -= skydata['PHOT'][key]['data'][None,...]
    for key in rawdata['OTHER']:
        rawdata['OTHER'][key]['data'] -= skydata['OTHER'][key]['data'][None,...]
        
    if plot:
        newfringes = rawdata['INTERF']['data']
        med = np.nanmedian(newfringes.flatten())
        mad = np.nanmedian(np.abs(newfringes.flatten() - med))
        vmin = med - 3 * mad
        vmax = med + 10 * mad
        
        rms = np.std(rawdata['INTERF']['data'], axis=0)
        axs[0+10].imshow(stats.trim_mean(rawdata['INTERF']['data'], 0.05, axis=0), vmin=vmin, vmax=vmax)
        axs[0+10].set_title(f'Intf shape')
        
        for i, key in enumerate(rawdata['PHOT']):
            axs[i+11].imshow(stats.trim_mean(rawdata['PHOT'][key]['data'], 0.05, axis=0), label=key, vmin=vmin, vmax=vmax)
            axs[i+11].set_title(f'Phot {i+1} shape')
        plt.show()
        
    return rawdata
    
##############################################
# Display the structure of a FITS file
def op_print_fits_structure(fits_data):
    for hdu in fits_data:
        print(f'-------\nHDU: {hdu.name}')
        #print(f'Header:\n{hdu.header}')
        if hdu.data is not None:
            if hdu.is_image:
                print('This is an image.')
            else:
                print('This is a table.')
            #print(f'Name: {hdu.hdr['EXTNAME']}')
            if isinstance(hdu.data, np.recarray):
                print(f'Columns: {hdu.data.dtype.names}')
            print(f'Data shape: {hdu.data.shape}')
    
##############################################
# Display the structure of a FITS file
def op_print_fits_header(fits_data, hdu=0):
    print('printing header')
    hdr = fits_data[hdu].header
    for key in hdr:
        if any(key):
           print(key, "\t", hdr[key])

##############################################
# Display the structure of a FITS file
def op_match_keys(fits_data, keys, values, hdu=0):
    print('printing header')
    hdr = fits_data[hdu].header
    match = False
    for iky, ky in enumerate(keys):
        if fnmatch.fnmatch(hdr, ky):
            if hdr[ky] == keys[iky]:
                match = True

##############################################
# Load raw data
def op_load_rawdata(filename, verbose=True):
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
        
    fh      =  fits.open(filename)
    data    = {'hdr': fh[0].header}
    data['hdr']['PROCSOFT'] = PIPELINE_NAME + ' ' + PIPELINE_VERSION
    
    data['filename'] = filename
    nframes = len(fh['IMAGING_DATA'].data)
    nexp    = len(fh['IMAGING_DATA'].data)//6
    #print('nexp:',nexp)
    nreg    = len(fh['IMAGING_DETECTOR'].data)
    
    # Set instrument here
    data['instrument'] = data['hdr']['INSTRUME']+'_'+ data['hdr']['ESO DET CHIP NAME']
    
    data['PHOT']   = {}
    data['INTERF'] = {}
    data['OTHER']  = {}
    
    data['ARRAY_DESCRIPTION'] = fh['ARRAY_DESCRIPTION'].data
    data['ARRAY_GEOMETRY']    = fh['ARRAY_GEOMETRY'].data
    data['OPTICAL_TRAIN']     = fh['OPTICAL_TRAIN'].data
    
    # Fill in the OI_ARRAY table
    data['OI_ARRAY'] = {}
    data['OI_ARRAY']['TEL_NAME']  = data['ARRAY_GEOMETRY']['TEL_NAME']
    data['OI_ARRAY']['STA_NAME']  = data['ARRAY_GEOMETRY']['STA_NAME']
    data['OI_ARRAY']['STA_INDEX'] = data['ARRAY_GEOMETRY']['STA_INDEX']
    data['OI_ARRAY']['DIAMETER']  = data['ARRAY_GEOMETRY']['DIAMETER']
    data['OI_ARRAY']['STAXYZ']    = data['ARRAY_GEOMETRY']['STAXYZ']
    
    # Load the local OPD table that contains the modulation information
    localopd = []
    mjds     = []
    tartyp   = []
    exptime  = []
    target   = []

    for i in np.arange(nframes):
        localopd.append(fh['IMAGING_DATA'].data[i]['LOCALOPD'].astype(float))
        mjds.append(    fh['IMAGING_DATA'].data[i]['TIME'].astype(float))
        exptime.append( fh['IMAGING_DATA'].data[i]['EXPTIME'])
        target.append(  fh['IMAGING_DATA'].data[i]['TARGET'])
        tartyp.append(  fh['IMAGING_DATA'].data[i]['TARTYP'])
    localopd = np.array(localopd) 
    mjds = np.array(mjds)
    tartyp = np.array(tartyp)
    exptime = np.array(exptime)
    target = np.array(target)

    #print('Localopd:', localopd)
    #print('MJDs:', mjds)
    #print('TARTYP:', tartyp)
    
    for j in np.arange(nreg):
        corner = fh['IMAGING_DETECTOR'].data[j]['CORNER']
        naxis  = fh['IMAGING_DETECTOR'].data[j]['NAXIS']
        #print(f'Processing region {j}:{fh['IMAGING_DETECTOR'].data['REGNAME'][j]}')
        datarray = []
        for i in np.arange(nframes):
            datarray.append(fh['IMAGING_DATA'].data[i][j+1].astype(float))
        if fnmatch.fnmatch(fh['IMAGING_DETECTOR'].data['REGNAME'][j], 'INTERF*'):
            data['INTERF']['data']     = datarray
            data['INTERF']['corner']   = corner
            data['INTERF']['naxis']    = naxis
            data['INTERF']['localopd'] = localopd
            data['INTERF']['mjds']     = mjds
            data['INTERF']['tartyp']   = tartyp
            data['INTERF']['exptime']  = exptime
            data['INTERF']['target']   = target

        elif fnmatch.fnmatch(fh['IMAGING_DETECTOR'].data['REGNAME'][j], 'PHOT*'):
            key = fh['IMAGING_DETECTOR'].data['REGNAME'][j]
            data['PHOT'][key]={}
            data['PHOT'][key]['data']   = datarray
            data['PHOT'][key]['corner'] = corner
            data['PHOT'][key]['naxis']  = naxis
        else:
            key = fh['IMAGING_DETECTOR'].data['REGNAME'][j].strip('\x001')
            data['OTHER'][key]={}
            data['OTHER'][key]['data']   = datarray
            data['OTHER'][key]['corner'] = corner
            data['OTHER'][key]['naxis']  = naxis
    fh.close()
    
    # Add a processing step to the header
    #---------------------------------------------------
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
    data['hdr'][f'HIERARCH PROC{count} FILE'] = os.path.basename(filename)

    return data

##############################################
# Load and calibrate raw data
def op_loadAndCal_rawdata(sciencefile, skyfile, bpm, ffm, instrument=op_MATISSE_L, verbose=False, plot=False):
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    
    # Load the star and sky data
    tardata  = op_load_rawdata(sciencefile)
    starname = tardata['hdr']['OBJECT']
    if verbose: print(starname)

    bcd1 = tardata['hdr']['HIERARCH ESO INS BCD1 ID']
    bcd2 = tardata['hdr']['HIERARCH ESO INS BCD2 ID']
    det  = tardata['hdr']['HIERARCH ESO DET CHIP NAME']
    if verbose: print('BCD1:', bcd1, 'BCD2:', bcd2, 'DET:', det)

    # Load the sky data
    skydata  = op_load_rawdata(skyfile)
    
    # Load the calibration data
    bpm = op_load_bpm(bpm)
    ffm = op_load_ffm(ffm)

    stardata = op_apply_ffm(tardata, ffm, verbose=verbose)
    stardata = op_apply_bpm(stardata, bpm, verbose=verbose)
    
    skydata = op_apply_ffm(skydata, ffm, verbose=verbose)
    skydata = op_apply_bpm(skydata, bpm, verbose=verbose)

    # Subtract the sky from the star data
    bdata = op_subtract_sky(stardata, skydata)
    
    #bdata = op_correct_masked_pixels(bdata, stripes=64, sides=1024, verbose=verbose)
    
    bdata['OI_BASELINES'] = {}
    bdata['OI_BASELINES']['TARGET_ID'] = bdata['INTERF']['target']
    bdata['OI_BASELINES']['TARGET']    = bdata['hdr']['ESO OBS TARG NAME']
    bdata['OI_BASELINES']['TIME']      = 86400 * (bdata['INTERF']['mjds'] - bdata['INTERF']['mjds'][0])
    bdata['OI_BASELINES']['MJD']       = bdata['INTERF']['mjds']
    bdata['OI_BASELINES']['INT_TIME']  = bdata['INTERF']['exptime']
    bdata['OI_BASELINES']['STA_INDEX'] = bdata['OI_ARRAY']['STA_INDEX'][instrument['scrB']]
    # print('Scrambling of baselines:', bdata['OI_ARRAY']['STA_INDEX'][instrument['scrB']])
    return bdata


##############################################
# Get location
def _op_get_location(hdr,plot, verbose=True):
    """
    DESCRIPTION
        Fill the loc dictionnary with :
            name, ntel , lon, lat, elev

    PARAMETERS
        - hdr : header of a .fits file of an OBS
    """
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    
    loc = dict()
    
    try:

        loc['name'] = hdr['ORIGIN']
        loc['ntel'] = hdr['HIERARCH ESO ISS CONF NTEL']
        loc['lon']  = hdr['HIERARCH ESO ISS GEOLON']
        loc['lat']  = hdr['HIERARCH ESO ISS GEOLAT']
        loc['elev'] = hdr['HIERARCH ESO ISS GEOELEV'] # should be close to 2635m
        loc['pos']  = _op_positionsTelescope(hdr,loc,plot)
    except:
        loc['name'] = "ESO, Cerro Paranal"
        loc['ntel'] = hdr['HIERARCH ESO ISS CONF NTEL']
        loc['lon']  = -70.40479659
        loc['lat']  = -24.62794830
        loc['elev'] = 2635
        loc['pos']  = _op_positionsTelescope(hdr,loc,plot)
        print('No location found in header, using default location: Paranal')
        
    return loc




##############################################
# Compute uvw
def _op_calculate_uvw(data,bvect,loc, verbose=False):
    """
    DESCRIPTION
        Corrects uvw coordinates using base vector of observation bvect, and update
        data fields with results.

    PARAMETERS
        - data : uvw coord dictionnary to be corrected.
        - bvect: base vector of observation."""
        
    data['ha'] = data['lst'] - data['ra']
    degr       = 180 / np.pi
    hour       = degr / 15
    Bnorm      =  lin.norm(bvect)
    
    #Baseline in alt-az coordinates
    Balt = np.arcsin(bvect[2] / (Bnorm)) * degr
    Baz  = np.arctan2(bvect[0], bvect[1]) * degr
    
    #Baseline vector in equatorial coordinates
    Bdec = np.arcsin(np.sin(Balt/degr) * np.sin(loc['lat']/degr) + np.cos(Balt/degr) * np.cos(loc['lat']/degr) * np.cos(Baz/degr)) * degr
    yBha = np.sin(Balt/degr) * np.cos(loc['lat']/degr) - np.cos(Balt/degr) * np.cos(Baz/degr) * np.sin(loc['lat']/degr)
    zBha = -1. * np.cos(Balt/degr) * np.sin(Baz/degr)
    Bha  = (np.arctan2(zBha,yBha) * hour + 24) % 24
    
    # baseline vector in the equatorial cartesian frame
    Lx = - (-Bnorm * np.cos(Bdec/degr) * np.cos(Bha/hour))
    Ly = - Bnorm * np.cos(Bdec/degr) * np.sin(Bha/hour)
    Lz = Bnorm * np.sin(Bdec/degr)
    
    # projection of the baseline vector on the u,v,w frame 
    data['u'] = (np.sin(data['ha']/hour) * Lx + np.cos(data['ha']/hour) * Ly)
    data['v'] = - np.sin(data['dec']/degr) * np.cos(data['ha']/hour) * Lx +\
                np.sin(data['dec']/degr) * np.sin(data['ha']/hour) * Ly +\
                np.cos(data['dec']/degr) * Lz
    data['w'] = (np.cos(data['dec']/degr) * np.cos(data['ha']/hour) * Lx  -\
                 np.cos(data['dec']/degr) * np.sin(data['ha']/hour) * Ly +\
                 np.sin(data['dec']/degr) * Lz);

    data['theta'] = np.arctan2(data['u'], data['v']) * degr
    data['base']  = np.sqrt(data['u']**2 + data['v']**2)
    data['delay'] = - data['w']
    
    return data




##############################################
# Get position of the telescopes
def _op_positionsTelescope(hdr,loc,plot, verbose=True):
    """
    DESCRIPTION
        get the positions of all the telescopes of the interferometer 

    PARAMETERS
        - hdr       : input header.
        - loc       : location of the interferometer
        - plot      : boolean
    """
    
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Positions from https://www.eso.org/observing/etc/doc/viscalc/vltistations.html
    
    positions = [['ID' ,   'P'   ,   'Q'   ,   'E'  ,   'N'  , 'ALT', 'D','C'],
                 ['A0' , -32.001 , -48.013 , -14.642, -55.812, 0.0  , 1.8, 1],
                 ['A1' , -32.001 , -64.021 ,  -9.434, -70.949, 0.0  , 1.8, 2],
                 ['B0' , -23.991 , -48.019 ,  -7.065, -53.212, 0.0  , 1.8, 3],
                 ['B1' , -23.991 , -64.011 ,  -1.863, -68.334, 0.0  , 1.8, 4],
                 ['B2' , -23.991 , -72.011 ,   0.739, -75.899, 0.0  , 1.8, 5],
                 ['B3' , -23.991 , -80.029 ,   3.348, -83.481, 0.0  , 1.8, 6],
                 ['B4' , -23.991 , -88.013 ,   5.945, -91.030, 0.0  , 1.8, 7],
                 ['B5' , -23.991 , -96.012 ,   8.547, -98.594, 0.0  , 1.8, 8],
                 ['C0' , -16.002 , -48.013 ,   0.487, -50.607, 0.0  , 1.8, 9],
                 ['C1' , -16.002 , -64.011 ,   5.691, -65.735, 0.0  , 1.8, 10],
                 ['C2' , -16.002 , -72.019 ,   8.296, -73.307, 0.0  , 1.8, 11],
                 ['C3' , -16.002 , -80.010 ,  10.896, -80.864, 0.0  , 1.8, 12],
                 ['D0' ,   0.010 , -48.012 ,  15.628, -45.397, 0.0  , 1.8, 13],
                 ['D1' ,   0.010 , -80.015 ,  26.039, -75.660, 0.0  , 1.8, 14],
                 ['D2' ,   0.010 , -96.012 ,  31.243, -90.787, 0.0  , 1.8, 15],
                 ['E0' ,  16.011 , -48.016 ,  30.760, -40.196, 0.0  , 1.8, 16],
                 ['G0' ,  32.017 , -48.0172,  45.896, -34.990, 0.0  , 1.8, 17],
                 ['G1' ,  32.020 ,-112.010 ,  66.716, -95.501, 0.0  , 1.8, 18],
                 ['G2' ,  31.995 , -24.003 ,  38.063, -12.289, 0.0  , 1.8, 19],
                 ['H0' ,  64.015 , -48.007 ,  76.150, -24.572, 0.0  , 1.8, 20],
                 ['I1' ,  72.001 , -87.997 ,  96.711, -59.789, 0.0  , 1.8, 21],
                 ['J1' ,  88.016 , -71.992 , 106.648, -39.444, 0.0  , 1.8, 22],
                 ['J2' ,  88.016 , -96.005 , 114.460, -62.151, 0.0  , 1.8, 23],
                 ['J3' ,  88.016 ,   7.996 ,  80.628,  36.193, 0.0  , 1.8, 24],
                 ['J4' ,  88.016 ,  23.993 ,  75.424,  51.320, 0.0  , 1.8, 25],
                 ['J5' ,  88.016 ,  47.987 ,  67.618,  74.009, 0.0  , 1.8, 26],
                 ['J6' ,  88.016 ,  71.990 ,  59.810,  96.706, 0.0  , 1.8, 27],
                 ['K0' ,  96.002 , -48.006 , 106.397, -14.165, 0.0  , 1.8, 28],
                 ['L0' , 104.021 , -47.998 , 113.977, -11.549, 0.0  , 1.8, 29],
                 ['M0' , 112.013 , -48.000 , 121.535,  -8.951, 0.0  , 1.8, 30],
                 ['U1' , -16.000 , -16.000 ,  -9.925, -20.335, 8.504, 8  , 31],
                 ['U2' ,  24.000 ,  24.000 ,  14.887,  30.502, 8.504, 8  , 32],
                 ['U3' ,  64.0013,  47.9725,  44.915,  66.183, 8.504, 8  , 33],
                 ['U4' , 112.000 ,   8.000 , 103.306,  43.999, 8.504, 8  , 34],
                 ['LAB',  52.000 , -40.000 ,  60    , -20    , 0.0]]
    tel_labels = []
    tel_position = []
    
    #Correct positons with positions from the header
    for i in range(loc['ntel']):
        keys = [f"HIERARCH ESO ISS CONF STATION{i+1}",
                f"HIERARCH ESO ISS CONF T{i+1}X",
                f"HIERARCH ESO ISS CONF T{i+1}Y",
                f"HIERARCH ESO ISS CONF T{i+1}Z"]
        for pos in positions:
            if pos[0] == hdr[keys[0]]:
                try :
                    pos[3] = -hdr[keys[1]]
                    pos[4] = -hdr[keys[2]]
                    pos[5] = hdr[keys[3]]
                except:
                    print('Oups...')
                tel_labels.append(pos[0])
                tel_position.append((pos[3],pos[4]))
                break
        
        if hdr[keys[0]] not in [row[0] for row in positions]:
                positions.insert(1,[hdr[keys[0]],0,0,hdr[keys[1]],hdr[keys[2]],0])
                tel_labels.append(hdr[keys[0]])
                tel_position.append((hdr[keys[1]],hdr[keys[2]]))   
    ############## PLOT MAP OF VLTI ##############
    if plot:
        labels = [pos[0] for pos in positions[1:]] 
        easts = [pos[3] for pos in positions[1:]]   
        norths= [pos[4] for pos in positions[1:]]    
        
        nu=-18.984 #degre

        plt.figure(figsize=(10, 10))
        
        ntel = loc['ntel']
        nbases = int(ntel * (ntel-1)/2)
        if ntel == 4:
            colors = COLORS6D
        else:
            colors = [plt.cm.get_cmap('hsv', nbases)(i) for i in range(nbases)]


        baseline_pos = []
        for i in range(len(tel_position)-1):
            for j in range(i+1,len(tel_position)):
                baseline_pos.append(([[tel_position[i][0],tel_position[j][0]], [tel_position[i][1],tel_position[j][1]]],tel_labels[i]+'-'+tel_labels[j]))
                    
        
        baseline_pos = sorted(baseline_pos,key=lambda p: np.sqrt((p[0][0][1]-p[0][0][0])**2+(p[0][1][1]-p[0][1][0])**2))
        for i in range(len(baseline_pos)):
            plt.plot(baseline_pos[i][0][0], baseline_pos[i][0][1], color=colors[i], linewidth=2,zorder=0)
        for east, north, label in zip(easts, norths, labels):
            is_ut = label.startswith('U')
            is_lab = label.startswith('LA')
            diameter = 8 if is_ut else 1.8
            radius = diameter/2 #Pour fit au mieux les noms
            fc_color = 'red' if label in tel_labels else 'white'
            fontsize = 13 if is_ut or is_lab else 11
            
            if is_lab:
                # Size of the lab 
                rect_width = 14  # (m)
                rect_height = 9  # (m)
                
                
                rect_origin = np.array([east,north]) - np.array([rect_width/2-1.9, rect_height/2+2.1])
                
                rectangle = plt.Rectangle(
                    rect_origin, rect_width, rect_height,
                    angle=-nu,
                    facecolor='lightgray', edgecolor='black', lw=1, zorder=0
                )   
                plt.gca().add_patch(rectangle)
                plt.text(east, north, label, fontsize=fontsize, ha='center', va='center', color='black', zorder=2,rotation=-nu)
           
            elif is_ut:
                circle = plt.Circle((east, north), radius=radius, facecolor=fc_color, edgecolor='black', lw=0.5, zorder=1)
                plt.gca().add_patch(circle)
       
                plt.text(east, north, label, fontsize=fontsize, ha='center', va='center', color='black', zorder=2,rotation=-nu)
                
            else:
            
                 circle = plt.Circle((east, north), radius=radius, facecolor=fc_color, edgecolor='black', lw=0.5, zorder=1)
                 plt.gca().add_patch(circle)
        
                 plt.text(east-1, north+2.5, label, fontsize=fontsize, ha='center', va='center', color='black', zorder=2,rotation=-nu,)
            
            
        
        
        plt.title(f"Map of the {loc['name']} interferometer (coordinate E/N)")
        
        plt.xlabel("Longitude (E)")
        plt.ylabel("Latitude (N)")
        plt.axis('equal')
        
        # plt.xlim((-55,155))
        # plt.ylim((-105,105))
        plt.grid(True)
        # plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        # plt.gca().yaxis.set_major_locator(MultipleLocator(10))
        
        #indicate north
        plt.annotate('N',xy=(0.05, 0.95), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
        plt.arrow(0.05, 0.87, 0, 0.05, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
        
        #indicate east
        plt.annotate('E',xy=(0.13, 0.88), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
        plt.arrow(0.05, 0.87, 0.05, 0, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
        plt.tight_layout()
        plt.show()
   
    return positions
            



##############################################
# Get baseline vector
def _op_get_baseVect(station1,station2,loc,delay = dict(), verbose=True):
    """
    DESCRIPTION
        Computes the vectored baseline between station1 and station2

    PARAMETERS
        - station1   : name of the first station
        - station2   : name of the second station
        - loc        : location of the interferometer
    """
    
    for tel in loc['pos']:
        if tel[0] == station1: 
            B1 = [tel[3], tel[4], tel[5]]
            BP1 = tel[1] 
            BQ1 = tel[2]
        if tel[0] == station2:
            B2  = [tel[3], tel[4], tel[5]]
            BP2 = tel[1] 
            BQ2 = tel[2]
        if tel[0] == "LAB":
            #Compute delay lines
            lab      = [tel[3], tel[4], tel[5]]
            A1L      = abs(BQ1-tel[2]) + abs(BP1-tel[1])
            A2L      = abs(BQ2-tel[2]) + abs(BP2-tel[1])
            fixDelay = A2L-A1L

            # print(f'DL {station1} =', A1L, f', DL {station2} =' ,A2L, f' fixDelay {station2}-{station1} =', fixDelay)
            delay[station1] = A1L
            delay[station2] = A2L
    return np.array([B2[0]-B1[0],B2[1]-B1[1],B2[2]-B1[2]]),delay

            

##############################################
# Get all baselines vector
def _op_compute_baseVect(hdr,loc,instrument=op_MATISSE_L, verbose=True):
    """
    DESCRIPTION
        Computes all the vectored baseline

    PARAMETERS
        - hdr    : input header
        - loc        : location of the interferometer
    """

         
    delay = dict()
    stations=[]
    for i in np.arange(instrument['ntel']):
        stations.append(hdr[f"HIERARCH ESO ISS CONF STATION{i+1}"])
    base_allvect=[]
    basescr  = instrument['scrB']
    # Get all baselines
    for itel1,itel2 in basescr:

        base, delay = _op_get_baseVect(stations[itel1], stations[itel2], loc,delay)
        base_allvect.append(base)

    
    return base_allvect,delay
    

##############################################
# Compute uv coordinates
def op_compute_uv(cfdata, plot ,instrument=op_MATISSE_L, verbose=True):
    """
    DESCRIPTION
        Computes UV coordinates with a fits file given as input. 
        You can then compare them with the Baselines and angles in the Header 

    PARAMETERS
        - cfdata      : ldata of the correlated fluxes
        - plot        : boolean
    """
    
    #---------------------------------------------------
    data = cfdata
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    # Add a processing step to the header
    count = 1
    while f'HIERARCH PROC{count}' in data['hdr']: count += 1
    data['hdr'][f'HIERARCH PROC{count}'] = inspect.currentframe().f_code.co_name
    #---------------------------------------------------
        
    #initialisation
    stardata = dict()  
    uCoord = []
    vCoord = []
    wCoord = []
    
    # get location and star data from the header   
    hdr = cfdata['hdr']
    loc = _op_get_location(hdr, plot)
    date = hdr["DATE-OBS"]
    stardata['date'] = date[0].split('T')[0]
    
    ra_j2000 = hdr['RA']
    dec_j2000 = hdr['DEC']
    
    try:
        pmra  = hdr['PMRA']
        pmdec = hdr['PMDEC']
        
        mjd = hdr['MJD-OBS']
        date_obs = Time(mjd,format = 'mjd').decimalyear 
        j2000 = Time('J2000').decimalyear
        delta_t = date_obs -  j2000 # in decimal year
        
        
        ra_obs  = ra_j2000 + pmra * delta_t 
        dec_obs = dec_j2000 + pmdec * delta_t
        
    except KeyError:
        
        coord = SkyCoord(ra_j2000, dec_j2000, unit=(u.deg, u.deg),frame='fk5', equinox = Time("J2000"))
        Simbad.add_votable_fields('otype', 'pmra', 'pmdec')
        try:
            result = Simbad.query_region(coord)
        except:
            result = None
    
        if result is not None:
            # Filter object : Star '*'
            mask = [('*' in otype) for otype in result['otype']]
            result = result[mask]
            ra_list  = result['ra']
            dec_list = result['dec']
            coord_query = SkyCoord(ra_list,dec_list, unit=(u.deg, u.deg),frame='fk5', equinox = Time("J2000"))
            separations = coord_query.separation(coord)
            amin = np.argmin(separations)
            
            pmra  = result['pmra'][amin] / (1000 * 3600) #mas/yr to deg/yr
            pmdec = result['pmdec'][amin] / (1000 * 3600) #mas/yr to deg/yr
            
            mjd = hdr['MJD-OBS']
            date_obs = Time(mjd,format = 'mjd').decimalyear 
            j2000 = Time('J2000').decimalyear
            delta_t = date_obs -  j2000 # in decimal year
            
            
            ra_obs  = ra_j2000 + pmra * delta_t 
            dec_obs = dec_j2000 + pmdec * delta_t
        
        else:
            print("les coordonnees RA-DEC sont au J2000")
            ra_obs  = ra_j2000 
            dec_obs = dec_j2000
    
    stardata['ra']   = ra_obs/15
    stardata['dec']  = dec_obs
    
    # Get the vector of all the baseline and compute uv Coords
    B, delay =_op_compute_baseVect(hdr, loc , instrument = instrument)
    ndit = hdr['HIERARCH ESO DET NDIT']
    dit  = hdr['HIERARCH ESO DET SEQ1 DIT']
    
    LST=[(hdr['LST']+i*dit/ndit)/3600 for i in range(ndit)]
    for i,bvect in enumerate(B):
        uco = [];vco=[];wco=[]
        for lst in LST :
            stardata['lst']=lst
            uvw=deepcopy(stardata)
            uvw=_op_calculate_uvw(uvw,bvect,loc)
            uco.append(uvw['u'])
            vco.append(uvw['v'])
            wco.append(uvw['w'])
        uCoord.append(uco)
        vCoord.append(vco)
        wCoord.append(wco)
    
    cfdata['OI_BASELINES']['UCOORD'] = uCoord
    cfdata['OI_BASELINES']['VCOORD'] = vCoord
    cfdata['OI_BASELINES']['WCOORD'] = wCoord
    # cfdata['OI_BASELINES']['DL'] = sorted(list(delay.values()))
    
    return cfdata  

##############################################
# Compute uv_coverage
def op_uv_coverage(uCoord,vCoord,cfdata,instrument = op_MATISSE_L, verbose=True):
    """
    DESCRIPTION
        Computes the UV coverage with all the fits files of an OBS given as input.
        You can then compare it with Aspro data 

    PARAMETERS
        - files     : list of input file
        - cfdata    : datas of the correlated fluxes
        
    """
    if verbose: print(f"executing --> {inspect.currentframe().f_code.co_name}")
    
    wlen     = cfdata['OI_WAVELENGTH']['EFF_WAVE']
    wlen_ref = cfdata['OI_WAVELENGTH']['EFF_REF']

    ######################### PLOT ################################
    
    plt.figure(figsize=(10, 10))
    ax=plt.gca()
    ax2 = plt.gca().twiny()
    ax3 = plt.gca().twinx()
    
    nObs   = len(uCoord)
    ntel = cfdata['hdr']['HIERARCH ESO ISS CONF NTEL']
    nBase  = int(ntel * (ntel-1)/2)
    nFrame = int(len(uCoord[0])//nBase)
    if ntel == 4:
        colors = COLORS6D
    else :
        colors = [plt.cm.get_cmap('hsv', nBase)(i) for i in range(nBase)]

    for iBase in range(nBase):
        u = []
        v = []
        for iObs in range(nObs):
            
            for iframe in range(0,nFrame):
                
               u.append(uCoord[iObs][iBase][iframe])
               v.append(vCoord[iObs][iBase][iframe])
            
        u = np.array(u)
        v = np.array(v)
        
        plt.scatter(u, v, color=colors[iBase],hatch='x',lw=0.5)
        plt.scatter(-u, -v, color=colors[iBase],hatch='x',lw=0.5)
        plt.plot(u, v, color=colors[iBase], lw=2)
        plt.plot(-u, -v, color=colors[iBase], lw=2)
        
        # spatial frequencies 
        for i in range(0,len(u),len(u)//5):
            plt.plot(u[i]/wlen*wlen_ref, v[i]/wlen*wlen_ref, color=colors[iBase], lw=2)
            plt.plot(-u[i]/wlen*wlen_ref, -v[i]/wlen*wlen_ref, color=colors[iBase], lw=2)
            

    plt.title("uv-coverage map", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("U (Mλ - 10⁶ cycles/rad)", fontsize=14, fontweight='bold')
    ax.set_ylabel("V (Mλ - 10⁶ cycles/rad)", fontsize=14, fontweight='bold')
    
    ulim = np.max(np.abs(uCoord))*1.15
    vlim = np.max(np.abs(vCoord))*1.15

    lim = np.max([ulim,vlim])

    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("U (m) ", fontsize=14, fontweight='bold')
    ax2.invert_xaxis()
    
    ax3.set_ylim(-lim, lim)
    ax3.set_ylabel("V (m) ", fontsize=14, fontweight='bold')
    
    ax.set_xlim(ax2.get_xlim()[0] / (wlen_ref * 1e6) , ax2.get_xlim()[1] / (wlen_ref * 1e6) )
    ax.set_ylim(ax3.get_ylim()[0] / (wlen_ref * 1e6) , ax3.get_ylim()[1] / (wlen_ref * 1e6) )
    # Twin axes for meters
    
    
    
    #limit anf grid
    # plt.xlim((-125,125))
    # plt.ylim((-125,125))
    
    if instrument['ntel'] <= 4 :
        ax2.xaxis.set_major_locator(MultipleLocator(20))
        ax3.yaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, which='both', linestyle='--', color='gray', linewidth=0.7)
    #ax3.grid(True, which='both', linestyle='--', color='gray', linewidth=0.7)
    # Highlight x=0 and y=0 lines
    ax2.axhline(0, color='black', linewidth=1.5)  # horizontal line at y=0
    ax2.axvline(0, color='black', linewidth=1.5)  # vertical line at x=0
    
    #indicate north
    plt.annotate('N',xy=(0.10, 0.95), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
    plt.arrow(0.10, 0.87, 0, 0.05, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
    
    #indicate east
    plt.annotate('E',xy=(0.03, 0.88), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
    plt.arrow(0.10, 0.87, -0.05, 0, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
    
    # LABELS
    hdr = cfdata['hdr']
    basescr  = instrument['scrB']
    telname = []
    labels = []

    for i in np.arange(instrument['ntel']): #REORDER PGOT
         telname.append(hdr[f"HIERARCH ESO ISS CONF T{i+1}NAME"])

    # Get all baselines
    for itel1,itel2 in basescr: #REORDER BASELINE
        labels.append(telname[itel1]+'-'+telname[itel2])
    
    
    handles = [plt.Line2D([], [], color=colors[i], label=labels[i]) for i in range(len(labels))]
    if instrument['ntel'] <=6:
        plt.legend(handles=handles, loc='lower right')
    
    plt.tight_layout()
    plt.show()
        
    return cfdata

#def _op_sortout_base