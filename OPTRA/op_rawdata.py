#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Cosmetics for raw data
# Author: fmillour
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
from op_instruments import *
from matplotlib.ticker import MultipleLocator
from copy import deepcopy


##############################################
# Function to interpolate bad pixels
# 
# 
def op_interpolate_bad_pixels(data, bad_pixel_map, verbose=False):
    if verbose:
        print('Interpolating bad pixels...')
    # Apply a median filter to the data
    filtered_data = median_filter(data, size=3)
    #plt.imshow(filtered_data, cmap='gray')
    # Replace bad pixels with the median filtered values
    data[bad_pixel_map] = filtered_data[bad_pixel_map]
    return data, filtered_data

##############################################
# Load bad pixel map
def op_load_bpm(filename, verbose=False):
    if verbose:
        print('Loading bad pixel map...')
    fh = fits.open(filename)
    bpm = fh[0].data.astype(bool)
    fh.close()
    return bpm

##############################################
# Apply bpm
def op_apply_bpm(rawdata, bpmap, verbose=False):
    if verbose:
        print('Applying bad pixel map...')
    # Subtract sky from rawdata
    
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf   = rawdata['INTERF']['data']
    nframe = np.shape(intf)[0]
    if verbose:
        print(f'Processing {nframe} frames')
    wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    fdata = []
    for i in range(nframe):
        intf[i],filtdata = op_interpolate_bad_pixels(intf[i], wbpm)
        fdata.append(filtdata)
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot   = rawdata['PHOT'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        fdata = []
        for i in range(nframe):
            phot[i],filtdata = op_interpolate_bad_pixels(phot[i], wbpm)
            fdata.append(filtdata)
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wbpm = bpmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        fdata = []
        for i in range(nframe):
            other[i],filtdata = op_interpolate_bad_pixels(other[i], wbpm)
            fdata.append(filtdata)
        rawdata['OTHER'][key]['data'] = other
    return rawdata


##############################################
# Load flat field map
def op_load_ffm(filename, verbose=False):
    if verbose:
        print('Loading flat field...')
    fh = fits.open(filename)
    ffm = fh[0].data.astype(float)
    fh.close()
    return ffm

##############################################
# Apply flat field map
def op_apply_ffm(rawdata, ffmap, verbose=False):
    if verbose:
        print('Applying flat field map...')
    # Subtract sky from rawdata
    
    corner = rawdata['INTERF']['corner']
    naxis  = rawdata['INTERF']['naxis']
    intf   = rawdata['INTERF']['data']
    
    nframe = np.shape(intf)[0]
    if verbose:
        print(f'Processing {nframe} frames')
        
    wffm = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
    # Interpolate bad pixels in each frame
    
    for i in range(nframe):
        intf[i] /= wffm
    rawdata['INTERF']['data'] = intf
    
    for key in rawdata['PHOT']:
        corner = rawdata['PHOT'][key]['corner']
        naxis  = rawdata['PHOT'][key]['naxis']
        phot   = rawdata['PHOT'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # Interpolate bad pixels in each frame
        
        for i in range(nframe):
            phot[i] /= wffm
        rawdata['PHOT'][key]['data'] = phot
        
    for key in rawdata['OTHER']:
        corner = rawdata['OTHER'][key]['corner']
        naxis  = rawdata['OTHER'][key]['naxis']
        other  = rawdata['OTHER'][key]['data']
        wffm   = ffmap[corner[1]-1:corner[1]+naxis[1]-1, corner[0]-1:corner[0]+naxis[0]-1]
        # correct flat field
        for i in range(nframe):
            other[i] /= wffm
        rawdata['OTHER'][key]['data'] = other
    return rawdata

##############################################
# Subtract sky
def op_subtract_sky(rawdata, skydata, verbose=False):
    if verbose:
        print('Subtracting sky...')
    # Compute robust average of sky
    skydata['INTERF']['data'] = stats.trim_mean(skydata['INTERF']['data'], 0.05, axis=0)
    #print(skydata['INTERF']['data'].shape)
    #print(len(skydata['PHOT']))
    for key in skydata['PHOT']:
        skydata['PHOT'][key]['data'] = stats.trim_mean(skydata['PHOT'][key]['data'], 0.05, axis=0)
        
    # Subtract sky from rawdata
    rawdata['INTERF']['data'] -= skydata['INTERF']['data']
    for key in skydata['PHOT']:
        rawdata['PHOT'][key]['data'] -= skydata['PHOT'][key]['data']
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
        #print('\n')
    
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
    print(f'Loading raw data {filename}...')
    fh      =  fits.open(filename)
    data    = {'hdr': fh[0].header}
    nframes = len(fh['IMAGING_DATA'].data)
    nexp    = len(fh['IMAGING_DATA'].data)//6
    #print('nexp:',nexp)
    nreg    = len(fh['IMAGING_DETECTOR'].data)
    
    data['PHOT'] = {}
    data['INTERF'] = {}
    data['OTHER'] = {}
    
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
    mjds = []
    tartyp = []
    exptime = []
    target = []

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
    return data

##############################################
# Load and calibrate raw data
def op_loadAndCal_rawdata(sciencefile, skyfile, bpm, ffm, instrument=op_MATISSE_L, verbose=False, plot=False):
    print('loading and calibrating raw data...')
    # Load the star and sky data
    tardata  = op_load_rawdata(sciencefile)
    starname = tardata['hdr']['OBJECT']
    if verbose:
        print(starname)

    bcd1 = tardata['hdr']['HIERARCH ESO INS BCD1 ID']
    bcd2 = tardata['hdr']['HIERARCH ESO INS BCD2 ID']
    det  = tardata['hdr']['HIERARCH ESO DET CHIP NAME']
    if verbose:
        print('BCD1:', bcd1, 'BCD2:', bcd2, 'DET:', det)

    # Load the sky data
    skydata  = op_load_rawdata(skyfile)

    # Subtract the sky from the star data
    stardata = op_subtract_sky(tardata, skydata)
    
    # Load the calibration data
    bpm = op_load_bpm(bpm)
    ffm = op_load_ffm(ffm)

    fdata = op_apply_ffm(stardata, ffm, verbose=verbose)
    bdata = op_apply_bpm(fdata, bpm, verbose=verbose)
    
    bdata['OI_BASELINES'] = {}
    bdata['OI_BASELINES']['TARGET_ID'] = bdata['INTERF']['target']
    bdata['OI_BASELINES']['TARGET']    = bdata['hdr']['ESO OBS TARG NAME']
    bdata['OI_BASELINES']['TIME']      = 86400 * (bdata['INTERF']['mjds'] - bdata['INTERF']['mjds'][0])
    bdata['OI_BASELINES']['MJD']       = bdata['INTERF']['mjds']
    bdata['OI_BASELINES']['INT_TIME']  = bdata['INTERF']['exptime']
    bdata['OI_BASELINES']['STA_INDEX'] = bdata['OI_ARRAY']['STA_INDEX'][instrument['scrB']]
    #print('Scrambling of baselines:', bdata['OI_ARRAY']['STA_INDEX'][instrument['scrB']])
    return bdata


##############################################
# Get location
def _op_get_location(hdr,plot):
    """
    DESCRIPTION
        Fill the loc dictionnary with :
            name, ntel , lon, lat, elev

    PARAMETERS
        - hdr : header of a .fits file of an OBS
    """
    
    loc=dict()
    
    try:

        loc['name'] = hdr['ORIGIN']
        loc['ntel'] = hdr['HIERARCH ESO ISS CONF NTEL']
        loc['lon'] = hdr['HIERARCH ESO ISS GEOLON']
        loc['lat'] = hdr['HIERARCH ESO ISS GEOLAT']
        loc['elev'] = 2635
        loc['pos']=_op_positionsTelescope(hdr,loc,plot)
    except:
        loc['name'] = "ESO, Cerro Paranal"
        loc['ntel'] = 4
        loc['lon'] = -70.40479659
        loc['lat'] = -24.62794830
        loc['elev'] = 2635
        loc['pos']=_op_positionsTelescope(hdr,loc,plot)
        
    return loc




##############################################
# Compute uvw
def _op_calculate_uvw(data,bvect,loc):
    """
    DESCRIPTION
        Corrects uvw coordinates using base vector of observation bvect, and update
        data fields with results.

    PARAMETERS
        - data : uvw coord dictionnary to be corrected.
        - bvect: base vector of observation."""
        
    data['ha'] = data['lst'] - data['ra']
    degr = 180 / np.pi
    hour = degr / 15
    Bnorm=  lin.norm(bvect)
    
    #Baseline in alt-az coordinates
    Balt = np.arcsin(bvect[2] / (Bnorm)) * degr
    Baz = np.arctan2(bvect[0], bvect[1]) * degr
    
    #Baseline vecor in equatorial coordinates
    Bdec = np.arcsin(np.sin(Balt/degr) * np.sin(loc['lat']/degr) + np.cos(Balt/degr) * np.cos(loc['lat']/degr) * np.cos(Baz/degr)) * degr
    yBha = np.sin(Balt/degr) * np.cos(loc['lat']/degr) - np.cos(Balt/degr) * np.cos(Baz/degr) * np.sin(loc['lat']/degr)
    zBha = -1. * np.cos(Balt/degr) * np.sin(Baz/degr)
    Bha = (np.arctan2(zBha,yBha) * hour + 24) % 24
    
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

    data['theta'] =np.arctan2(data['u'], data['v']) * degr
    data['base'] = np.sqrt(data['u']**2 + data['v']**2)
    data['delay'] = - data['w']
    
    return data




##############################################
# Get position of the telescopes
def _op_positionsTelescope(hdr,loc,plot):
    """
    DESCRIPTION
        get the positions of all the telescopes of the interferometer 

    PARAMETERS
        - hdr       : input header.
        - loc       : location of the interferometer
        - plot      : boolean
    """
    
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
                pos[3] = -hdr[keys[1]]
                pos[4] = -hdr[keys[2]]
                pos[5] = hdr[keys[3]]
                tel_labels.append(pos[0])
                tel_position.append((pos[3],pos[4]))
                
    ############## PLOT ##############
    if plot:
        labels = [pos[0] for pos in positions[1:]] 
        easts = [pos[3] for pos in positions[1:]]   
        norths= [pos[4] for pos in positions[1:]]    

        
        plt.figure(figsize=(10, 10))
        
        colors = ['#1f77b4','#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf']

        
        for east, north, label in zip(easts, norths, labels):
            
            fc_color = 'red' if label in tel_labels else 'white'
            fontsize = 15 if label.startswith('U') else 10
            plt.text(east, north, label,fontsize=fontsize,ha='center', va='center', color='black',
                                 bbox=dict(boxstyle="circle,pad=0.3", fc=fc_color, ec="black", lw=0.5))
            
        for i in range(len(tel_position)-1):
            for j in range(i+1,len(tel_position)):
                plt.plot([tel_position[i][0],tel_position[j][0]], [tel_position[i][1],tel_position[j][1]], color=colors[(i * len(tel_position) + j) % len(colors)], linewidth=2)
                
        plt.title(f"Map of the {loc['name']} interferometer (coordinate E/N)")
        plt.xlabel("Longitude (E)")
        plt.ylabel("Latitude (N)")
        plt.xlim((-60,160))
        plt.ylim((-110,110))
        plt.grid(True)
        
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
def _op_get_baseVect(station1,station2,loc):
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
            BP1, BQ1 = tel[1], tel[2]
        if tel[0] == station2:
            B2 = [tel[3], tel[4], tel[5]]
            BP2, BQ2 = tel[1], tel[2]
        if tel[0] == "LAB":
            #Compute delay lines
            lab = [tel[3], tel[4], tel[5]]
            DL1 = abs(BQ1-tel[1]) + abs(BP1-tel[1])
            DL2 = abs(BQ2-tel[1]) + abs(BP2-tel[1])
            fixDelay = DL2-DL1
            
    return np.array([B2[0]-B1[0],B2[1]-B1[1],B2[2]-B1[2]])
            

##############################################
# Get all baselines vector
def _op_compute_baseVect(hdr,loc):
    """
    DESCRIPTION
        Computes all the vectored baseline

    PARAMETERS
        - hdr    : input header
        - loc        : location of the interferometer
    """
        
     #initialisation  
    keys = ["HIERARCH ESO ISS CONF STATION1",
            "HIERARCH ESO ISS CONF STATION2",
            "HIERARCH ESO ISS CONF STATION3",
            "HIERARCH ESO ISS CONF STATION4"]
    stations=[]
    base_allvect=[]
    for key in keys:
        stations.append(hdr[key])  
    ntel = len(stations)
    
    # Get all baseline
    for itel1 in range(ntel-1):
        for itel2 in range(itel1+1,ntel):
            base_allvect.append(_op_get_baseVect(stations[itel1], stations[itel2], loc))
    return base_allvect
    

##############################################
# Compute uv coordinates
def op_compute_uv(file, plot):
    """
    DESCRIPTION
        Computes UV coordinates with a fits file given as input. 
        You can then compare them with the Baselines and angles in the Header 

    PARAMETERS
        - file    : input file
        - plot    : boolean
    """
    
    #initialisation
    stardata = dict()
    hdr = None   
    data=[]
    with fits.open(file) as fh:
        hdr = fh[0].header
        
    # get location and star data from the header   
    loc=_op_get_location(hdr, plot)
    date=hdr["DATE-OBS"]
    stardata['date']=date[0].split('T')[0]
    stardata['ra']=hdr['RA']/15
    stardata['dec']=hdr['DEC']
    stardata['lst']=hdr['LST']/3600
    
    # Get the vector of all the baseline and compute uv Coords
    B=_op_compute_baseVect(hdr, loc)
    for bvect in B:
        uvw=deepcopy(stardata)
        uvw=_op_calculate_uvw(uvw,bvect,loc)
        data.append(uvw)
    return data  

##############################################
# Compute uv_coverage
def op_uv_coverage(files,plot):
    """
    DESCRIPTION
        Computes the UV coverage with all the fits files of an OBS given as input.
        You can then compare it with Aspro data 

    PARAMETERS
        - files    : list of input file
        - plot    : boolean
    """
    
    data = []
    for file in files:
        data.append(op_compute_uv(file,False))
        
        
    if plot:
        plt.figure(figsize=(10, 10))
        colors = ['red','blue', 'lightgreen', 'orange', 'purple', 'cyan']
        nObs=len(data)
        nBase=len(data[0])
        for iBase in range(nBase):
            u=[]
            v=[]
            for iObs in range(nObs):
               u.append(data[iObs][iBase]['u'])
               v.append(data[iObs][iBase]['v'])
    
            u=np.array(u)
            v=np.array(v)
            
            plt.plot(u, v, color=colors[iBase], linewidth=2)
            plt.plot(-u, -v, color=colors[iBase], linewidth=2)
            
        plt.title("uv-coverage map")
        plt.xlabel("u (m)")
        plt.ylabel("v (m)")
        
        #limit anf grid
        plt.xlim((-125,125))
        plt.ylim((-125,125))
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(10))
        plt.grid(True, which='major', linestyle='--', color='gray', linewidth=0.7)

        # Highlight x=0 and y=0 lines
        plt.axhline(0, color='black', linewidth=1.5)  # horizontal line at y=0
        plt.axvline(0, color='black', linewidth=1.5)  # vertical line at x=0
        
        #indicate north
        plt.annotate('N',xy=(0.10, 0.95), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
        plt.arrow(0.10, 0.87, 0, 0.05, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
        
        #indicate east
        plt.annotate('E',xy=(0.03, 0.88), xycoords='axes fraction',fontsize=14, fontweight='bold', ha='center')
        plt.arrow(0.10, 0.87, -0.05, 0, transform=plt.gca().transAxes,width=0.002, head_width=0.01, head_length=0.02,fc='k', ec='k', zorder=5)
        
        labels = ['UT1-UT2', 'UT1-UT3', 'UT1-UT4', 'UT2-UT3', 'UT2-UT4', 'UT3-UT4']
        handles = [plt.Line2D([], [], color=colors[i], label=labels[i]) for i in range(len(colors))]
        plt.legend(handles=handles, loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
    return data