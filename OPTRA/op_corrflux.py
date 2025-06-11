#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Correlated flux computation
# Author: fmillour, jscigliuto, mhoulle
# Date: 01/07/2024
# Project: OPTRA
#
# This module contains functions for computing correlated flux, apodizing data,
# computing FFT of interferograms, extracting correlated flux, demodulating 
# MATISSE fringes, sorting out beams and peaks, and computing the air refractive 
# index.
#
################################################################################

from os import error
from astropy.io import fits
from scipy import *
import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
from op_instruments import *
from scipy import interpolate
import math
from scipy.optimize import minimize
from scipy.ndimage   import uniform_filter1d
import astropy.constants as cst 
from astropy.convolution import convolve, Gaussian1DKernel
from op_parameters import *

##############################################
# Apodization function
def op_apodize(data, verbose=True,plot=False, frac=0.85):
    if verbose:
        print('Apodizing data...')
    # Apply an apodizing window to the data
    nframes = np.shape(data['INTERF']['data'])[0]
    nwlen   = np.shape(data['INTERF']['data'])[1]
    nreg    = len(data['PHOT'])
    
    if verbose:
        print('computing apodizing window...')
    intf  = stats.trim_mean(data['INTERF']['data'],axis=0, proportiontocut=0.05)
    n     = np.shape(intf)[1]
    argmx = np.argmax(stats.trim_mean(intf, axis=0, proportiontocut=0.05))
    #argmx=n//2
    argmx=272
    if verbose:
        print('argmx', argmx)
    if verbose:
        print('n', n)
    dx = int((n - 2*np.abs(n//2 - argmx))*frac)
    if verbose:
        print('dx', dx)
    if n%2 == 1:
        dx+=1
    wn    = signal.get_window(('kaiser', 14), dx)
    zewin = np.zeros(n)
    zewin[argmx-dx//2:argmx+dx//2] = wn
    centered_win_intf = zewin
    
    if plot:
        # Plot the Hanning window used for apodization
        plt.figure()
        plt.plot(centered_win_intf)
        plt.title('Window for Apodization')
        plt.xlabel('Pixel Index')
        plt.ylabel('Window Value')
        plt.grid(True)
        plt.show()
    
    data['INTERF']['center'] = argmx
    for i in np.arange(nframes):
        data['INTERF']['data'][i] *= centered_win_intf
        
    for key in data['PHOT']:
        pht   = stats.trim_mean(data['PHOT'][key]['data'],axis=0, proportiontocut=0.05)
        argmx = np.argmax(stats.trim_mean(pht, axis=0, proportiontocut=0.05))
        n     = np.shape(pht)[1]
        dx    = int(n - 2*np.abs(n/2 - argmx))
        if verbose:
            print('dx', dx)
        wn    = signal.get_window(('kaiser', 14), dx)
        zewin = np.zeros(n)
        zewin[argmx-dx//2:argmx+dx//2] = wn
        centered_win_pht               = zewin
        data['PHOT'][key]['center']    = argmx
        for i in np.arange(nframes):
            data['PHOT'][key]['data'][i] *= centered_win_pht
            
    return data

##############################################
# compute the FFT of interferograms
def op_calc_fft(data, verbose=True):
    if verbose:
        print('Computing FFT of interferograms...')
    intf = data['INTERF']['data']
    nframe = np.shape(intf)[0] # Frames dimension
    nwlen  = np.shape(intf)[1] # Wavelength dimension
    npix   = np.shape(intf)[2] # Pixels dimension (fringed Airy pattern)
    
    # Compute FFT 1D of intf along the pixels axis
    fft_intf = np.fft.fft(intf, axis=2)
    # Compute the phasor corresponding to the shift of the center of the window
    center_shift = data['INTERF']['center']
    phasor       = np.exp(2j * np.pi * center_shift * (np.arange(npix)) / npix)
    fft_intf    *= phasor[None,None,:]
    
    fft_intf_magnitude = np.abs(fft_intf)   # Compute the magnitude of the FFT
    dsp_intf     = fft_intf_magnitude**2    # Compute the power spectrum
    sum_dsp_intf = np.sum(dsp_intf, axis=0) # Sum the power spectrum over all frames
    sdi_resh     = np.fft.fftshift(sum_dsp_intf, axes=1) # Shift the zero frequency to the center
    freqs        = np.fft.fftfreq(npix)     # Compute the frequencies
    if verbose:
        print('Shape of sum_dsp_intf:', sum_dsp_intf.shape)
    
    data['FFT'] = {'data': fft_intf, 'magnitude': fft_intf_magnitude, 'dsp': dsp_intf, 'sum_dsp': sum_dsp_intf, 'sdi': sdi_resh, 'freqs': freqs}
    return data

##############################################
# compute the wavelength 
def op_get_wlen(shift_map, rawdata, verbose=True, plot=False):
    if verbose:
        print('Computing wavelength map...')
    # Compute wavelength map from shift map
    fh        = fits.open(shift_map)
    shift_map = fh['SHIFT_MAP'].data
    disp      = shift_map['DISP']
    if verbose:
        print('shape of disp:', np.shape(disp))
        print('disp:', disp)
        print('disp0:', disp[0])
        print('disp1:', disp[1])
    
    corner = rawdata['INTERF']['corner']
    px     = corner[1]+np.arange(np.shape(rawdata['INTERF']['data'])[1])
    wlen   = disp[0][0] + disp[0][1]*px + disp[0][2]*px**2 + disp[0][3]*px**3 + disp[0][4]*px**4
    wlen2   = disp[1][0] + disp[1][1]*px + disp[1][2]*px**2 + disp[0][3]*px**3 + disp[0][4]*px**4
    
    if verbose:
        print('Corner',corner)
        print('px',px)
        print('Disp',disp)
    
        print('Wavelength',wlen)
    
    # FIXME: Set the bandwidth of wavelength table
    band = np.diff(wlen)*5
    
    if verbose:
        print(disp)
        print(wlen)
        
    if plot:
        plt.figure()
        plt.plot(wlen)
        plt.show()
    
    rawdata['OI_WAVELENGTH'] = {}
    rawdata['OI_WAVELENGTH']['EFF_WAVE'] = wlen * 1e-6 # Convert to meters
    rawdata['OI_WAVELENGTH']['EFF_BAND'] = band * 1e-6 # Convert to meters
    rawdata['OI_WAVELENGTH']['EFF_REF']  = rawdata['hdr']['HIERARCH ESO SEQ DIL WL0'] * 1e-6

    return wlen * 1e-6

##############################################
# get the peaks position
def op_get_peaks_position(fftdata, instrument=op_MATISSE_L, verbose=True):
    wlen = fftdata['OI_WAVELENGTH']['EFF_WAVE'] *1e6 # Convert to micrometers
    
    print('Instrument',instrument)
    if instrument['name'] == 'MATISSE_L':
        peaks =       np.arange(7)
        interfringe = instrument['interfringe']
        peakswd =     instrument['peakwd']
        pkwds   = np.ones_like(peaks) * peakswd
        peak    = peaks[:,None] * interfringe / wlen[None,:]
        peakwd  = pkwds[:,None] * interfringe / wlen[None,:]
    else:
        error('Instrument not recognized')
    
    if verbose:
        print('Shape of peak:', np.shape(peak))
        print('Peak:', peak)
    return peak, peakwd

##############################################
# extract the correlated flux
def op_extract_CF(fftdata, peaks, peakswd, verbose=True, plot=False):
    if verbose:
        print('Extracting correlated flux...')
    bck    = np.copy(fftdata['FFT']['data'])
    nfreq  = np.shape(bck)[2]
    nfreq2 = int(nfreq/2)
    bck    = bck[:,:,0:nfreq2]
    if verbose:
        print('Shape of bck:', np.shape(bck))
        print('nfreq:', nfreq)
    ifreq  = np.arange(nfreq2)
    npeaks = np.shape(peaks)[0]
    ibase  = np.arange(npeaks)
    FT = []
    CF = []
    NIZ = []
    for i in ibase:
        fti    = fftdata['FFT']['data'][:,:,0:nfreq2]
        zone   = np.logical_and(ifreq[None,:] >= peaks[i,:][:,None]-peakswd[i,:][:,None]/2, ifreq[None,:] <= peaks[i,:][:,None]+peakswd[i,:][:,None]/2)
        weight = np.exp(-0.5 * ((ifreq[None,:] - peaks[i,:][:,None]) / (2*peakswd[i,:][:,None] / 2.355))**2)
        
        NIZ.append(np.sum(weight * zone, axis=1))
        FT.append(fti*zone)
        CF.append(np.sum(weight * fti * zone, axis=2))
        bck *= (1-zone)
    FT  = np.array(FT)
    CF  = np.array(CF)
    NIZ = np.array(NIZ)
    
    if verbose:
        print('Shape of FT:', np.shape(FT))
        print('Shape of CF:', np.shape(CF))
        print('shape of NIZ:', np.shape(NIZ))
        
    if plot:
        fig, axes = plt.subplots(1, 8, figsize=(16, 8))
        fig.tight_layout()
        FTavg = np.mean(FT, axis=1)
        for i, ax in enumerate(axes.flat):
            if i < npeaks:
                epsilon = 0#1e-6
                #ax.imshow(np.log(np.abs(FTavg[i,...])+epsilon), cmap='gray')
                ax.imshow((np.abs(FTavg[i,...])+epsilon), cmap='gray')
                ax.set_title('Peak {}'.format(i))
            else:
                #ax.imshow(np.log(np.abs(bck[0,...]+epsilon)), cmap='gray')
                ax.imshow((np.abs(bck[0,...]+epsilon)), cmap='gray')
                ax.set_title('Background')
        plt.show()
    
    if verbose:
        print('Shape of FT:', np.shape(FT))
    fftdata['CF'] = {'data': FT, 'CF': CF, 'CF_nbpx': NIZ, 'bckg': bck}
    return fftdata

##############################################
# demodulate MATISSE fringes
def op_demodulate(CFdata, cfin='CF', verbose=False, plot=False):
    wlen = CFdata['OI_WAVELENGTH']['EFF_WAVE'] * 1e6  # Convert to micrometers
    
    if verbose:
        print('Demodulating correlated flux...')
    npeaks  = np.shape(CFdata['CF']['data'])[0]
    nframes = np.shape(CFdata['CF']['data'])[1]
    nwlen   = np.shape(CFdata['CF']['data'])[2]
    npix    = np.shape(CFdata['CF']['data'])[3]
    if verbose:
        print('npeaks:', npeaks)
        print('nframes:', nframes)
        print('nwlen:', nwlen)
        print('npix:', npix)
        
    localopd = CFdata['INTERF']['localopd']
    if verbose == 2:
        print('Shape of localopd:', np.shape(localopd))
    
    ntel = 4
    # Compute baseline OPD from local OPD
    teli = (3,1,2,2,1,1)
    telj = (4,2,3,4,3,4)
    localopdij = []
    ibase=0
    for itel in np.arange(ntel-1):
        for jtel in np.arange(ntel - itel - 1) + itel + 1:
            if verbose:
                print('ij:',itel,jtel)
            loij = 1 * (localopd[:,teli[ibase]-1] - localopd[:,telj[ibase]-1])
            localopdij.append(loij)
            if verbose:
                print('ij:',itel,jtel, 'localopdij:', loij)
            ibase+=1
    localopdij = np.array(localopdij)
    # Compute the phasor from localopd
    phasor = np.exp(2j * np.pi * localopdij[:,:,None] / wlen[None,None,:] )
    
    CFdata['CF']['mod_phasor']         = phasor
    CFdata['CF']['CF_demod']           = np.zeros_like(CFdata['CF'][cfin])
    CFdata['CF']['CF_demod'][0,...]    = CFdata['CF'][cfin][0,...]
    CFdata['CF']['CF_demod'][1:,...]   = CFdata['CF'][cfin][1:,...]   * np.conjugate(phasor)
    CFdata['CF']['data_demod']         = np.zeros_like(CFdata['CF']['data'])
    CFdata['CF']['data_demod'][0,...]  = CFdata['CF']['data'][0,...]
    CFdata['CF']['data_demod'][1:,...] = CFdata['CF']['data'][1:,...] * np.conjugate(phasor[...,None])
    
    if verbose: 
        print('wlen:', wlen)
    
    if plot:
        iframe = 0
        colors = COLORS7D
        plt.figure(4)
        for i in np.arange(6):
            plt.plot(np.angle(phasor[i,iframe,:]),color=colors[i])
        plt.show()
    
    if verbose:
        print('Shape of phasor:', np.shape(phasor))
        print('Shape of CF:', np.shape(CFdata['CF']['CF']))
        
    return CFdata


##############################################
# Function to compute correlated flux
def op_get_corrflux(bdata, shiftfile, bindata=True, verbose=False, plot=False, corr_opd=True ):
    if verbose:
        print('Computing correlated flux...')
        
    #########################################################
    # Get the wavelength
    wlen = op_get_wlen(shiftfile, bdata, verbose=verbose)
    if verbose:
        print(wlen)
        
        
    #########################################################
    # Apodization
    bdata = op_apodize(bdata, verbose=verbose, plot=plot)
        
    if plot:
        # Compute the average of intf after apodization
        avg_intf = np.mean(bdata['INTERF']['data'], axis=0)
        vmn = 1e-9
        vmx = np.max(avg_intf)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.tight_layout()
        split_images = np.array_split(avg_intf, 3, axis=0)
        for ax, img in zip(axes, split_images):
            ax.imshow(img, cmap='gray')
        plt.title('Average of intf after Apodization')
        plt.show()
        
    #########################################################
    #compute fft
    bdata = op_calc_fft(bdata)

    if plot:
        # Compute the average of intf after apodization
        sum_dsp = np.log(bdata['FFT']['sum_dsp'])
        #fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig, axes = plt.subplots(1, 1, figsize=(6, 10))
        fig.tight_layout()
        
        axes.imshow(sum_dsp, cmap='gray')
        '''
        split_images = np.array_split(sum_dsp, 3, axis=0)
        for ax, img in zip(axes, split_images):
            ax.imshow(img, cmap='gray')
            '''
        plt.title('Sum of dsp after Apodization')

        #plt.show()

    


    #########################################################
    # Get the peaks position
    peaks, peakswd = op_get_peaks_position(bdata, verbose=verbose)
    if verbose:
        print(wlen)

    colors = COLORS7D

    if plot:
        for i in range(np.shape(peaks)[0]):
            plt.plot(peaks[i,:], np.arange(np.shape(peaks)[1]),color=colors[i])
            plt.plot(peaks[i,:]+peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
            plt.plot(peaks[i,:]-peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
        plt.show()
        
    # peaks = op_sortout_peaks(peaks,bdata)
    
    # for i in range(np.shape(peaks)[0]):
    #     plt.plot(peaks[i,:], np.arange(np.shape(peaks)[1]),color=colors[i])
    #     plt.plot(peaks[i,:]+peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
    #     plt.plot(peaks[i,:]-peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
    # plt.show()

    #########################################################
    # Extract the correlated flux
    bdata = op_extract_CF(bdata, peaks, peakswd, verbose=verbose)
    if verbose:
        print(wlen)
    
        
    if plot:
        iframe = 0
        iwlen = 70
        plt.figure(1)
        plt.imshow(np.angle(bdata['CF']['data'][1,iframe,:,:]), cmap='gray')
        plt.title('2D phase map for one peak')

        plt.figure(2)
        for i in np.arange(7):
            plt.plot(np.angle(bdata['CF']['data'][i,iframe,iwlen,:]),color=colors[i])
        plt.title('Cut of the phase of CF Data')

        plt.figure(3)
        for i in np.arange(7):
            plt.plot(np.abs(bdata['CF']['data'][i,iframe,iwlen,:]),color=colors[i])
        plt.plot(np.abs(bdata['CF']['bckg'][iframe,iwlen,:]))
        plt.yscale('log')
        plt.title('Modulus of Complex Values for CF Data and Background')
        
    #########################################################
    # Demodulate MATISSE fringes
    bdata = op_demodulate(bdata, verbose=verbose, plot=plot)
    
    #########################################################
    # Reorder baselines
    bdata, cfdata_reordered = op_reorder_baselines(bdata)
    
    if corr_opd == True:
        #########################################################
        # Get the air refractive index
        Temp, Pres, hum, dPath    = op_get_amb_conditions(bdata)
        if verbose:
            print('Temp:', Temp, 'Pres:', Pres, 'hum:', hum)
        n_air = op_air_index(wlen, Temp, Pres, hum, N_CO2=423, bands='all')
        if verbose:
            print('n_air:', n_air)
            
        #########################################################
        # Correct the phase for air path
        bdata, phase_layer_air_slope = op_corr_n_air(bdata, n_air, dPath, cfin='CF_reord', wlmin=3.3e-6, wlmax=3.7e-6, verbose=verbose, plot=plot)
        
        #########################################################
        # Get the piston    
        bdata, OPD_list = op_get_piston_fft(bdata, cfin='CF_chr_phase_corr', verbose=verbose, plot=plot)
        
        #########################################################
        # Correct the piston
        bdata = op_corr_piston(bdata, cfin='CF_chr_phase_corr', verbose=verbose, plot=plot)
        nextone = 'CF_piston_corr'
    else:
        nextone = 'CF_demod'

    #########################################################
    # Correct for residual phase
    totvis = np.sum(bdata['CF'][nextone],axis=-1)
    cvis = bdata['CF'][nextone] * np.exp(-1j * np.angle(totvis[...,None]))
    bdata['CF']['CF_piston_corr2'] = cvis

    #########################################################
    # Get the error and the snr
    bdata=op_get_error_vis(bdata,cfin='CF_piston_corr2',plot=plot)
    
    #########################################################
    # Get the semi theorical snr
    op_snr_theory(bdata,plot=plot)

        #########################################################
        # Bin the data
        
    if bindata: 
        bdata = op_bin_data(bdata, cfin='CF_piston_corr2', verbose=verbose, plot=plot)
    else:
        bdata = op_bin_data(bdata, cfin='CF_reord', verbose=verbose, plot=plot)
        
    
    
   
    #return cfdem
    #return cfreord
    return bdata

##############################################
# Function to sort out peaks
# The combiner entrance MATISSE pupil looks
# like that in L band (BCD out)
#  S1       S2          S3    S4
#      2        3          1
#  _        _           _     _
# / \      / \         / \   / \
# \_/      \_/         \_/   \_/
# and like this in N band (BCD out)
#  S4   S3           S2       S1
#     1        3          2
#  _     _           _        _
# / \   / \         / \      / \
# \_/   \_/         \_/      \_/
#
# The BCD inverts S1 <-> S2 and S3 <-> S4
# 
def op_sortout_peaks(peaksin, bdata, instrument=op_MATISSE_L, verbose=False):
    if verbose:
        print('Sorting out peaks...')
        
    bcd1 = bdata['hdr']['HIERARCH ESO INS BCD1 ID']
    bcd2 = bdata['hdr']['HIERARCH ESO INS BCD2 ID']
    det  = bdata['hdr']['HIERARCH ESO DET CHIP NAME']
    
    if verbose:
        print('BCD1:', bcd1)
        print('BCD2:', bcd2)
    
    tel= bdata['OPTICAL_TRAIN']['INDEX']
    ntel    = len(tel)
    nbases  = ntel*(ntel-1)//2
    telname = bdata['OPTICAL_TRAIN']['TEL_NAME']
    if verbose:
        print('Telescope names:', telname)
    DL_number = bdata['OPTICAL_TRAIN']['VALUE1']
    IP_number = bdata['OPTICAL_TRAIN']['VALUE2']
    
    #########################################
    # Internal beams scrambling
    basescr  = instrument['scrB']
    band     = instrument['band']
    
    
        
    base_unscr = [l for _,l in sorted(zip(basescr[::],peaksin[1:]))]
    # coding = (1,3,6,7)
   
    # ibase=0
    # peakscr = np.zeros(nbases)
    peaks_unscr = peaksin[0]
    peaks_unscr= np.append([peaksin[0]],base_unscr,axis = 0)
    
    # for i in range(nbases):
    #     if peakunscr[i] > 0:
    #         peaksin['CF']['CF'][i+1,...] = peaksin['CF']['CF'][int(peakunscr[i]),...]
    #     else:
    #         peaksin['CF']['CF'][i+1,...] = -peaksin['CF']['CF'][int(-peakunscr[i]),...]
            
            
    # for itel in range(ntel - 1):
    #     for jtel in range(itel + 1, ntel):
    #         telnamei = telname[tel[bcdscr[beamscr[itel]-1]-1]-1]
    #         telnamej = telname[tel[bcdscr[beamscr[jtel]-1]-1]-1]
    #         teli = coding[bcdscr[beamscr[itel]-1]-1]
    #         telj = coding[bcdscr[beamscr[jtel]-1]-1]
    #         lng = telj-teli
    #         if verbose:
    #             print("base",ibase+1, "telescopes", itel, "and", jtel, "tel1",telnamei,"tel2",telnamej, "peak",lng)
    #         peakscr[ibase] = -lng
    #         ibase+=1
            
    # # peakunscr_tmp = np.arange(nbases)
    # peakunscr = np.zeros(nbases)
    # for i in np.arange(nbases):
    #     for j in np.arange(nbases):
    #         if int(np.abs(peakscr[j])-1) == i:
    #             peakunscr[i] = np.sign(peakscr[j]) * (j+1)
    # if verbose:
    #     print("peakscr",peakscr)
    #     print("peakunscr",peakunscr)
    # for i in np.arange(nbases):
    #     if peakunscr[i] > 0:
    #         peaksin['CF']['CF'][i+1,...] = peaksin['CF']['CF'][int(peakunscr[i]),...]
    #     else:
    #         peaksin['CF']['CF'][i+1,...] = -peaksin['CF']['CF'][int(-peakunscr[i]),...]
            
            
    return peaks_unscr

##############################################
# reorder baselines 
def op_reorder_baselines(data, cfin='CF_demod'):

    cfdata = data['CF'][cfin]
    # print(cfdata.shape) #base/frame/wl 7/6/1560
    n_frames = np.shape(cfdata)[1]
    n_exp = n_frames // 6
    bcd1 = data['hdr']['ESO INS BCD1 NAME']
    bcd2 = data['hdr']['ESO INS BCD2 NAME']

    # Reorder the data in cfdem
    bcd_base_reorder = {'OUT-OUT': [0, 1, 2, 3, 4, 5, 6],  # phot+OUT-OUT, 
                         'OUT-IN': [0, 1, 2, 5, 6, 3, 4],  # phot+OUT-IN,
                         'IN-OUT': [0, 1, 2, 4, 3, 6, 5],  # phot+IN-OUT,
                          'IN-IN': [0, 1, 2, 6, 5, 4, 3]}  # phot+IN-IN
    bcd_sign = {'OUT-OUT': [1, 1, 1, 1, 1, 1, 1],   # phot+OUT-OUT,
                 'OUT-IN': [1, 1, -1, 1, 1, 1, 1],  # phot+OUT-IN,
                 'IN-OUT': [1, -1, 1, 1, 1, 1, 1],  # phot+IN-OUT,
                  'IN-IN': [1, -1, -1, 1, 1, 1, 1]} # phot+IN-IN
    
    cfdata_reordered = np.zeros_like(cfdata)

    # Get the BCD positions
    bcd = f'{bcd1}-{bcd2}'
    # Reorder the baselines
    for i, new_idx in enumerate(bcd_base_reorder[bcd]):
        cfdata_reordered[i] = cfdata[new_idx]

    # Apply sign phase change
    cfdata_amp = np.abs(cfdata_reordered)
    cfdata_phase = np.angle(cfdata_reordered)
    for i, sign in enumerate(bcd_sign[bcd]):
        cfdata_reordered[i] = cfdata_amp[i] * np.exp(1j * sign * cfdata_phase[i])

    data['CF']['CF_reord'] = cfdata_reordered

    return data, cfdata_reordered

##############################################
# Function to get the ambient conditions
def op_get_amb_conditions(data, verbose=True):

    # Relative humidity
    humidity = data['hdr']['ESO ISS AMBI RHUM'] / 100

    # Temperature (°C)
    T1 = data['hdr']['ESO ISS TEMP TUN1'] # Temp in tunnel west [C], duct A.         
    T2 = data['hdr']['ESO ISS TEMP TUN2'] # Temp in tunnel centre [C], M16 west.     
    T3 = data['hdr']['ESO ISS TEMP TUN3'] # Temp in tunnel centre [C], M16 east.     
    T4 = data['hdr']['ESO ISS TEMP TUN4'] # Temp in tunnel east [C], duct L.         
    temperature = (T1+T2+T3+T4)/4

    #Others temperatures (°C)
    Tlab = data['hdr']['ESO ISS TEMP LAB1'] # Temp in lab.                             
    Tamb = data['hdr']['ESO ISS AMBI TEMP'] # Observatory ambient temperature [C].      
    Tamb_dew = data['hdr']['ESO ISS AMBI TEMPDEW'] # Observatory ambient dew temperature [C]
    Tsky = data['hdr']['ESO ISS AMBI IRSKY TEMP'] # Temperature of the IR sky, from rad
    
    # MATISSE-specific temps
    try:
        Tins1 = data['hdr']['ESO INS SENS34 VAL'] #  [C] Ambient temperature Value.           
        Tins2 = data['hdr']['ESO INS SENS147 VAL'] # [C] Ambient temperature Value.      
        Tins3 = data['hdr']['ESO INS SENS240 VAL'] # [C] SOS Ambient temperature Value.     
        Tins4 = data['hdr']['ESO INS SENS242 VAL'] # [C] CPL Ambient temperature Value.      
        Tins5 = data['hdr']['ESO INS SENS244 VAL'] # [C] CPN Ambient temperature Value.
        Tins6 = data['hdr']['ESO INS SENS246 VAL'] # [C] ARC Ambient temperature Value. 
        Tins7 = data['hdr']['ESO INS SENS258 VAL'] # [C] Cabinet ambient temperature Value.
        print("Loaded MATISSE-specific temperatures")
    except:
        print("WARNING: No MATISSE-specific temperatures found!")
    
    # temperature = Tlab

    # keys=[]
    # for key in data['hdr']:
    #     keys.append(key)
    # print(keys)
    if verbose:
        print('Temperature:', temperature)

    # Pressure (hPa)
    pressure = data['hdr']['ESO ISS AMBI PRES']
    if verbose:
        print('Pressure:', pressure)

    # Get the MJDs
    mjds   = data['INTERF']['mjds'] 
    if verbose:
        print('MJDs:',np.shape(mjds))
    tartyp = data['INTERF']['tartyp']
    mjds_valid = mjds#[tartyp == 'T']
    n_valid_frames = np.sum(tartyp == 'T')

    ## Get the path lengths
    static_lengths = np.zeros(4)
    start_OPLs     = np.zeros(4)
    end_OPLs       = np.zeros(4)
    OPLs           = np.zeros(4,n_valid_frames)
    dPaths         = np.zeros(6,n_valid_frames)
    for i_tel in range(4):
        # Static paths (m)
        static_lengths[i_tel] = data['hdr'][f'ESO ISS CONF A{i_tel+1}L']
        # Optical path lengths (m)
        start_OPLs[i_tel] = data['hdr'][f'ESO DEL DLT{i_tel+1} OPL START']
        end_OPLs[i_tel]   = data['hdr'][f'ESO DEL DLT{i_tel+1} OPL END']

    # Compute optical path lengths of individual frames with linear interpolation
    start_mjd, end_mjd = mjds[0], mjds[-1] + data['hdr']['EXPTIME'] / 86400.
    if verbose:
        print('Start MJD:', start_mjd, 'End MJD:', end_mjd)
    relative_mjds      = (mjds_valid - start_mjd) / (end_mjd - start_mjd)
    if verbose:
        print('Relative MJDs:', np.shape(relative_mjds))
    OPLs = (np.outer(end_OPLs - start_OPLs, relative_mjds).T + start_OPLs).T
    if verbose:
        print('OPLs inside function:', np.shape(OPLs))

    # FIXME: attention, take into account baseline scrambling here
    dPaths  = np.array([static_lengths[3] + OPLs[3] - static_lengths[2] - OPLs[2],
                        static_lengths[1] + OPLs[1] - static_lengths[0] - OPLs[0],
                        static_lengths[2] + OPLs[2] - static_lengths[1] - OPLs[1],
                        static_lengths[3] + OPLs[3] - static_lengths[1] - OPLs[1],
                        static_lengths[2] + OPLs[2] - static_lengths[0] - OPLs[0],
                        static_lengths[3] + OPLs[3] - static_lengths[0] - OPLs[0]])
    #print('dPaths inside function:', dPaths)
    

    return temperature, pressure, humidity, dPaths

##############################################
# Function to compute the air refractive index
def op_air_index(wlen, T, P, h, N_CO2=435, bands='all'):
    """ Compute the refractive index as a function of wavelength at a given temperature,
        pressure, relative humidity and CO2 concentration, using Equation (5) of Voronin & Zheltikov (2017).
        
        Reference: Voronin, A. A. and Zheltikov, A. M. The generalized Sellmeier equation for air. 
        Sci. Rep. 7, 46111; doi: 10.1038/srep46111 (2017).
        
        Inputs:
        - wlen: array of wavelengths in meters,
        - T: temperature in °C,
        - P: pressure in hPa,
        - h: relative humidity,
        - N_CO2: CO2 concentration in dry air in ppm,
        - bands: list of absorption bands to include according to the numbering in Table 1 of
            Voronin & Zheltikov (2017). Default: 'all' (includes all 15 bands).
        Output:
        - n_air: array of refractive indices at each wl value.
"""
    wl = wlen * 1e6 # m -> µm

    ## Characteristic wavelengths (microns)
    wl_abs1 = 1e-3 * np.array([15131, 4290.9, 2684.9, 2011.3, 47862, 6719.0, 2775.6, 1835.6,
               1417.6, 1145.3, 947.73, 85, 127, 87, 128])
    wl_abs2 = 1e-3 * np.array([14218, 4223.1, 2769.1, 1964.6, 16603, 5729.9, 2598.5, 1904.8, 
               1364.7, 1123.2, 935.09, 24.546, 29.469, 22.645, 34.924])
    
    ## Absorption coefficients
    A_abs1 = np.array([4.051e-6, 2.897e-5, 8.573e-7, 1.550e-8, 2.945e-5, 3.273e-6, 
              1.862e-6, 2.544e-7, 1.126e-7, 6.856e-9, 1.985e-9, 1.2029482,
              0.26507582, 0.93132145, 0.25787285])
    A_abs2 = np.array([1.010e-6, 2.728e-5, 6.620e-7, 5.532e-9, 6.583e-8, 3.094e-6,
              2.788e-6, 2.181e-7, 2.336e-7, 9.479e-9, 2.882e-9, 5.796725,
              7.734925, 7.217322, 4.742131])
    
    ## Conversion
    T += 273.15 # °C -> K
    P *= 1e2    # hPa -> Pa
    
    ## Water vapor
    T_cr = 647.096 # critical-point temperature of water (K)
    P_cr = 22.064e6 # critical-point pressure of water (Pa)
    tau = T_cr / T
    th = 1 - T / T_cr
    a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502
    # Saturated water vapor pressure (Pa)
    P_sat = P_cr * np.exp(tau*(a1*th + a2*th**1.5 + a3*th**3 + a4*th**3.5 + a5*th**4 + a6*th**7.5))
    # Water vapor density (m-3)
    N_H2O = h * P_sat / (const.k_B.value * T)

    ## Dry air pressure (Pa)
    P_dry = P - h * P_sat

    ## Gas concentrations (m-3) (mixing ratio in ppm of dry air * concentration of dry air)
    N_air = P_dry / (const.k_B.value * T)
    N_N2 = (780840 * 1e-6) * N_air
    N_O2 = (209460 * 1e-6) * N_air
    N_Ar = (9340 * 1e-6) * N_air
    N_CO2 = (N_CO2 * 1e-6) * N_air
    
    N_gas = np.zeros_like(wl_abs1)
    N_gas[0:4] = N_CO2
    N_gas[4:11] = N_H2O
    N_gas[11] = N_N2
    N_gas[12] = N_O2
    N_gas[13] = N_Ar
    N_gas[14] = N_H2O
    
    ## Critical plasma density (m-3)
    N_cr = const.m_e.value * const.eps0.value * (2 * np.pi * const.c.value / (const.e.value * wlen))**2
   
    ## Selection of absorption bands
    if bands == 'all':
        bands = np.arange(15)
    else:
        bands = np.array(bands) - 1
    
    ## Refractive index
    n_air = 1
    for i_band in bands:
        dn_air1 = (N_gas[i_band]/N_cr) * A_abs1[i_band] * wl_abs1[i_band]**2 / (wl**2 - wl_abs1[i_band]**2)
        dn_air2 = (N_gas[i_band]/N_cr) * A_abs2[i_band] * wl_abs2[i_band]**2 / (wl**2 - wl_abs2[i_band]**2)
        n_air += dn_air1 + dn_air2
    
    return n_air

##############################################
# Function to correct for the chromatic phase
def op_corr_n_air( data, n_air, dPath, cfin='CF_reord', wlmin=3.3e-6, wlmax=3.7e-6, verbose=False, plot=False):
    if verbose:
        print('Correcting for the chromatic phase...')

    #data, cfdem = op_reorder_baselines(data)
    cfdem = data['CF'][cfin]
    wlen = data['OI_WAVELENGTH']['EFF_WAVE']
    if verbose:
        print('cfdem shape:', cfdem.shape)
    n_bases = np.shape(cfdem)[0]-1
    n_frames = np.shape(cfdem)[1]
    n_wlen_0 = np.shape(cfdem)[2]
    n_wlen = len(wlen)
    
    if verbose:
        print('n_bases:', n_bases, 'n_frames:', n_frames, 'n_wlen_0:', n_wlen_0, 'n_wlen:', n_wlen)

    data['CF']['CF_chr_phase_corr'] = np.copy(cfdem)
    phase_layer_air = np.zeros((6, n_frames, n_wlen))
    slope = np.zeros((6, n_frames))
    phase_layer_air_slope = np.zeros((6, n_frames, n_wlen))
    # wlen *= 1e-6 #µm -> m
    
    if plot:
        fig1, ax1 = plt.subplots(6, 2, figsize=(8, 8), sharex=1, sharey=0)
    colors = COLORS7D

    for i_base in np.arange(6):

        for i_frame in range(n_frames):
            # Model the phase introduced by the extra layer of air
            phase_layer_air[i_base, i_frame] = 2 * np.pi * (n_air-1) * dPath[i_base, i_frame] / (wlen)
            wl_mask_lin = (wlen > wlmin) & (wlen < wlmax)
            wlm         = wlen[wl_mask_lin]
            phasem      = phase_layer_air[i_base, i_frame, wl_mask_lin]
            slope[i_base, i_frame] = np.sum((phasem-phasem.mean())*(1/wlm-np.mean(1/wlm))) / np.sum((1/wlm-np.mean(1/wlm))**2)
            phase_layer_air_slope[i_base, i_frame] = phase_layer_air[i_base, i_frame] - slope[i_base, i_frame] / (wlen)

            # Correct the achromatic phase
            cfobs = cfdem[i_base+1,i_frame]
            corr  = np.exp(1j * phase_layer_air_slope[i_base, i_frame])
            cfcorr = cfobs * np.conj(corr)
            
            if plot :
                phiObs= np.angle(cfobs)
                phi   = np.angle(corr)
                corrphi = np.angle(cfcorr)
                ax1[i_base,0].plot(wlen, corrphi, color=colors[i_base])\
                    
                ax1[i_base,1].plot(wlen, phi, color=colors[i_base],alpha = 0.3)
                ax1[i_base,1].plot(wlen, phiObs, color=colors[i_base])
                ax1[i_base,1].set_ylabel(f'phase {i_base+1}')
                ax1[i_base,1].set_ylim(-np.pi, np.pi)
                
            data['CF']['CF_chr_phase_corr'][i_base+1, i_frame] = cfcorr

    if plot:
        plt.show()
      
    return data, phase_layer_air_slope


##############################################
# Function to get the residual piston on the correlated flux phase, via a FFT's method
def op_get_piston_fft(data, cfin='CF_Binned', verbose=False, plot=False):
    if verbose:
        print('Calculating piston from FFT...')

    if cfin == 'CF_Binned':
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE_Binned']
    else:
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE']
        
    cf = data['CF'][cfin]
    
    #Linear interpolation in wavenumber sigma
    sigma     = 1.0/wlen
    
    dsigma = np.diff(sigma)
    # Make sigma increasing
    if np.mean(dsigma) < 0:
        sigma     = sigma[::-1]
        cf        = cf[...,::-1]
        
    step      = np.min(np.abs(dsigma))
    sigma_lin = np.arange(min(sigma), max(sigma), step)
    if verbose:
        print('cf shape:', cf.shape)
    n_base  = np.shape(cf)[0]
    n_frame = np.shape(cf)[1]
    n_wlen  = np.shape(cf)[2]
    if verbose:
        print('n_base:', n_base, 'n_frame:', n_frame, 'n_wlen:', n_wlen)

    data['CF']['CF_sigma'] = np.zeros((cf.shape[0],cf.shape[1],sigma_lin.shape[0]), dtype=complex)
    OPD_lst = np.zeros((n_base,n_frame))
    if plot:
        fig, ax = plt.subplots(n_base, 1, figsize=(8, 8))
        colors = COLORS7D

    for i_base in range(n_base):
        for i_frame in range(n_frame):
            #Interpolation of correlated flux 
            f = interpolate.interp1d(sigma, np.real(cf[i_base, i_frame,...]))
            cf_real_interp = f(sigma_lin)
            f = interpolate.interp1d(sigma, np.imag(cf[i_base, i_frame,...]))
            cf_imag_interp = f(sigma_lin)
            cf_interp = cf_real_interp + 1j * cf_imag_interp

            data['CF']['CF_sigma'][i_base, i_frame] = cf_interp
            
            log_base_2 = int(math.log2(cf_interp.size)) 
            new_size = int(2**(log_base_2+4))
            cf_interp = np.pad(cf_interp, (new_size//2 - cf_interp.shape[0]//2), mode='constant', constant_values=0)

            fft_cf = np.fft.fftshift(np.fft.fft(cf_interp))
            OPDs   = np.fft.fftshift(np.fft.fftfreq(cf_interp.shape[0], step))

            dsp = np.abs(fft_cf)
            mx = np.argmax(dsp)
            #OPD determination
            OPD0 = OPDs[mx]
            OPDp1 = OPDs[mx+1]
            OPDm1 = OPDs[mx-1]
            
            peak0 = dsp[mx]
            peakp1 = dsp[mx+1]
            peakm1 = dsp[mx-1]
            
            OPD = (OPD0 * peak0 + OPDp1 * peakp1 + OPDm1 * peakm1)/(peak0 + peakp1 + peakm1)
            
            OPD_lst[i_base, i_frame] = OPD
        


        if plot:
            ax[i_base].plot(OPDs * 1e6, np.sqrt(fft_cf.real**2 + fft_cf.imag**2), color=colors[i_base])
            ax[i_base].set_xlabel('OPD [µm]')
            ax[i_base].set_xlim(-600, 600)

    # if verbose:
    #     print('OPD:', OPD_lst)

    data['CF']['piston_fft'] = OPD_lst
    data['CF']['pistons']    = OPD_lst
    return data, OPD_lst

#################################################
# Function to get the piston slope on the correlated flux phase
def op_get_piston_slope(data, cfin='CF_Binned', wlenmin=3.1e-6, wlenmax=3.8e-6, verbose=False, plot=False):
    if verbose:
        print('Calculating piston slope...')

    if cfin == 'CF_Binned':
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE_Binned']
    else:
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE']
        
    print('wlen:',wlen)
    cf = data['CF'][cfin]
    n_bases  = cf.shape[0]
    n_frames = cf.shape[1]
    slopes   = np.zeros((n_bases, n_frames))
    ord_og   = np.zeros((n_bases, n_frames))
    
    for i_base in range(n_bases):
        for i_frame in range(n_frames):
            wl_mask = (wlen > wlenmin) & (wlen < wlenmax)
            wlenm   = wlen[wl_mask]
            phasem  = np.angle(cf[i_base, i_frame])[wl_mask]
            slope   = np.polyfit(wlenm, phasem, 1)[0]
            ords_og = np.polyfit(wlenm, phasem, 1)[1]
            slopes[i_base, i_frame] = slope
            ord_og[i_base, i_frame] = ords_og
            
    if plot:
        fig, ax = plt.subplots(7, 1, figsize=(8, 8))
        colors = COLORS7D
        for i_base in range(n_bases):
            ax[i_base].plot(wlen, ord_og[i_base,0]+slopes[i_base,0]*wlen, color=colors[i_base])
            ax[i_base].plot(wlen, np.angle(cf[i_base, 0]), color=colors[i_base], linestyle='--', alpha=0.4)
            ax[i_base].set_xlabel('Frame')
            ax[i_base].set_ylabel('Piston slope')

    wlenmm = np.mean(wlenm)
    slopes *= -(wlenmm**2) / (2 * np.pi) #rad/m -> µm
    data['CF']['piston_slope'] = slopes
    data['CF']['pistons'] = slopes

    return data, slopes


####################################################
# Function to get the final piston using a chi2 minimization
def op_get_piston_chi2(data, init_guess, wlenmin=3.2e-6, wlenmax=3.8e-6, cfin='CF_Binned', verbose=False, plot=False):
    if verbose:
        print('Calculating piston using chi2 minimization...')

    if cfin == 'CF_Binned':
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE_Binned']
    else:
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE']
        
    wl_mask = (wlen > wlenmin) & (wlen < wlenmax)
    wlenm   = wlen[wl_mask]
    wlenMean     = np.mean(wlen)
    # cf           = data['CF']['CF_achr_phase_corr']
    cf           = data['CF'][cfin]
    cfm  = cf[...,...,wl_mask]
    n_bases      = cf.shape[0]
    n_frames     = cf.shape[1]
    pistons      = np.zeros((n_bases, n_frames))

    if init_guess in ['slope', 'Slope', 'SLOPE']:
        slopes = data['CF']['piston_slope']
        init_guess = slopes
    elif init_guess in ['fft', 'FFT', 'Fft']:
        OPDs = data['CF']['piston_fft']
        init_guess = OPDs

    def chi2_1(piston, cf1D, wlen):
        phasor_model = np.exp(-2j * np.pi * piston / wlen)
        cres = cf1D * phasor_model
        chi2_val    = np.sum(cres.imag**2) / np.sum(np.abs(cres)**2)
        return chi2_val

    def chi2_1b(piston_offset, cf1D, wlen):
        phasor_model = np.exp(-2j * np.pi * (piston_offset[0] / wlen - piston_offset[1]))
        cres = cf1D * phasor_model
        chi2_val    = np.sum(cres.imag**2) / np.sum(np.abs(cres)**2)
        return chi2_val

    def chi2_2(piston, cf1D, wlen):
        phasor_model = np.exp(2j * np.pi * piston / wlen)
        phasor = cf1D * np.conj(phasor_model)
        chi2_val    = np.sum(np.angle(phasor))**2 * np.sum(np.abs(cf1D)) **2
        return chi2_val

    for i_base in range(n_bases):
        for i_frame in range(n_frames):
            cf_frame = cf[i_base, i_frame]
            initial_guess = (init_guess[i_base, i_frame],0)  # Initial guess for the piston
            result = minimize(chi2_1b, initial_guess, args=(cf_frame, wlen))
            print('result:', result.x)
            pistons[i_base, i_frame] = result.x[0]

    if plot:
        fig, ax = plt.subplots(7, 2, figsize=(8, 8))
        colors = COLORS7D
        for i_base in range(n_bases):
            ax[i_base,0].plot(pistons[i_base], color=colors[i_base], marker='*', linestyle='dashed', label='Final')
            ax[i_base,0].plot(init_guess[i_base], color=colors[i_base], marker='*', label='Initial')
            ax[i_base,0].legend()
            ax[i_base,0].set_xlabel('Frame')
            ax[i_base,0].set_ylabel('Piston [µm]')
            ax[i_base,0].set_ylim(-10e-6,10e-6)
            
            pistonss = np.linspace(-100e-6, 100e-6, 1000)
            chi2s = np.copy(pistonss)
            for i_frame in range(n_frames):
                for ipist,pist in enumerate(pistonss):
                    chi2s[ipist] = chi2_1(pist, cf[i_base, i_frame], wlen)
                ax[i_base,1].plot(pistonss, chi2s)
            ax[i_base,1].set_xlim(-10e-6,10e-6)

        plt.show()
    #if verbose:
    print('OPD chi2:', pistons)
    
    data['CF']['pistons'] = -pistons
    return data, pistons

####################################################
# Function to correct from the residual piston
def op_corr_piston(data, cfin='CF_Binned', verbose=False, plot=False):
    if verbose:
        print('Correcting for the residual piston...')

    if cfin == 'CF_Binned':
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE_Binned']
    else:
        wlen         = data['OI_WAVELENGTH']['EFF_WAVE']
        
    cf       = data['CF'][cfin]
    pistons  = data['CF']['pistons']
    n_bases  = cf.shape[0]
    n_frames = cf.shape[1]

    data['CF']['CF_piston_corr'] = np.copy(cf)

    for i_base in np.arange(6):
        for i_frame in range(n_frames):
            cf_frame = cf[i_base+1, i_frame]
            piston = pistons[i_base+1, i_frame]
            # print('piston:', piston)   
            corr =  np.exp(1j * 2 * np.pi * piston / wlen)
            cf_corr = cf_frame * np.conj(corr)
            data['CF']['CF_piston_corr'][i_base+1, i_frame] = cf_corr
    
    if plot:
        colors = COLORS7D
        fig, ax = plt.subplots(n_bases, 1, figsize=(8, 8))
        for i_base in range(n_bases):
            ax[i_base].plot(np.angle(data['CF']['CF_piston_corr'][i_base, 0]), color=colors[i_base])
        plt.show()
 
    return data

##############################################
# Bin data
def op_bin_data(data, binning=5, cfin='CF_achr_phase_corr', verbose=False, plot=False):
    wlen   = data['OI_WAVELENGTH']['EFF_WAVE']
    cfdem  = data['CF'][cfin]
    n_wlen = len(wlen)
    n_bins = n_wlen // binning
    binned_wlen = np.zeros(n_bins)
    for i in range(n_bins):
        binned_wlen[i] = np.mean(wlen[i*binning:(i+1)*binning])
    if verbose:
        print('Binned wavelengths:', binned_wlen)
    
    binned_cf = np.zeros((cfdem.shape[0], cfdem.shape[1], n_bins), dtype=complex)
    for i in range(cfdem.shape[0]):
        for j in range(cfdem.shape[1]):
            for k in range(n_bins):
                binned_cf.real[i, j, k] = np.mean(cfdem.real[i, j, k*binning:(k+1)*binning])
                binned_cf.imag[i, j, k] = np.mean(cfdem.imag[i, j, k*binning:(k+1)*binning])

    data['OI_WAVELENGTH']['EFF_WAVE_Binned'] = binned_wlen
    data['CF']['CF_Binned'] = binned_cf

    if plot:
        plt.figure()
        plt.plot(wlen, np.abs(cfdem[1, 0,:]), label='Original')
        plt.plot(binned_wlen, np.abs(binned_cf[1, 0,:]), label='Binned')
        plt.xlabel('Wavelength')
        plt.ylabel('Correlated Flux')
        plt.legend()
        plt.show()

    return data


##############################################
# 
def op_get_error_vis(data,cfin='CF_piston_corr2',plot=False):
    colors = COLORS6D
    wlen   = data['OI_WAVELENGTH']['EFF_WAVE']

    cf  = data['CF'][cfin][1:]
    nbases    = cf.shape[0]
    nframes   = cf.shape[1]
    visAmpErr = np.zeros_like(cf)
    visPhiErr = np.zeros_like(cf)
    
    # Parameters of the smoothing / RMS
    width = 31 
    dev = width/(2*np.sqrt(2*np.log10(2)))
    kernel = Gaussian1DKernel(dev)
    
    for iBase in range(nbases):
        
        if plot:
            fig1, ax1 = plt.subplots(nframes, 2, figsize=(8, 8), sharex=1, sharey=0)
        if nframes < 10 : # IF GRAV4MAT
            for iFrame in range(nframes):
                amp  = np.abs(cf[iBase][iFrame]) 
                phi  = np.angle(cf[iBase][iFrame])
                
                #Smoothing Function
                smooth_amp = convolve(amp,kernel,normalize_kernel = True)
                smooth_phi = convolve(phi,kernel,normalize_kernel = True)
                
                sq_amp = np.abs(amp-smooth_amp)**2
                sq_phi = np.abs(phi-smooth_phi)**2
                
                local_mean_sq = uniform_filter1d(sq_amp, size=width, mode='nearest')
                visAmpErr[iBase, iFrame, :] = np.sqrt(local_mean_sq/width)
    
                local_mean_sq_phi = uniform_filter1d(sq_phi, size=width, mode='nearest')
                visPhiErr[iBase, iFrame, :] = np.sqrt(local_mean_sq_phi/width)
                
                if plot:
                    ax1[iFrame,0].plot(wlen, smooth_amp, color='black',alpha=0.9)
                    ax1[iFrame,0].set_ylim(0,np.max(amp)*1.1)
                    ax1[iFrame,0].set_ylabel(f'frame {iFrame+1} AMP')
                    ax1[iFrame,1].plot(wlen, np.degrees(smooth_phi), color='black',alpha=0.9)
                    ax1[iFrame,1].set_ylabel(f'frame {iFrame+1} PHI (°)')
                    ax1[iFrame,1].set_ylim(-15,15)
                    ax1[iFrame,0].plot(wlen,  amp, color=colors[iFrame],alpha=0.3)
                    ax1[iFrame,1].plot(wlen, np.degrees(phi), color=colors[iFrame],alpha=0.3)
                        
                    plt.suptitle(f'CF data and smoothened data for error base = {iBase+1} \n width = {width}')
                    # plt.savefig(os.path.expanduser(bbasedir+'Result/Smoothing/'+f'Smoothed_{window_length}_{polyorder}_{WL_err}.png'))
                    plt.tight_layout()
                    
        else : 
            for iFrame in range(0,nframes,nframes//10):
                if iFrame not in (range(nframes-9,nframes)):
                    amp = np.abs(cf[iBase][iFrame:(iFrame+10)]) 
                    phi = np.angle(cf[iBase][iFrame:(iFrame+10)])
                    
                    #Mean over the Frame
                    mean_amp = np.mean(amp,axis=0)
                    mean_phi = np.mean(phi,axis=0)
                    
                    sq_amp = np.abs(amp-mean_amp[None,:])**2
                    sq_phi = np.abs(phi-mean_phi[None,:])**2
                    
                    mean_sq = np.mean(sq_amp,axis=0)
                    visAmpErr[iBase, iFrame:iFrame+10, :] = np.sqrt(mean_sq)
                    
                    mean_sq_phi = np.mean(sq_phi,axis=0)
                    visPhiErr[iBase, iFrame:iFrame+10, :] = np.sqrt(mean_sq_phi)
                    
                    if plot:
                        ax1[iFrame,0].plot(wlen, mean_amp, color='black',alpha=0.9)
                        ax1[iFrame,0].set_ylim(0,np.max(amp)*1.1)
                        ax1[iFrame,0].set_ylabel(f'frame {iFrame+1} AMP')
                        ax1[iFrame,1].plot(wlen, np.degrees(mean_phi), color='black',alpha=0.9)
                        ax1[iFrame,1].set_ylabel(f'frame {iFrame+1} PHI (°)')
                        ax1[iFrame,1].set_ylim(-15,15)
                        
                        for i in range(iFrame,iFrame+10):
                            ax1[iFrame,0].plot(wlen,  amp[i], color=colors[iFrame],alpha=0.3)
                            ax1[iFrame,1].plot(wlen, np.degrees(phi[i]), color=colors[iFrame],alpha=0.3)
                            
                        plt.suptitle('CF data and smoothened data for error ')
                        plt.tight_layout()
                        
        
            
        if plot:
            # plt.savefig(os.path.expanduser(bbasedir+'Result/Smoothing/'+f'Smoothed_base{iBase+1}_{window_length}_{polyorder}_{WL_err}.png'))
            plt.show()
        
                
           
    
    op_snr(wlen,cf,visAmpErr,visPhiErr,width,plot=plot)
    visAmpErr=np.reshape(np.swapaxes(visAmpErr, 0,1), (visAmpErr.shape[0]* visAmpErr.shape[1],visAmpErr.shape[2]))
    visPhiErr=np.reshape(np.swapaxes(visPhiErr, 0,1), (visPhiErr.shape[0]* visPhiErr.shape[1],visPhiErr.shape[2]))
    data['OI_BASELINES']['VISAMPERR']=np.real(visAmpErr)
    data['OI_BASELINES']['VISPHIERR']=np.real(visPhiErr)
    return data


##############################################
# 
def op_snr(wlen,vis,visAmpErr,visPhiErr,width,plot=False):
    visAmp = np.abs(vis)
    snr_amp = visAmp/visAmpErr-1 #TATULLI 2007
    snr_phi = np.abs(1/visPhiErr)
    
    mean_snr = np.mean(snr_amp,axis=1)
    mean_snr_phi = np.mean(snr_phi,axis=1)
    
    if plot :
        nbases = len(vis)
        nframes = len(vis[0])
        colors = COLORS6D
        fig1, ax1 = plt.subplots(nbases, 2, figsize=(8, 8), sharex=1, sharey=0)
        for ibase in range(nbases):

            ax1[ibase,0].plot(wlen, mean_snr[ibase,:], color='black',alpha=0.9)
            ax1[ibase,0].set_ylim(0,np.max(mean_snr)*1.05)
            ax1[ibase,0].set_ylabel(f'base {ibase+1} AMP')
            ax1[ibase,1].plot(wlen, mean_snr_phi[ibase,:], color='black',alpha=0.9)
            
            ax1[ibase,1].set_ylabel(f'base {ibase+1} PHI (°)')
            ax1[ibase,1].set_ylim(0,np.max(mean_snr_phi)*1.05)
            # for iframe in range(nframes):
            #     ax1[ibase,0].plot(wlen,  snr_amp[ibase,iframe,:], color=colors[ibase],alpha=0.3)
            #     ax1[ibase,1].plot(wlen, snr_phi[ibase,iframe,:], color=colors[ibase],alpha=0.3)
            
        plt.suptitle(f'SNR \n width = {width}')
        plt.tight_layout()
        # plt.savefig(os.path.expanduser(bbasedir+'Result/SNR/'+f'SNR_{window_length}_{polyorder}_{WL_err}.png'))
        plt.show()
    return snr_amp,snr_phi

##############################################
#          

def op_snr_theory(data,cfin = 'CF_piston_corr2',plot=False):
    wlen   = data['OI_WAVELENGTH']['EFF_WAVE']
    cf  = data['CF'][cfin][0]/4
    nframes = cf.shape[0]
    
    def temperature_profile(z):
        return np.where(z < 11000, 288.15 - 0.0065 * z , 216.65)
    
    def planck(Wavelength, T):
        h = cst.h.value
        c = cst.c.value
        k_B = cst.k_B.value
        return (2 * h * c**2 / Wavelength**5) / (np.exp(h * c / (Wavelength * k_B * T)) - 1)
    ndit    = 6
    D       = 8
    obs     = 1.2
    DITs    = np.ones_like(wlen)*0.125
    texp   = np.ones_like(wlen)*10
    Dwlen   = np.ones_like(wlen)*3.5e-6/30
    pinhole = 1.5 * 3.5e-6 / D
    collecting_surface = (np.pi/4) * (D**2 - obs**2)
    T_ground = 298.15
    N_pix = 72*3
    em_sky  = 0.1
    t_ow    = 0.99
    t_oc    = 0.99
    obs_alt = 2635#m
    max_alt = 20000#m
    ratio  = 0.93
    nT     = 4
    Vstar  = 1
    Vinst  = 0.85
    alphaI = 2/3
    #US Standard Atmosphere
    z_grid = np.linspace(obs_alt, max_alt, 500)
    T_grid = temperature_profile(z_grid)
    dz= z_grid[1] - z_grid[0]
    
    # Sky background radiation
    rad_tot= np.zeros(len(wlen))
    for  T in zip( T_grid):
        rad_tot += planck(wlen, T) * em_sky * dz* (t_ow * t_oc)/(max_alt-obs_alt)
    
    
    #Sky background radiation on floor
    #bckg_sky_received = planck(wlen, T_ground) * em_sky * (t_ow * t_oc)  
    # Optical background
    bckg_opt =  (1 - t_ow**31)  * planck(wlen, T_ground) * t_oc**20
    # Total received thermal noise
    bckg_tot_recu = rad_tot + bckg_opt #+bckg_sky_received
    
    Bckg_lr = wlen / (cst.h.value * cst.c.value) * DITs * Dwlen * bckg_tot_recu * np.pi/4* pinhole**2 * collecting_surface
    nIb = alphaI * Bckg_lr
    # n_star = 0.002 * 1e-26 / (cst.h.value * wlen ) * collecting_surface * texp * Dwlen * ratio 
    # nI = alphaI * n_star
    RON = 15
    F_RON = (RON * cst.h.value * cst.c.value / wlen) / (collecting_surface * texp * Dwlen * ratio)
    nI = alphaI * cf
    snr = nI * Vstar * Vinst / np.sqrt( nT * nIb + nT * nI + N_pix * F_RON)*np.sqrt(ndit)
    
    if plot:
        fig1, ax1 = plt.subplots(nframes, 1, figsize=(4, 8), sharex=1, sharey=0)
        for i in range(nframes):
            ax1[i].plot(wlen,np.abs(snr[i]))
            ax1[i].set_ylabel(f'frame Amp {i+1}')   
        plt.suptitle('theorical SNR \n ')
        plt.tight_layout()
        plt.show()
    return snr