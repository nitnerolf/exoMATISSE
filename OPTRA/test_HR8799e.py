from op_corrflux import *
from op_cosmetics import *
from op_flux import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from scipy.ndimage import median_filter
from scipy import *
from scipy import stats


plot = False
plotdsp = False
verb = False

bbasedir = '~/Documents/G+/'
bbasedir = '~/SynologyDrive/driveFlorentin/GRAVITY+/HR8799e/'
basedir  = bbasedir+'GPAO_HR8799e/'
starfile = 'MATISSE_OBS_SIPHOT_LM_OBJECT_272_0005.fits'
skyfile  = 'MATISSE_OBS_SIPHOT_LM_SKY_272_0001.fits'

caldir    = bbasedir+'CALIB2024/'
kappafile = 'KAPPA_MATRIX_L_MED.fits'
shiftfile = 'SHIFT_L_MED.fits'
flatfile  = 'FLATFIELD_L_SLOW.fits'
badfile   = 'BADPIX_L_SLOW.fits'

# Load the star and sky data
tardata  = op_load_rawdata(basedir+starfile)
starname = tardata['hdr']['OBJECT']
print(starname)

# Load the sky data
skydata  = op_load_rawdata(basedir+skyfile)

# Subtract the sky from the star data
stardata = op_subtract_sky(tardata, skydata)

# Load the calibration data
bpm = op_load_bpm(caldir+badfile)
ffm = op_load_ffm(caldir+flatfile)

fdata = op_apply_ffm(stardata, ffm, verbose=verb)
bdata = op_apply_bpm(fdata, bpm, verbose=verb)

print('Shape of bdata:', bdata['INTERF']['data'].shape)

if plot:
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
    # Plot the first frame of intf
    ax1.imshow(np.mean(bdata['INTERF']['data'],axis=0), cmap='gray')
    ax1.set_title('average intf')

    plt.show()

# Apodization
adata = op_apodize(bdata, verbose=verb, plot=plot)

if plot:
    # Compute the average of intf after apodization
    avg_intf = np.mean(adata['INTERF']['data'], axis=0)
    vmn = 1e-9
    vmx = np.max(avg_intf)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.tight_layout()
    split_images = np.array_split(avg_intf, 3, axis=0)
    for ax, img in zip(axes, split_images):
        ax.imshow(img, cmap='gray')
    plt.title('Average of intf after Apodization')
    plt.show()

#compute fft
fdata = op_calc_fft(adata)

if plotdsp:
    # Compute the average of intf after apodization
    sum_dsp = np.log(fdata['FFT']['sum_dsp'])
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

# Get the wavelength
wlen = op_get_wlen(caldir+shiftfile, fdata, verbose=verb)

colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']

peaks, peakswd = op_get_peaks_position(fdata, wlen, 'MATISSE', verbose=verb)

if plot:
    for i in range(np.shape(peaks)[0]):
        plt.plot(peaks[i,:], np.arange(np.shape(peaks)[1]),color=colors[i])
        plt.plot(peaks[i,:]+peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
        plt.plot(peaks[i,:]-peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
    plt.show()


cfdata = op_extract_CF(fdata, wlen, peaks, peakswd, verbose=verb)

plt.figure(1)
plt.imshow(np.angle(cfdata['CF']['data'][1,0,:,:]), cmap='gray')
plt.figure(2)
for i in np.arange(7):
    plt.plot(np.angle(cfdata['CF']['data'][i,0,100,:]))
plt.figure(3)
for i in np.arange(7):
    plt.plot(np.angle(cfdata['CF']['CF'][i,0,:]))
plt.figure(4)
for i in np.arange(6)+1:
    plt.plot(np.abs(cfdata['CF']['CF'][i,0,:]) / np.abs(cfdata['CF']['CF'][0,0,:])*3)
plt.show()