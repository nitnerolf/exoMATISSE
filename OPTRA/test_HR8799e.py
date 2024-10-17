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

#plt.ion()

plot    = False
plotdsp = False
verb    = False

bbasedir = '~/Documents/G+/'
bbasedir = '~/SynologyDrive/driveFlorentin/GRAVITY+/HR8799e/'
basedir  = bbasedir+'GPAO_HR8799e/'
starfile = 'MATISSE_OBS_SIPHOT_LM_OBJECT_272_0001.fits'
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

#########################################################
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

#########################################################
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

#########################################################
# Get the wavelength
wlen = op_get_wlen(caldir+shiftfile, fdata, verbose=verb)


#########################################################
peaks, peakswd = op_get_peaks_position(fdata, wlen, 'MATISSE', verbose=verb)

colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']

if plotdsp:
    for i in range(np.shape(peaks)[0]):
        plt.plot(peaks[i,:], np.arange(np.shape(peaks)[1]),color=colors[i])
        plt.plot(peaks[i,:]+peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
        plt.plot(peaks[i,:]-peakswd[i,:]/2, np.arange(np.shape(peaks)[1]),color=colors[i])
    plt.show()

#########################################################
cfdata = op_extract_CF(fdata, wlen, peaks, peakswd, verbose=verb)

iframe = 0
iwlen = 1400
    
if plotdsp:
    plt.figure(1)
    plt.imshow(np.angle(cfdata['CF']['data'][1,iframe,:,:]), cmap='gray')
    plt.title('2D phase map for one peak')

    plt.figure(2)
    for i in np.arange(7):
        plt.plot(np.angle(cfdata['CF']['data'][i,iframe,iwlen,:]),color=colors[i])
    plt.title('Cut of the phase of CF Data')

    plt.figure(3)
    for i in np.arange(7):
        plt.plot(np.abs(cfdata['CF']['data'][i,iframe,iwlen,:]),color=colors[i])
    plt.plot(np.abs(cfdata['CF']['bck'][iframe,iwlen,:]))
    plt.title('Modulus of Complex Values for CF Data and Background')

#########################################################
op_demodulate(cfdata, wlen, verbose=True, plot=False)



import matplotlib.animation as animation
fig, ax = plt.subplots()
#lines = [ax.plot([], [], color=colors[i])[0] for i in range(1)]
lines  = [ax.plot([], [], color=colors[i])[0] for i in np.arange(7)]
#lines2 = [ax.plot([], [], '--', color=colors[i])[0] for i in np.arange(7)]
#ax.set_xlim(0, cfdata['CF']['CF'].shape[2])
ax.set_xlim(np.min(wlen), np.max(wlen))
ax.set_ylim(-np.pi, np.pi)
ax.set_title('Phase as a function of the wavelength for CF Data')
def init():
    for line in lines:
        line.set_data([], [])
    return lines
def update(frame):
    for i, line in enumerate(lines):
        if i == 1:
            #line.set_data(wlen, np.angle(cfdata['CF']['CF'][i, frame, :] * np.conjugate(cfdata['CF']['mod_phasor'][5, frame, :])))
            line.set_data(wlen, np.angle(cfdata['CF']['CF_demod'][i, frame, :]))
            #lines2[i].set_data(wlen, np.angle(cfdata['CF']['mod_phasor'][i-1, frame, :]))
            
    return lines
ani = animation.FuncAnimation(fig, update, frames=cfdata['CF']['CF'].shape[1], init_func=init, blit=True)
plt.show()

    
if plotdsp:
    plt.figure(5)
    for i in np.arange(6)+1:
        plt.plot(np.abs(cfdata['CF']['CF'][i,iframe,:]) / np.abs(cfdata['CF']['CF'][0,0,:])*3,color=colors[i])
        plt.plot(np.max(np.abs(cfdata['CF']['data'][i,iframe,:,:]),axis=1) / np.abs(cfdata['CF']['CF'][0,0,:])*3*7,'--',color=colors[i])
    plt.ylim(-0.2,1.2)
    plt.title('This one should resemble a visibility curve')



plt.show()