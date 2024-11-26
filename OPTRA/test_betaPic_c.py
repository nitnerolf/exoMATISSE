"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example of OPTRA pipeline for the beta Pic c data
Author: fmillour
Date: 19/11/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
from op_vis        import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os
from scipy.ndimage import median_filter
from scipy         import *
from scipy         import stats

#plt.ion()
plot = True
plotFringes   = plot
plotPhi       = plot
plotDsp       = plot
plotRaw       = plot
plotCorr      = plot

#bbasedir = '~/SynologyDrive/driveFlorentin/GRAVITY+/HR8799e/'
bbasedir = os.path.expanduser('~/Documents/ExoMATISSE/beta_Pic_c/')

basedir  = bbasedir+'2023-11-27/'

starfiles = os.listdir(basedir)
fitsfiles = [f for f in starfiles if ".fits" in f]

for fi in fitsfiles:
    fh = fits.open(basedir+fi)
    op_print_fits_header(fh)
    hdr = fh[0].header
    #print(fi, hdr['ESO'])


print('Starfiles:', starfiles)
#starfiles = [f for f in starfiles if "LM_OBJECT" in f]
starfiles = [f for f in starfiles if "LM_STD" in f]
print('Filtered Starfiles:', starfiles)

for ifile in starfiles:
    starfile = basedir + ifile
#starfile = basedir + 'MATISSE_OBS_SIPHOT_LM_OBJECT_323_0003.fits'
    skyfile  = basedir + 'MATISSE_OBS_SIPHOT_LM_SKY_323_0003.fits'

    caldir    = '~/Documents/ExoMATISSE/CALIB2024/'
    kappafile = caldir+'KAPPA_MATRIX_L_MED.fits'
    shiftfile = caldir+'SHIFT_L_MED.fits'
    flatfile  = caldir+'FLATFIELD_L_SLOW.fits'
    badfile   = caldir+'BADPIX_L_SLOW.fits'

    ##########################################################

    bdata = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=True, plot=plotRaw)

    ##########################################################

    print('Shape of bdata:', bdata['INTERF']['data'].shape)

    if plotFringes:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
        # Plot the first frame of intf
        ax1.imshow(np.mean(bdata['INTERF']['data'], axis=0), cmap='viridis')
        ax1.set_title('average intf')

        plt.show()
        
    cfdata, wlen = op_get_corrflux(bdata, shiftfile, plot=plotCorr)

    print(wlen)

    #########################################################

    vis2, mask = op_extract_simplevis2(cfdata, verbose=False, plot=False)
    print(mask)
    print(~mask)
    notvis2 = np.ma.masked_array(np.ma.getdata(vis2), ~mask)
    allvis2 = np.ma.getdata(vis2)
    
    print('Shape of vis2:', vis2.shape)
    print('Shape of notvis2:', notvis2.shape)

    fig0, ax0 = plt.subplots(7, 1, figsize=(8, 8), sharex=1, sharey=1)
    print('Shape of ax1:', ax0.shape)
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
    for i in np.arange(7):
        print('i:', i)
        ax0[i].plot(wlen, allvis2[i,:], color='lightgray')
        ax0[i].plot(wlen, vis2[i,:], color=colors[i])
        ax0[i].set_ylabel(f'vis2 {i}')

    basename = os.path.basename(starfile)
    base = os.path.splitext(basename)[0]
    print('Basename of starfile:', base)
    plt.suptitle(f'Visibility as a function of $\lambda$, {base}')
    plt.xlim(np.min(wlen), np.max(wlen))
    plt.ylim(-0.1, 1.1)
    print(os.path.expanduser(bbasedir+f'{base}_vis2.png'))
    plt.savefig(os.path.expanduser(bbasedir+f'{base}_vis2.png'))
    plt.show()

    #########################################################

    '''
    cfdem = op_demodulate(cfdata, wlen, verbose=True, plot=False)

    print('Shape of cfdata:', cfdem['CF']['CF_demod'].shape)
    cf = cfdem['CF']['CF_demod']
    sumcf = np.sum(cf, axis=1)
    print('Shape of sumcf:', sumcf.shape)

    fig1, ax1 = plt.subplots(6, 2, figsize=(8, 8), sharex=1, sharey=0)
    print('Shape of ax1:', ax1.shape)
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
    for i in np.arange(6):
        print('i:', i)
        ax1[i,0].plot(wlen,   np.abs(sumcf[i+1,:]), color=colors[i])
        ax1[i,0].set_ylabel(f'corr. flux {i+1}')
        ax1[i,1].plot(wlen, np.angle(sumcf[i+1,:]), color=colors[i])
        ax1[i,1].set_ylabel(f'phase {i+1}')
    plt.suptitle('Sum CF data (1 exposure)')
    plt.tight_layout()
    plt.show()

    iframe = 0
    fig2, ax2 = plt.subplots(6, 2, figsize=(8, 8), sharex=1, sharey=0)
    for i in np.arange(6):
        ax2[i,0].plot(wlen, np.abs(cf[i+1,iframe,:]), color=colors[i])
        ax1[i,0].set_ylabel(f'corr. flux {i+1}')
        ax2[i,1].plot(wlen, np.angle(cf[i+1,iframe,:]), color=colors[i])
        ax1[i,1].set_ylabel(f'phase {i+1}')
    plt.suptitle(f'frame {iframe} of CF data')
    plt.tight_layout()
    plt.show()


                #line.set_data(wlen, np.angle(cfdata['CF']['CF'][i, frame, :]  * np.conjugate(cfdata['CF']['mod_phasor'][2, frame, :])))
                '''

    '''
    #scfdata = op_sortout_peaks(cfdata, verbose=True)
    #scfdata = cfdata

    iframe = 0
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
            if i == 5:
                #line.set_data(wlen, np.angle(cfdata['CF']['CF'][i, frame, :]  * np.conjugate(cfdata['CF']['mod_phasor'][2, frame, :])))
                
                # CF 1 phi 5 -> 6
                # CF 2 phi 0 -> 1
                # CF 3 phi 3 -> 4
                # CF 4 phi 4 -> 5
                # CF 5 phi 1 -> 2
                # CF 6 phi 2 -> 3
                line.set_data(wlen, np.angle(cfdata['CF']['CF_demod'][i, frame, :]))
                
                #lines2[i].set_data(wlen, np.angle(cfdata['CF']['mod_phasor'][i-1, frame, :]))
                
        return lines
    ani = animation.FuncAnimation(fig, update, frames=cfdata['CF']['CF'].shape[1], init_func=init, blit=True)
    plt.show()
        
    if plotDsp:
        plt.figure(5)
        for i in np.arange(6)+1:
            if i == 5:
                plt.plot(np.abs(cfdata['CF']['CF'][i,iframe,:]) / np.abs(cfdata['CF']['CF'][0,0,:])*3,color=colors[i])
                plt.plot(np.max(np.abs(cfdata['CF']['data'][i,iframe,:,:]),axis=1) / np.abs(cfdata['CF']['CF'][0,0,:])*3*7,':',color=colors[i])
        plt.ylim(-0.2,1.2)
        plt.title('This one should resemble a visibility curve')



    plt.show()
    '''