#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Example script to process MATISSE data of beta Pic b with GPAO
# Author: fmillour
# Date: 18/11/2024
# Project: OPTRA
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

#plt.ion()
plot = 0
plotFringes = plot
plotPhi     = plot
plotDsp     = plot
plotRaw     = plot
plotCorr    = plot
plotPist    = plot

verbose = False

#bbasedir = '/Users/jscigliuto/Nextcloud/DATA/betaPicb/'
#basedir = bbasedir+'betaPicb_rawdata_2024-11-17/'
#bbasedir = '~/SynologyDrive/driveFlorentin/ExoMATISSE/beta_Pic_b/'
bbasedir = os.path.expandvars('$HOME/driveFlorentin/DATA/beta_Pic_b/')
#bbasedir = os.path.expanduser('~/Documents/ExoMATISSE/beta_Pic_b/')
basedir  = bbasedir+'2024-11-17_MATISSE_betaPic_b/'

outdir = os.path.expandvars('$HOME/beta_pic_b_GPAO/')
starfiles = os.listdir(basedir)
#print(starfiles)
fitsfiles = [f for f in starfiles if ".fits" in f and not "M." in f]
#print(fitsfiles)

caldir = os.path.expandvars('$HOME/driveFlorentin/DATA/CALIB2024/')

ext = '.fits.gz'
kappafile = caldir+'KAPPA_MATRIX_L_MED'+ext
shiftfile = caldir+'SHIFT_L_MED'+ext
flatfile  = caldir+'FLATFIELD_L_SLOW'+ext
badfile   = caldir+'BADPIX_L_SLOW'+ext

colors = ['#7a0e04', '#7a4f04', '#6a7a04', '#317a04', '#047d6f', '#04477d', '#45077a']

# select only fits files that correspond to observations
obsfilesL     = []
obsfilesL_MJD = []
obsfilesN     = []
skyfilesL     = []
skyfilesL_MJD = []
darkfiles     = []

for fi in fitsfiles:
    #print(fi)
    fh = fits.open(basedir+fi)
    #op_print_fits_header(fh)
    hdr = fh[0].header
    fh.close()
    inst = hdr['INSTRUME']
    catg = hdr['ESO DPR CATG']
    type = hdr['ESO DPR TYPE']
    try:
        chip = hdr['ESO DET CHIP NAME']
    except:
        do_nothing=1
    try:
        dit  = hdr['ESO DET SEQ1 DIT']
    except:
        do_nothing=1
        #print('No DIT in header')
    try:
        ndit = hdr['ESO DET NDIT']
    except:
        do_nothing=1
    #print(fi, inst, catg, type, chip, dit, ndit)
    if catg == 'SCIENCE' and type == 'OBJECT' and chip == 'HAWAII-2RG' :
        #print("science file!")
        obsfilesL.append(fi)
        obsfilesL_MJD.append(hdr['MJD-OBS'])
    elif catg == 'CALIB' and type == 'STD' and chip == 'HAWAII-2RG':
        #print("calibrator file!", fi)
        obsfilesL.append(fi)
        obsfilesL_MJD.append(hdr['MJD-OBS'])
    elif catg == 'CALIB' and type == 'SKY' and chip == 'HAWAII-2RG' :
        #print("sky file!", fi)
        skyfilesL.append(fi)
        skyfilesL_MJD.append(hdr['MJD-OBS'])
    else:
        do_nothing = 1
        #print('Not a science or sky file:', fi)

skyfilesL_MJD = np.array(skyfilesL_MJD)
starfiles = sorted(obsfilesL)
#starfiles = [f for f in starfiles if 'STD' in f]
print('Starfiles:', starfiles)
print('Skyfiles:', skyfilesL)


for ifile in starfiles:
    starfile = basedir + ifile
    print('\nWorking on file:', ifile)

    fh = fits.open(starfile)
    hdr = fh[0].header
    fh.close()
    mjd_obs = hdr['MJD-OBS']
    
    # associate the two sky files matching properties of the star file
    mjdiff = np.abs(skyfilesL_MJD - mjd_obs)
    # Sort skyfiles by ascending time distance to the starfile
    skyfilesL_sorted = [x for _,x in sorted(zip(mjdiff,skyfilesL))]
    
    for isky in skyfilesL_sorted:
        skyfile = basedir + isky
        fh = fits.open(skyfile)
        hdrsky = fh[0].header
        fh.close()
        
        keys_to_match = ['INSTRUME','ESO DET CHIP NAME','ESO DET SEQ1 DIT']
        imatch = 0
        for key in keys_to_match:
            if hdr[key] == hdrsky[key]:
                #print('Matching key:', key)
                imatch += 1
        if imatch == len(keys_to_match):
            print('Matching sky file:', skyfile)
            break


    ##########################################################

    bdata = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=verbose, plot=plotRaw)
    

    ##########################################################

    print('Shape of bdata:', bdata['INTERF']['data'].shape)

    if plotFringes:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
        # Plot the first frame of intf
        ax1.imshow(np.mean(bdata['INTERF']['data'], axis=0), cmap='viridis')
        ax1.set_title('average intf')

        plt.show()
        
    cfdata = op_get_corrflux(bdata, shiftfile, plot=plotCorr, verbose=verbose)

    wlen = cfdata['OI_WAVELENGTH']['EFF_WAVE']
    #print(wlen)

    #########################################################

    basename = os.path.basename(starfile)
    basen    = os.path.splitext(basename)[0]
    directory = cfdata['hdr']['DATE-OBS'].split('T')[0]+'_OIFITS/'
    if not os.path.exists(outdir+directory):
        os.makedirs(outdir+directory)
    chip = cfdata['hdr']['ESO DET CHIP NAME']
    if 'HAWAII' in chip:
        band = 'L'
    elif 'AQUARIUS' in chip:
        band = 'N'
    basen = directory+cfdata['hdr']['INSTRUME'][0:3]    + '_' +\
    cfdata['hdr']['DATE-OBS'].replace(':','-')          + '_' +\
    cfdata['hdr']['ESO OBS TARG NAME'].replace(' ','_') + '_' +\
    cfdata['hdr']['ESO DPR CATG'][0:3]                  + '_' +\
    band                                                + '_' +\
    cfdata['hdr']['ESO INS DIL ID']                     + '_' +\
    cfdata['hdr']['ESO INS BCD1 ID']                          +\
    cfdata['hdr']['ESO INS BCD2 ID']
        
    if 1:
        cfdata, vis2, mask = op_extract_simplevis2(cfdata, verbose=verbose, plot=0)
        
    if plotDsp:
        #print(mask)
        #print(~mask)
        notvis2 = np.ma.masked_array(np.ma.getdata(vis2), ~mask)
        allvis2 = np.ma.getdata(vis2)
        
        #print('Shape of vis2:', vis2.shape)
        #print('Shape of notvis2:', notvis2.shape)

        fig0, ax0 = plt.subplots(7, 1, figsize=(8, 8), sharex=1, sharey=1)
        #print('Shape of ax1:', ax0.shape)
        for i in np.arange(7):
            #print('i:', i)
            ax0[i].plot(wlen, allvis2[i,:], color='lightgray')
            ax0[i].plot(wlen, vis2[i,:], color=colors[i])
            ax0[i].set_ylabel(f'vis2 {i}')

        #print('Basename of starfile:', basen)
        plt.suptitle(r'Visibility as a function of $\lambda$, {basen}')
        plt.xlim(np.min(wlen), np.max(wlen))
        plt.ylim(-0.1, 1.1)
        #print(os.path.expanduser(bbasedir+f'{basen}_vis2.png'))
        plt.savefig(os.path.expanduser(outdir+f'{basen}_vis2.png'))
        #plt.show()

    #########################################################

    #cfdem = op_demodulate(cfdata, wlen, verbose=True, plot=False)
    cfdem = cfdata
    
    #print('Shape of cfdata:', cfdem['CF']['CF_demod'].shape)
    #cf = cfdem['CF']['CF_achr_phase_corr']
    cf   = cfdem['CF']['CF_Binned']
    nbframes = cf.shape[1]
    wlen = cfdata['OI_WAVELENGTH']['EFF_WAVE_Binned']
    #cf = cfdem['CF']['CF_demod']
    #cf = cfdem['CF']['CF_reord']
    sumcf = np.sum(cf, axis=1)
    avgcf = np.mean(cf, axis=1)
    #print('Shape of sumcf:', sumcf.shape)
    shp = sumcf.shape
    nbs = shp[0]
    
    if 1:
        fig1, ax1 = plt.subplots(nbs, 2, figsize=(8, 8), sharex=1, sharey=0)
        #print('Shape of ax1:', ax1.shape)
        for i in np.arange(nbs):
            #print('i:', i)
            ax1[i,0].plot(wlen,   np.abs(avgcf[i,:]), color=colors[i])
            for j in np.arange(nbframes):
                ax1[i,0].plot(wlen, np.abs(cf[i,j,:]), color=colors[i], alpha=0.1)
            if i == 0 and nbs == 7:
                ax1[i,0].set_ylabel(f'flux {i+1}')
            else:
                ax1[i,0].set_ylabel(f'corr. flux {i+1}')
            ax1[i,1].plot(wlen, np.angle(avgcf[i,:]), color=colors[i])
            for j in np.arange(nbframes):
                ax1[i,1].plot(wlen, np.angle(cf[i,j,:]), color=colors[i], alpha=0.1)
            ax1[i,1].set_ylabel(f'phase {i+1}')
        plt.suptitle('Sum CF data (1 exposure)')
        plt.tight_layout()
        plt.savefig(os.path.expanduser(outdir+f'{basen}_corrflux.png'))
        #plt.show()

    # iframe = 0
    # fig2, ax2 = plt.subplots(2, 6, figsize=(8, 4))
    # for i in np.arange(6): #+1 ????
    #     ax2[0,i].plot(wlen, np.abs(cf[i,iframe,:]), color=colors[i])
    #     ax2[1,i].plot(wlen, np.angle(cf[i,iframe,:]), color=colors[i])
    # plt.title(f'frame {iframe} of CF data')
    # plt.show()

    data, OPD_list = op_get_piston_fft(cfdem, verbose=True, plot=plotPist)
    #print('OPD:',OPD_list)

    #data, slopes = op_get_piston_slope(cfdem, verbose=True, plot=True)
    # print('Slopes:',slopes)

    #data, pistons = op_get_piston_chi2(data, 'fft', verbose=False, plot=True)
    # print('Pistons:',pistons)

    data = op_corr_piston(data, verbose=False, plot=plotPist)

    if plotPist:
        fig, ax = plt.subplots(7, 2, figsize=(8, 8))
        fig.suptitle('Piston corrected phase')
        for i_base in range(7):
            for i_frame in range(6):
                ax[i_base,0].plot(wlen, np.angle(data['CF']['CF_Binned'][i_base, i_frame]), color=colors[i_base])
                ax[i_base,1].plot(wlen, np.angle(data['CF']['CF_piston_corr'][i_base, i_frame]), color=colors[i_base])
        plt.show()
    

    #########################################################
    outfilename = os.path.expanduser(outdir+f'{basen}_corrflux_oi.fits')
    hdr = cfdem['hdr']
    oiwavelength = op_gen_oiwavelength(cfdem, verbose=verbose)
    oitarget     = op_gen_oitarget(cfdem, verbose=True, plot=False)
    oirray       = op_gen_oiarray(cfdem, verbose=True, plot=False)
    oivis        = op_gen_oivis(cfdem, cfin='CF_piston_corr', verbose=verbose, plot=False)
    #oivis2       = op_gen_oivis2(cfdem, v2in='simplevis2', verbose=verbose, plot=False)
    oivis2=None
    op_write_oifits(outfilename, hdr, oiwavelength, oirray, oitarget, oivis, oivis2=oivis2, oit3=None)

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