###########################################################################
# Low frequency filling... in python!
# Coded by Florentin Millour
###########################################################################

import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import os
from astropy.io import fits

###########################################################################

def loadallv2(files):
    VIS2    = []
    VIS2ERR = []
    U      = []
    V      = []
    WAVEL  = []
    for file in files:
        print("Loading file", file)
        with fits.open(file) as hdul:
            oi_vis2_hdus = [hdu for hdu in hdul if hdu.name == 'OI_VIS2']
            for v2hdu in oi_vis2_hdus:
                vis2 = v2hdu.data['VIS2DATA']
                vis2err = v2hdu.data['VIS2ERR']
                u = v2hdu.data['UCOORD']
                v = v2hdu.data['VCOORD']
                insname = v2hdu.header['INSNAME']
                oi_wavelength_hdus = [hdu for hdu in hdul if hdu.name == 'OI_WAVELENGTH']
                for whdu in oi_wavelength_hdus:
                    if whdu.header['INSNAME'] == insname:
                        wavel = whdu.data['EFF_WAVE']
                        break
                
                VIS2.append(vis2)
                VIS2ERR.append(vis2err)
                U.append(u)
                V.append(v)
                WAVEL.append(wavel)
            
    return VIS2, VIS2ERR, U, V, WAVEL
    
###########################################################################

def miral_slice_wlens(input_files, wlen_range=None):
  # Load OIFITS files
  sci = loadallv2(files)
  n_files = len(input_files)
  n_tab = len(sci)

  wlens = []
  bands = []
  for k_file in range(n_tab):
    wlens.extend(sci[k_file]['EFF_WAVE'])
    bands.extend(sci[k_file]['EFF_BAND'])
    print("nw", len(sci[k_file]['EFF_WAVE']))

  # Get the wavelengths
  wlens = np.unique(wlens)
  bands = np.array(bands)

  # Sort the wavelengths
  widx = np.argsort(wlens)
  wlens = wlens[widx]
  bands = bands[widx]

  # Remove wlens set to zero
  wnz = wlens != 0
  wlens = wlens[wnz]
  bands = bands[wnz]

  # Remove bands set to zero
  bnz = bands != 0
  wlens = wlens[bnz]
  bands = bands[bnz]

  min_wlen = np.min(wlens)
  max_wlen = np.max(wlens)

  print("*** Min wlen:", min_wlen, " max Wlen:", max_wlen)
  
  bandwidth = np.median(bands)

  interv = max_wlen - min_wlen
  n_wlen = round(interv / bandwidth)
  
  print("*** Interv:", interv, " nb Wlen:", n_wlen)

  if n_wlen <= 1:
    n_wlen = 2
  waves = np.linspace(min_wlen, max_wlen, n_wlen)
  bandes = np.full(n_wlen, interv / n_wlen)

  if wlen_range is not None:
    idx = (waves > wlen_range[0]) & (waves < wlen_range[1])
    waves = waves[idx]
    bandes = bandes[idx]
    min_wlen = np.min(waves)
    max_wlen = np.max(waves)
    n_wlen = len(waves)
  
  print("*****************************************")
  print("Min wlen:", min_wlen, "Max wlen:", max_wlen)
  print("Bandwidth:", bandwidth, "Nb wlen:", n_wlen)
  print("*****************************************")

  return waves, bandes

###########################################################################

def miral_lff(inputFiles, outputFile=None, binSize=1, plot=True, nLFF=None, useVisAmp=False, csym=1, freqMax=None, pse=0, maxBase=None, saveClos=False, errV2=None, errClos=None):
    print("Low Frequency Filling...")

    if plot:
        plt.figure(figsize=(6, 6))

    if outputFile is None:
        outputFile = inputFiles[0].replace('.fits', '_LFF.fits')

    nbFiles = len(inputFiles)

    # Load wavelength range of data and bandwidth
    waves, bands = miral_slice_wlens(inputFiles)

    nWlen = len(waves)

    if binSize != 1:
        nWlen = int(nWlen / binSize)
        if nWlen <= 1:
            nWlen = 1
            bands = max(waves) - min(waves)
            waves = np.mean(waves)
        else:
            bands = np.array([np.mean(bands) * binSize] * nWlen)
            waves = np.linspace(min(waves) + bands[0] / 2, max(waves) - bands[0] / 2, nWlen)

    nBands = len(waves)

    PAR = []
    for kBand in range(nBands):
        # Load OIFITS files
        sci = amplLoadOiDatas(inputFiles, wlenRange=[waves[kBand] - bands[kBand] / 2, waves[kBand] + bands[kBand] / 2], quiet=1)

        nFiles = len(sci)
        UF, VF, UF2, VF2, U, V, V2, V2E, VIS, VISE, W = [], [], [], [], [], [], [], [], [], [], []
        minR = 1e99

        # Choose whether using visibilities or squared visibilities
        for kF in range(nFiles):
            v2 = sci[kF].VIS2DATA
            if v2 is not None:
                U = sci[kF].UCOORD
                V = sci[kF].VCOORD
                R = np.abs(U, V)
                minR = min(minR, min(R))
                W = sci[kF].EFF_WAVE
                UF2.append((U / W).flatten())
                VF2.append((V / W).flatten())

                v2 = v2.flatten()
                v2e = sci[kF].VIS2ERR.flatten()

                V2.append(v2)
                V2E.append(v2e)
            else:
                vis = sci[kF].VISAMP
                if vis is not None:
                    U = sci[kF].UCOORD
                    V = sci[kF].VCOORD
                    R = np.abs(U, V)
                    minR = min(minR, min(R))
                    W = sci[kF].EFF_WAVE

                    UF.append((U / W).flatten())
                    VF.append((V / W).flatten())

                    vis = vis.flatten()
                    viserr = sci[kF].VISAMPERR.flatten()

                    VIS.append(vis)
                    VISE.append(viserr)

        # Get data
        if V2:
            vis = V2
            visErr = V2E
            UF = UF2
            VF = VF2
        if useVisAmp:
            vis = np.sign(VIS) * np.square(VIS)
            visErr = 2 * VISE * VIS

        # Get radius
        u = UF
        if nLFF is None:
            nLFF = len(u)
        v = VF
        uv = np.array([u, v])
        r = np.abs(u, v)
        theta = np.arctan2(v, u)

        if plot:
            plt.errorbar(r, np.sqrt(vis), yerr=visErr / (2 * np.sqrt(vis)), fmt='o', color='black')
            plt.xlabel("Frequency")
            plt.ylabel("Visibility")

        # Threshold for data cut
        minV = 0.5
        if freqMax is None:
            lf = np.where((r < max(r) / 10) & ((vis > minV) | (visErr < 0.1) & (r < 1.15 * min(r)) & (vis > 0)))
        else:
            lf = np.where(r < freqMax)

        if len(lf[0]) < 2:
            lf = np.argsort(r)[:2]

        vLFF = vis[lf]
        veLFF = visErr[lf]
        rLFF = r[lf]
        uvLFF = uv[:, lf]

        vavg = np.mean(vLFF)
        ravg = np.mean(rLFF)
        if plot:
            plt.errorbar(rLFF, np.sqrt(vLFF), yerr=veLFF / (2 * np.sqrt(vLFF)), fmt='o', color='red')
            plt.plot(ravg, np.sqrt(vavg), 'bo')
            plt.ylim(0, 1)

        if csym == 1 or len(vLFF) < 3:
            param = [np.sqrt(2 - vavg) / (ravg + (ravg == 0))]
            line = np.transpose(UVLine(0, (ravg + (ravg == 0)), 0, 100))

            res = lmfit(lff_visibility_round, uvLFF, param, np.power(vLFF, 4), 1. / np.square(veLFF) / np.power(rLFF, 4))
            if plot:
                plt.plot(np.abs(line[:, 0], line[:, 1]), np.sqrt(lff_visibility_round(line, param)), 'b-')
                plt.title(f"Low Frequency Filling...\n{waves[kBand] * 1e6} microns")
        else:
            siz = np.power(1 - vavg, 1. / 2.) / ravg
            param = [siz, 0.9 * siz, 0.1]

            res = lmfit(lff_visibility_2D, uvLFF, param, vLFF, 1. / np.square(veLFF), itmax=20)
            if plot:
                angle = param[2]
                line = np.transpose(UVLine(0, 1.2 * ravg, (angle % (2 * np.pi)), 100))
                line2 = np.transpose(UVLine(0, 1.2 * ravg, ((angle % (2 * np.pi)) + np.pi / 2), 100))

                plt.plot(np.abs(line[:, 0], line[:, 1]), np.sqrt(lff_visibility_2D(line, param)), 'b-')
                plt.plot(np.abs(line2[:, 0], line2[:, 1]), np.sqrt(lff_visibility_2D(line2, param)), 'b-')
                plt.title(f"Low Frequency Filling...\n{waves[kBand] * 1e6} microns")

        rsort = np.sort(r)
        vrmin = vis[rsort[:min(10, len(rsort))]]

        PAR.append(param)

    if maxBase is None:
        maxBase = minR * 0.5

    # Store the result in a new oifits file
    data = amplOiData()
    data.EFF_WAVE = waves
    data.EFF_BAND = bands
    data.hdr = sci[0].hdr

    # Set the UV coordinates of data as random in a circle
    nObs = nLFF
    nWlen = len(waves)

    if False:
        R = maxBase * np.random.random(nObs)
        T = 2 * np.pi * (np.random.random(nObs) - 0.5)
        wr = np.where(R > maxBase)
        nwr = len(wr[0])
        while nwr > 0:
            R[wr] = maxBase * np.random.random(nwr)
            T[wr] = 2 * np.pi * (np.random.random(nwr) - 0.5)
            wr = np.where(R > maxBase)
            nwr = len(wr[0])
        U = R * np.cos(T)
        V = R * np.sin(T)
    else:
        U = 2 * maxBase * (np.random.random(nObs) - 0.5)
        V = 2 * maxBase * (np.random.random(nObs) - 0.5)
        R = np.abs(U, V)
        wr = np.where(R > maxBase)
        nwr = len(wr[0])
        while nwr > 0:
            U[wr] = 2 * maxBase * (np.random.random(nwr) - 0.5)
            V[wr] = 2 * maxBase * (np.random.random(nwr) - 0.5)
            R = np.abs(U, V)
            wr = np.where(R > maxBase)
            nwr = len(wr[0])

    UV = np.array([U, V])
    data.UCOORD = U
    data.VCOORD = V

    VIS2 = []
    for kBand in range(nBands):
        if csym == 1 or len(PAR[kBand]) == 1:
            visib2 = lff_visibility_round(UV / waves[kBand], PAR[kBand])
        else:
            visib2 = lff_visibility_2D(UV / waves[kBand], PAR[kBand])
        VIS2.append(visib2)
    VIS2 = np.array(VIS2)
    VISE = np.median(veLFF) / np.median(vLFF) * np.sqrt(VIS2) * R / max(R)
    VIS2E = np.median(veLFF) / np.median(vLFF) * VIS2 * R / max(R)

    if plot:
        R = np.abs(U[:, None] / waves[None, :], V[:, None] / waves[None, :])
        plt.figure()
        plt.plot(R, np.sqrt(VIS2), 'o')

    # Compute squared visibilities for the given UV coordinates
    data.VIS2DATA = VIS2.T
    data.VIS2ERR = VIS2E.T
    data.STA_INDEX = np.array([[1, 2]] * VIS2.shape[1])
    data.FLAG = np.zeros(VIS2.T.shape)
    data.TIME = np.zeros(VIS2.shape[1])

    # Save data
    print("Saving LFF data")
    amplSaveOiData(outputFile, data)
