import numpy as np
import os
from astropy.io import fits
from pathlib import Path

###############

mode = 'average'

path = '/data/home/jscigliuto/Pipeline/HD72946B/corrPhase_without_rmnrec/'

path_frames = path + '/corrected_data/'
path_frames_stacked = '/data/home/jscigliuto/Pipeline/HD72946B' + '/corrPhase_stacked_Mathis/'


#####################


def wrap(angle):
    return np.angle(np.exp(1j*angle))


all_frames = sorted(os.listdir(path_frames))
all_ob_numbers = set(sorted([int(f[f.find('OB')+2:f.find('_exp')]) for f in all_frames if 'flagged' not in f]))

# Create output directory if necessary
if not os.path.isdir(path_frames_stacked):
    os.makedirs(path_frames_stacked)

for i_OB in all_ob_numbers:

    # Get all exposure numbers for this OB
    all_exp_numbers = set(sorted([int(f[f.find('exp')+3:f.find('_frame')]) for f in all_frames if ('OB' + str(i_OB) + '_') in f and 'flagged' not in f]))

    for i_exp in all_exp_numbers:

        # Get all frames
        all_frames_exp = sorted([f for f in all_frames if ('OB' + str(i_OB) + '_exp' + str(i_exp) + '_') in f and 'flagged' not in f])

        print('Stacking OB ' + str(i_OB) + ' exposure ' + str(i_exp) + ' with ' + str(len(all_frames_exp)) + ' frames.')

        ## Create the stacked OIFITS

        # Copy some extensions from the RAW_VIS2 files
        hdul_ref = fits.open(path_frames + all_frames_exp[0])
        
        # Primary extension (main header)
        hdu0 = fits.PrimaryHDU(header=hdul_ref[0].header)

        # Get common HDUs
        hdu_target = fits.BinTableHDU(name='OI_TARGET', data=hdul_ref['OI_TARGET'].data, header=hdul_ref['OI_TARGET'].header)
        hdu_array = fits.BinTableHDU(name='OI_ARRAY', data=hdul_ref['OI_ARRAY'].data, header=hdul_ref['OI_ARRAY'].header)
        hdu_wave = fits.BinTableHDU(name='OI_WAVELENGTH', data=hdul_ref['OI_WAVELENGTH'].data, header=hdul_ref['OI_WAVELENGTH'].header)
        n_wave = len(hdu_wave.data['EFF_WAVE'])

        if mode == 'stack':
            # visdata = np.array([], dtype=complex).reshape(0, n_wave)
            # viserr = np.array([], dtype=complex).reshape(0, n_wave)

            visamp = np.array([], dtype=complex).reshape(0, n_wave)
            visamperr = np.array([], dtype=complex).reshape(0, n_wave)
            visphi = np.array([], dtype=complex).reshape(0, n_wave)
            visphierr = np.array([], dtype=complex).reshape(0, n_wave)
            ucoord = []
            vcoord = []
            mjd = []
            inttime = []
            staindex = np.array([], dtype=int).reshape(0, 2)
            targetid = []

            t3_amp = np.array([]).reshape(0, n_wave)
            t3_phi = np.array([]).reshape(0, n_wave)
            t3_amp_err = np.array([]).reshape(0, n_wave)
            t3_phi_err = np.array([]).reshape(0, n_wave)
            u1coord = []
            u2coord = []
            v1coord = []
            v2coord = []
            mjd_t3 = []
            inttime_t3 = []
            staindex_t3 = np.array([], dtype=int).reshape(0, 3)
            targetid_t3 = []

            #exograv_visref = np.array([], dtype=complex).reshape(0, n_wave)

        elif mode == 'average':
            # visdata = np.zeros((len(all_frames_exp), 6, n_wave), dtype=complex)
            # viserr = np.zeros((len(all_frames_exp), 6, n_wave), dtype=complex)

            visamp = np.zeros((len(all_frames_exp), 6, n_wave)) #np.array([]).reshape(0, n_wave)
            visamperr = np.zeros((len(all_frames_exp), 6, n_wave))
            visphi = np.zeros((len(all_frames_exp), 6, n_wave))
            visphierr = np.zeros((len(all_frames_exp), 6, n_wave))

            ucoord = np.zeros((len(all_frames_exp), 6))
            vcoord = np.zeros((len(all_frames_exp), 6))
            mjd = np.zeros((len(all_frames_exp), 6))
            inttime = np.zeros((len(all_frames_exp), 6))
            staindex = np.zeros((len(all_frames_exp), 6, 2), dtype=int)
            targetid = np.zeros((len(all_frames_exp), 6), dtype=int)

            t3_amp = np.zeros((len(all_frames_exp), 4, n_wave))
            t3_phi = np.zeros((len(all_frames_exp), 4, n_wave))
            t3_amp_err = np.zeros((len(all_frames_exp), 4, n_wave))
            t3_phi_err = np.zeros((len(all_frames_exp), 4, n_wave))
            u1coord = np.zeros((len(all_frames_exp), 4))
            u2coord = np.zeros((len(all_frames_exp), 4))
            v1coord = np.zeros((len(all_frames_exp), 4))
            v2coord = np.zeros((len(all_frames_exp), 4))
            mjd_t3 = np.zeros((len(all_frames_exp), 4))
            inttime_t3 = np.zeros((len(all_frames_exp), 4))
            staindex_t3 = np.zeros((len(all_frames_exp), 4, 3), dtype=int)
            targetid_t3 = np.zeros((len(all_frames_exp), 4), dtype=int)

            #exograv_visref = np.zeros((len(all_frames_exp), 6, n_wave), dtype=complex)


        for i_frame, frame in enumerate(all_frames_exp):

            with fits.open(path_frames + frame) as hdul_frame:

                if mode == 'stack':
                    # visdata = np.concatenate([visdata, hdul_frame['OI_VIS'].data['VISDATA']])
                    # viserr = np.concatenate([viserr, hdul_frame['OI_VIS'].data['VISERR']])

                    visamp = np.concatenate([visamp, hdul_frame['OI_VIS'].data['VISAMP']])
                    visamperr = np.concatenate([visamperr, hdul_frame['OI_VIS'].data['VISAMPERR']])
                    visphi = np.concatenate([visphi, hdul_frame['OI_VIS'].data['VISPHI']])
                    visphierr = np.concatenate([visphierr, hdul_frame['OI_VIS'].data['VISPHIERR']])

                    ucoord = np.concatenate([ucoord, hdul_frame['OI_VIS'].data['UCOORD']])
                    vcoord = np.concatenate([vcoord, hdul_frame['OI_VIS'].data['VCOORD']])
                    mjd = np.concatenate([mjd, hdul_frame['OI_VIS'].data['MJD']])
                    inttime = np.concatenate([inttime, hdul_frame['OI_VIS'].data['INT_TIME']])
                    staindex = np.concatenate([staindex, hdul_frame['OI_VIS'].data['STA_INDEX']])
                    targetid = np.concatenate([targetid, hdul_frame['OI_VIS'].data['TARGET_ID']])

                    t3_amp = np.concatenate([t3_amp, hdul_frame['OI_T3'].data['T3AMP']])
                    t3_phi = np.concatenate([t3_phi, hdul_frame['OI_T3'].data['T3PHI']])
                    t3_amp_err = np.concatenate([t3_amp_err, hdul_frame['OI_T3'].data['T3AMPERR']])
                    t3_phi_err = np.concatenate([t3_phi_err, hdul_frame['OI_T3'].data['T3PHIERR']])
                    u1coord = np.concatenate([u1coord, hdul_frame['OI_T3'].data['U1COORD']])
                    u2coord = np.concatenate([u2coord, hdul_frame['OI_T3'].data['U2COORD']])
                    v1coord = np.concatenate([v1coord, hdul_frame['OI_T3'].data['V1COORD']])
                    v2coord = np.concatenate([v2coord, hdul_frame['OI_T3'].data['V2COORD']])
                    mjd_t3 = np.concatenate([mjd_t3, hdul_frame['OI_T3'].data['MJD']])
                    inttime_t3 = np.concatenate([inttime_t3, hdul_frame['OI_T3'].data['INT_TIME']])
                    staindex_t3 = np.concatenate([staindex_t3, hdul_frame['OI_T3'].data['STA_INDEX']])
                    targetid_t3 = np.concatenate([targetid_t3, hdul_frame['OI_T3'].data['TARGET_ID']])

                    #exograv_visref = np.concatenate([exograv_visref, hdul_frame['EXOGRAV_VISREF'].data['EXOGRAV_VISREF']])

                elif mode == 'average':
                    # visdata[i_frame] = hdul_frame['OI_VIS'].data['VISDATA']
                    # viserr[i_frame] = hdul_frame['OI_VIS'].data['VISERR']

                    visamp[i_frame] = hdul_frame['OI_VIS'].data['VISAMP']
                    visamperr[i_frame] = hdul_frame['OI_VIS'].data['VISAMPERR']
                    visphi[i_frame] = hdul_frame['OI_VIS'].data['VISPHI']
                    visphierr[i_frame] = hdul_frame['OI_VIS'].data['VISPHIERR']
                    
                    ucoord[i_frame] = hdul_frame['OI_VIS'].data['UCOORD']
                    vcoord[i_frame] = hdul_frame['OI_VIS'].data['VCOORD']
                    mjd[i_frame] = hdul_frame['OI_VIS'].data['MJD']
                    inttime[i_frame] = hdul_frame['OI_VIS'].data['INT_TIME']
                    staindex[i_frame] = hdul_frame['OI_VIS'].data['STA_INDEX']
                    targetid[i_frame] = hdul_frame['OI_VIS'].data['TARGET_ID']

                    t3_amp[i_frame] = hdul_frame['OI_T3'].data['T3AMP']
                    t3_phi[i_frame] = hdul_frame['OI_T3'].data['T3PHI']
                    t3_amp_err[i_frame] = hdul_frame['OI_T3'].data['T3AMPERR']
                    t3_phi_err[i_frame] = hdul_frame['OI_T3'].data['T3PHIERR']
                    u1coord[i_frame] = hdul_frame['OI_T3'].data['U1COORD']
                    u2coord[i_frame] = hdul_frame['OI_T3'].data['U2COORD']
                    v1coord[i_frame] = hdul_frame['OI_T3'].data['V1COORD']
                    v2coord[i_frame] = hdul_frame['OI_T3'].data['V2COORD']
                    mjd_t3[i_frame] = hdul_frame['OI_T3'].data['MJD']
                    inttime_t3[i_frame] = hdul_frame['OI_T3'].data['INT_TIME']
                    staindex_t3[i_frame] = hdul_frame['OI_T3'].data['STA_INDEX']
                    targetid_t3[i_frame] = hdul_frame['OI_T3'].data['TARGET_ID']

                    #exograv_visref[i_frame] = hdul_frame['EXOGRAV_VISREF'].data['EXOGRAV_VISREF']

                else:
                    raise ValueError('Unknown stacking mode: ' + mode)
                
        if mode == 'average':
            # visdata = np.mean(visdata, axis=0)
            # viserr = np.sum(viserr.real**2, axis=0) / np.sqrt(len(viserr)) + 1j * np.sum(viserr.imag**2, axis=0) / np.sqrt(len(viserr))

            #Complexify 
            visphi = np.deg2rad(visphi)
            visphierr = np.deg2rad(visphierr)
            cf_full = visamp * np.exp(1j*visphi)

            cf_full_real_err = np.sqrt((np.cos(visphi) * visamperr) ** 2 \
                                + (visamp * np.sin(visphi) * visphierr) ** 2)
            cf_full_imag_err = np.sqrt((np.sin(visphi) * visamperr) ** 2 \
                                + (visamp * np.cos(visphi) * visphierr) ** 2)
            #Average in the complex space 
            mean_cf_full = np.mean(np.real(cf_full), axis=0) + 1j * np.mean(np.imag(cf_full), axis=0)
            mean_cf_full_real_err = np.sqrt(np.sum(cf_full_real_err**2, axis=0)) / len(all_frames_exp)
            mean_cf_full_imag_err = np.sqrt(np.sum(cf_full_imag_err**2, axis=0)) / len(all_frames_exp)

            print('SIZE', mean_cf_full.shape, mean_cf_full_real_err.shape, mean_cf_full_imag_err.shape)

            #Go back to amp/phase space
            visamp = np.abs(mean_cf_full)
            visphi = np.angle(mean_cf_full)
            visamperr = np.sqrt((np.real(mean_cf_full) * mean_cf_full_real_err) ** 2 \
                                    + (np.imag(mean_cf_full) * mean_cf_full_imag_err) ** 2) / visamp
            visphierr = np.sqrt((np.imag(mean_cf_full) * mean_cf_full_real_err) ** 2 \
                                     + (np.real(mean_cf_full) * mean_cf_full_imag_err) ** 2) /visphi**2

            ucoord = np.mean(ucoord, axis=0)
            vcoord = np.mean(vcoord, axis=0)
            mjd = np.mean(mjd, axis=0)
            inttime = inttime[0]
            staindex = staindex[0]
            targetid = targetid[0]

            t3_amp = np.mean(t3_amp, axis=0)
            t3_phi = wrap(np.mean(t3_phi, axis=0))
            t3_amp_err = np.sum(t3_amp_err**2, axis=0) / np.sqrt(len(t3_amp_err))
            t3_phi_err = np.sum(t3_phi_err**2, axis=0) / np.sqrt(len(t3_phi_err))
            u1coord = np.mean(u1coord, axis=0)
            u2coord = np.mean(u2coord, axis=0)
            v1coord = np.mean(v1coord, axis=0)
            v2coord = np.mean(v2coord, axis=0)
            mjd_t3 = np.mean(mjd_t3, axis=0)
            inttime_t3 = inttime_t3[0]
            staindex_t3 = staindex_t3[0]
            targetid_t3 = targetid_t3[0]

            #exograv_visref = np.mean(exograv_visref, axis=0)

        # OI_VIS
        # visdata_col = fits.Column(name='VISDATA', array=visdata, format=f'{n_wave}M')
        # viserr_col = fits.Column(name='VISERR', array=viserr, format=f'{n_wave}M')

        visamp_col = fits.Column(name='VISAMP', array=visamp, format=f'{n_wave}D')
        visamperr_col = fits.Column(name='VISAMPERR', array=visamperr, format=f'{n_wave}D')
        visphi_col = fits.Column(name='VISPHI', array=np.rad2deg(visphi), unit='deg', format=f'{n_wave}D')
        visphierr_col = fits.Column(name='VISPHIERR', array=np.rad2deg(visphierr), unit='deg', format=f'{n_wave}D')
        ucoord_col = fits.Column(name='UCOORD', array=ucoord, unit='m', format='D')
        vcoord_col = fits.Column(name='VCOORD', array=vcoord, unit='m', format='D')
        mjd_col = fits.Column(name='MJD', array=mjd, format='D')
        inttime_col = fits.Column(name='INT_TIME', array=inttime, unit='s', format='D')
        staindex_col = fits.Column(name='STA_INDEX', array=staindex, format='2I')
        targetid_col = fits.Column(name='TARGET_ID', array=targetid, format='I')
        hdu_vis = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, visamp_col, visamperr_col, visphi_col, visphierr_col, ucoord_col, vcoord_col, staindex_col], name='OI_VIS', header=hdul_ref['OI_VIS'].header)

        # OI_T3
        t3amp_col = fits.Column(name='T3AMP', array=t3_amp, format=f'{n_wave}D')
        t3phi_col = fits.Column(name='T3PHI', array=t3_phi, unit='deg', format=f'{n_wave}D')
        t3amperr_col = fits.Column(name='T3AMPERR', array=t3_amp_err, format=f'{n_wave}D')
        t3phierr_col = fits.Column(name='T3PHIERR', array=t3_phi_err, unit='deg', format=f'{n_wave}D')
        u1coord_col = fits.Column(name='U1COORD', array=u1coord, unit='m', format='D')
        u2coord_col = fits.Column(name='U2COORD', array=u2coord, unit='m', format='D')
        v1coord_col = fits.Column(name='V1COORD', array=v1coord, unit='m', format='D')
        v2coord_col = fits.Column(name='V2COORD', array=v2coord, unit='m', format='D')
        mjd_col = fits.Column(name='MJD', array=mjd_t3, format='D')
        inttime_col = fits.Column(name='INT_TIME', array=inttime_t3, unit='s', format='D')
        staindex_t3_col = fits.Column(name='STA_INDEX', array=staindex_t3, format='3I')
        targetid_col = fits.Column(name='TARGET_ID', array=targetid_t3, format='I')
        hdu_t3 = fits.BinTableHDU.from_columns([targetid_col, mjd_col, inttime_col, t3amp_col, t3phi_col,
                                                t3amperr_col, t3phierr_col, u1coord_col, u2coord_col,
                                                v1coord_col, v2coord_col, staindex_t3_col], name='OI_T3', header=hdul_ref['OI_T3'].header)
            
        # EXOGRAV_VISREF
        #visref_col = fits.Column(name='EXOGRAV_VISREF', array=exograv_visref, format=f'{n_wave}M')  
        #hdu_visref = fits.BinTableHDU.from_columns([visref_col], name='EXOGRAV_VISREF', header=hdul_ref['EXOGRAV_VISREF'].header)

        # Update NDIT
        if mode == 'average':
            hdu0.header["HIERARCH ESO TPL NDIT OBJECT"] = 1
        elif mode == 'stack':
            hdu0.header["HIERARCH ESO TPL NDIT OBJECT"] = len(all_frames_exp)        

        # Stack in a HDU list and save
        hdul_stacked = fits.HDUList([hdu0, hdu_target, hdu_array, hdu_wave, hdu_vis, hdu_t3]) #, hdu_visref])
        date_obs = Path(all_frames_exp[0]).stem[:all_frames_exp[0].find('_OB')]
        target = Path(all_frames_exp[0]).stem[all_frames_exp[0].find(f'_frame1_')+8:all_frames_exp[0].find('.fits')]
        hdul_stacked.writeto(f"{path_frames_stacked}/{date_obs}_OB{i_OB}_exp{i_exp}_{target}.fits", overwrite=True)

        hdul_ref.close()
        hdul_stacked.close()