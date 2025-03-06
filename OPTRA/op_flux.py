#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Flux handling methods
# Author: fmillour
# Date: 01/07/2024
# Project: OPTRA
#
################################################################################


def op_extract_beams(rawdata):
    for key in rawdata['PHOT']:
        photi = rawdata['PHOT'][key]['data']
        # Do something with photi
        photishape = np.mean(photi,axis=0)
        photishape = photishape / np.mean(photishape)
        photivar   = np.mean(photi,axis=[1,2])
        print('Photi:', np.shape(photi))
        

def op_compute_kappa():
    here_do_domething

def op_apply_kappa():
    here_do_domething

def op_filter_beams():
    here_do_domething

def op_compute_flux_fac():
    here_do_domething
