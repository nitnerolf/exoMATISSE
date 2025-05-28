#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
#
# Instrument-specific parameters for the OPTRA project.
# Author: fmillour
# Date: 04/12/2024
# Project: OPTRA
#
################################################################################
import numpy as np
#################################################
# Instrument-specific parameters for VINCI
#################################################
op_VINCI     = {'name':        'VINCI',
                'ntel': 2, 
                'recombination': 'coaxial_pairwise'}

#################################################
# Instrument-specific parameters for MIDI
#################################################
op_MIDI      = {'name':        'MIDI',
                'ntel': 2, 
                'recombination': 'coaxial_pairwise'}

#################################################
# Instrument-specific parameters for AMBER
#################################################
op_AMBER     = {'name':        'AMBER',
                'ntel': 3, 
                'recombination': 'multiaxial_allinone'}

#################################################
# Instrument-specific parameters for PIONIER
#################################################
op_PIONIER   = {'name':        'MATISSE_L',
                'ntel': 4, 
                'recombination': 'multiaxial_pairwise'}

#################################################
# Instrument-specific parameters for MATISSE
#################################################


op_MATISSE_L = {'name':        'MATISSE_L',
                'band':        'L',
                'ntel':        4,
                'recombination': 'multiaxial_allinone',
                'interfringe': 17.88*2.75*2*0.85, # in D/lambda
                'peakwd':      0.9,
                'scrP':        [1,2,4,3],
                'scrB':        [[2,3],[0,1],[1,2],[1,3],[0,2],[0,3]],# See MATISSE document: MATISSE-TP-TN-003
                'bcd_offset':  [1,2,3,4,5,6],
                'ron':         15}

op_MATISSE_N = {'name':        'MATISSE_N',
                'band':        'N',
                'ntel': 4, 
                'recombination': 'multiaxial_allinone',
                'interfringe': 17.88*2.75*2*0.85, # in D/lambda
                'peakwd':      0.9,
                'scrP':[3,4,2,1], 
                'scrB': [[3,2],[1,0],[2,1],[3,1],[2,0],[3,0]],# See MATISSE document: MATISSE-TP-TN-003
                'bcd_offset':[1,2,3,4,5,6],
                'ron':75}

#################################################
# Instrument-specific parameters for GRAVITY
#################################################
op_GRAVITY   = {'name':        'GRAVITY',
                'ntel': 4,
                'recombination': 'multiaxial_pairwise'}


#print(op_MATISSE_L)