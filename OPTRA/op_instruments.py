"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instrument-specific parameters for the OPTRA project.
Author: fmillour
Date: 04/12/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#################################################
# Instrument-specific parameters for VINCI
#################################################
op_VINCI     = {'ntel': 2}

#################################################
# Instrument-specific parameters for MIDI
#################################################
op_MIDI      = {'ntel': 2}

#################################################
# Instrument-specific parameters for AMBER
#################################################
op_AMBER     = {'ntel': 3}

#################################################
# Instrument-specific parameters for PIONIER
#################################################
op_PIONIER   = {'ntel': 4}

#################################################
# Instrument-specific parameters for MATISSE
#################################################
op_MATISSE_L = {'name':        'MATISSE_L',
                'ntel':        4,
                'recombination': 'multiaxial_allinone',
                'interfringe': 17.88*2.75*2*0.85, # in D/lambda
                'peakwd':      0.9,
                'scrP':        [1,2,4,3],
                'scrB':        [[2,3],[0,1],[1,2],[1,3],[0,2],[0,3]],
                'bcd_offset':  [1,2,3,4,5,6],
                'ron':         20}

op_MATISSE_N = {'ntel': 4, 'scrP':[3,4,2,1], 'scrB':[[4,3],[2,1],[3,2],[4,2],[3,1],[4,1]],
                'bcd_offset':[1,2,3,4,5,6],
                'ron':1000}

#################################################
# Instrument-specific parameters for GRAVITY
#################################################
op_GRAVITY   = {'name':        'GRAVITY',
                'ntel': 4}


#print(op_MATISSE_L)