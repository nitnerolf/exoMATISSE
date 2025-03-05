#!/usr/bin/env bash

# Define the directories of the input OIFITS and the output results
path_oifits=/data/home/jscigliuto/Pipeline/corrPhase0_MACAO/
path_output=/data/home/jscigliuto/Pipeline/Result_betPic0_MACAO/

# Load the Python distribution
echo 'Loading Python...'
module purge
module load intel-oneapi/2023.1.0
module load anaconda3/2022.10/oneapi-2023.1.0
set -x

# Create the output directories
echo 'Creating directories...'
rm -r $path_output
mkdir $path_output
mkdir $path_output/astrometry_fits_files

# Compute the errors and stellar averages
echo 'Computing errors and stellar averages...'
python prepare_astrometry.py $path_oifits $path_output

# List and count the frames to process
nfiles=`find "${path_oifits}" -type f -name "*planet*.fits" | wc -l`

# Dispatch the processing of each frame (one frame per node/job)
echo 'Sending the job array...'
sbatch --array=0-$((nfiles-1)) astrometry_multiprocessing_array.slurm $path_oifits $path_output