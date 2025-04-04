#!/usr/bin/env bash
#SBATCH --job-name=astrm
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=./logs/astrometry-out-%a.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=jules.scigliuto@oca.eu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

path_oifits=$1
path_output=$2

echo "tata"
filelist=($(find $path_oifits -type f -printf '%P\n' | grep "planet" | sort))
echo $filelist
file=${filelist[$SLURM_ARRAY_TASK_ID]}

# Dispatch the processing of each frame to the nodes (one frame per node)
echo "Fitting astrometry of $file..."
srun -N1 -n1 -c40 --exclusive python ./astrometry_multiprocessing_realimagpoly.py $file $path_oifits $path_output