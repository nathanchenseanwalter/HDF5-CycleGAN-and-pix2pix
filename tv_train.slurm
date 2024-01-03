#!/bin/bash
#SBATCH --job-name=1_tv_synth_p2p          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=nathaniel@princeton.edu

module purge
module load anaconda3/2023.9
conda activate torch-env

python train.py --dataroot tv_synth.h5 --name tv_synth_mdl --model pix2pix --direction BtoA --dataset_mode hdf5 --gpu_ids 0 --display_id 0 --input_nc 1 --output_nc 1

# ,1,2,3 --batch_size 4 --norm instance