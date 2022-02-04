#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J CycleGAN2
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set span if number of cores is more than 1
###BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request xGB of system-memory
#BSUB -R "rusage[mem=60GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o CycleGAN2_%J.out
#BSUB -e CycleGAN2_%J.err
# -- end of LSF options --



### Load modules
# module load python3
module load cuda

### Run setup
#sh setup.sh $run_dir || exit 1
# source ../VC-env/bin/activate
source ../Deep_voice_conversion/StarGAN/run_scripts/StarGAN-env/bin/activate

# Train
# sh train.sh
# sh SMK_trains/train4.sh
# python preprocess_training.py --train_A_dir ../Deep_voice_conversion/data/SMK_original/hilde/ --train_B_dir ../Deep_voice_conversion/data/SMK_original/yangSMK/ --cache_folder ./hilde_cache/
# python train.py --validation_A_dir ./data/louise_train/ --output_A_dir ./converted_sound/louise_train --validation_B_dir ./data/yangSMK/ --output_B_dir ./converted_sound/yangSMK/ 
# python train.py --validation_A_dir ./data/louise/ --output_A_dir ./converted_sound/louise --validation_B_dir ./data/yangSMK/ --output_B_dir ./converted_sound/yangSMK/

python train.py --validation_A_dir ./data/hilde/ --output_A_dir ./converted_sound/hilde --validation_B_dir ./data/yangSMK/ --output_B_dir ./converted_sound/yangSMK/ \
    --logf0s_normalization ./hilde_cache/logf0s_normalization.npz --mcep_normalization ./hilde_cache/mcep_normalization.npz --coded_sps_A_norm ./hilde_cache/coded_sps_A_norm.pickle --coded_sps_B_norm ./hilde_cache/coded_sps_B_norm.pickle --model_checkpoint ./model_checkpoint_hilde/


