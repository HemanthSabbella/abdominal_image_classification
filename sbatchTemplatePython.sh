#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=8GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=24:00:00        # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=project                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=cs701qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=adam.ho.2020@scis.smu.edu.sg,my.wang.2024@msc.smu.edu.sg,zhe.li.2024@phdcs.smu.edu.sg,shrsabbella.2024@phdcs.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=cs701Job     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.7.12 
module load CUDA/11.3.1

# Create a virtual environment
python3 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/myenv/bin/activate

# If you require any packages, install it as usual before the srun job submission.
# pip3 install numpy
pip3 install torch

# Submit your job to the cluster
srun --gres=gpu:1 python3 /common/home/projectgrps/CS701/CS701G8/gitcodes/CS701_G8/main.py