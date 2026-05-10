#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p gpu-16GB AND gpu_compute_capability_major>=5
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

# Load mamba
module load mamba

# Install the conda env if "tis" doesn't exist
if ! mamba env list | grep -q "^tis "; then
    echo "Creating 'tis' environment..."
    mamba create --yes -n tis -c conda-forge -c nvidia python=3.12 cuda-toolkit=12.6
    source activate tis
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "'tis' environment already exists."
    source activate tis
fi

# Run the selection pipeline
echo "Running select_random.sh..."
bash runs/select_random.sh
