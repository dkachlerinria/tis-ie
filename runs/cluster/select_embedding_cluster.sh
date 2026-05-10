#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p gpu-16GB AND gpu_compute_capability_major>=7
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 

# Load mamba and initialize shell integration
module load mamba
source /etc/profile.d/conda.sh 2>/dev/null || source /etc/profile.d/mamba.sh 2>/dev/null || source ~/.bashrc 2>/dev/null

# Install the conda env if "tis" doesn't exist
if ! mamba env list | grep -q "^tis "; then
    echo "Creating 'tis' environment..."
    mamba create --yes -n tis -c conda-forge -c nvidia python=3.12 cuda-toolkit=12.6
    mamba activate tis
    echo "Installing requirements..."
    python3 -m pip install -r requirements.txt
else
    echo "'tis' environment already exists."
    mamba activate tis
fi

# Prevent system packages from leaking into the environment
export PYTHONPATH=""
export PYTHONNOUSERSITE=1

# Verify the environment
echo "Using python from: $(which python3)"
python3 --version

# Run the selection pipeline
echo "Running select_embedding.sh..."
bash runs/select_embedding.sh
