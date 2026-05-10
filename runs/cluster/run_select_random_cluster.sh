#!/bin/bash

# Make sure the OAR script is executable
chmod u+x runs/cluster/select_random_cluster.sh

# Submit the job
echo "Submitting select_random_cluster.sh to OAR..."
oarsub -S ./runs/cluster/select_random_cluster.sh
