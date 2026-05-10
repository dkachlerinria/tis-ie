#!/bin/bash

# This script should be run on your local laptop before git pushing.
# It marks all cluster scripts as executable in Git so they have the 
# correct permissions when pulled on the cluster.

echo "Setting executable bit for cluster scripts in Git index..."

# Cluster OAR scripts
git update-index --chmod=+x runs/cluster/select_embedding_cluster.sh
git update-index --chmod=+x runs/cluster/select_influcoder_cluster.sh
git update-index --chmod=+x runs/cluster/select_iprox_cluster.sh
git update-index --chmod=+x runs/cluster/select_less_cluster.sh
git update-index --chmod=+x runs/cluster/select_logra_cluster.sh
git update-index --chmod=+x runs/cluster/select_random_cluster.sh

# Launcher script
git update-index --chmod=+x runs/cluster/run_select_method_cluster.sh

echo "Done. You can now commit and push."
