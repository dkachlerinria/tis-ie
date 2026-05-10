# This script should be run on your local Windows machine before git pushing.
# It marks all cluster scripts as executable in Git.

Write-Host "Setting executable bit for cluster scripts in Git index..." -ForegroundColor Cyan

git update-index --chmod=+x runs/cluster/select_embedding_cluster.sh
git update-index --chmod=+x runs/cluster/select_influcoder_cluster.sh
git update-index --chmod=+x runs/cluster/select_iprox_cluster.sh
git update-index --chmod=+x runs/cluster/select_less_cluster.sh
git update-index --chmod=+x runs/cluster/select_logra_cluster.sh
git update-index --chmod=+x runs/cluster/select_random_cluster.sh
git update-index --chmod=+x runs/cluster/run_select_method_cluster.sh

Write-Host "Done. You can now commit and push." -ForegroundColor Green
