#!/bin/bash

# Launcher script for cluster selection jobs
# Usage: ./runs/cluster/run_select_method_cluster.sh <method>
# Example: ./runs/cluster/run_select_method_cluster.sh random

METHOD=$1

if [ -z "$METHOD" ]; then
    echo "Usage: $0 <method>"
    echo "Available methods: random, less, logra, iprox, influcoder, embedding"
    exit 1
fi

SCRIPT="./runs/cluster/select_${METHOD}_cluster.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script $SCRIPT not found."
    echo "Make sure the method name is correct."
    exit 1
fi

echo "Submitting $METHOD selection job to OAR..."
oarsub -S "$SCRIPT" -q production 
