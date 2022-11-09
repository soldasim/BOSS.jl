#!/bin/sh

ml Julia/1.8.0-linux-x86_64
ml Python/3.9.6-GCCcore-11.2.0
ml SciPy-bundle/2021.10-foss-2021b
ml matplotlib/3.4.3-foss-2021b

DATE=$(date +"%Y-%m-%d")
ID="${DATE}-${RANDOM}"
OUT="./motor/logs/log-${ID}.txt"
METHOD=$1

export JULIA_NUM_THREADS=12
julia ./motor/script.jl $ID $METHOD 1>$OUT 2>$OUT
