#!/bin/sh
# Run from ~/home/soldasim/boss/BOSS.jl
sbatch -p cpu --cpus-per-task=12 ./motor/script.sh $1
