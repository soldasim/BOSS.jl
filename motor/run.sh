#!/bin/sh
# Run from ~/home/soldasim/boss/BOSS.jl
sbatch -p cpuextralong --cpus-per-task=12 ./motor/script.sh
