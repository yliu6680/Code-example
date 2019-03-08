#!/bin/bash -l

#$ -S /bin/bash

#$ -P camplab

#$ -cwd

#$ -j y

#$ -o analysis4k50-90.log

#$ -N analysis4k50-90

#$ -l h_rt=5:00:00

#$ -l mem_free=5g

#$ -pe omp 8

source ~/.bashrc
module load R/3.4.0
module load gcc/5.1.0
module load mpfr/3.1.2

R --no-save < analysis4k50-90.R &> analysis4k50-90.Rout
