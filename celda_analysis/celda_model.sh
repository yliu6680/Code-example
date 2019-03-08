#!/bin/bash -l

#$ -S /bin/bash

#$ -P camplab

#$ -cwd

#$ -j y

#$ -o model8k_filterbyseurat.log

#$ -N model8k_filterbyseurat

#$ -l h_rt=50:00:00

#$ -l mem_free=5g

#$ -pe omp 8

source ~/.bashrc
module load R/3.4.0
module load gcc/5.1.0
module load mpfr/3.1.2

R --no-save < model8k_filterbyseurat.R &> model8k_filterbyseurat.Rout
