#!/bin/bash
# Builds PETSc 3.25.5 (CPU, 64-bit indices) + HDF5 1.14.6 for SuperfluidGPU on
# bwUniCluster 3.0 and installs it to $HOME/opt/petsc-3.25.5-gnu14.2-int64.
# PETSc builds its own HDF5 because the lib/hdf5 modules ship no hdf5.pc,
# which CMakeLists.txt needs.
#
# Fetch the tarballs on a login node first (compute nodes may be offline):
#   see $PKGS below; then  sbatch build-petsc-cpu.sh
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH -J petsc-cpu-build
#SBATCH -o build-petsc-cpu-%j.log

source "$(dirname "$(readlink -f "$0")")/modules.sh" 2>/dev/null \
  || source "$SLURM_SUBMIT_DIR/modules.sh"
module list 2>&1

set -euo pipefail

WS=/pfs/work9/workspace/scratch/hd_pn277-super
PREFIX=$HOME/opt/petsc-3.25.5-gnu14.2-int64
export PETSC_DIR=$WS/petsc-build/petsc-3.25.5
export PETSC_ARCH=arch-bwuni3-cpu-int64
PKGS=$WS/petsc-build/pkgs
cd $PETSC_DIR

echo "== configure  $(date)"
./configure --prefix=$PREFIX \
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
  --with-64-bit-indices --with-debugging=0 \
  --with-cuda=0 \
  --download-f2cblaslapack=$PKGS/f2cblaslapack-3.8.0.q2.tar.gz \
  --download-hdf5=$PKGS/hdf5-1.14.6.tar.gz \
  --with-make-np=${SLURM_CPUS_PER_TASK:-8} \
  COPTFLAGS="-O3 -march=native" CXXOPTFLAGS="-O3 -march=native"

echo "== make all  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

echo "== make install  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH install

echo "== make check (installed)  $(date)"
make PETSC_DIR=$PREFIX PETSC_ARCH="" check || echo "make check FAILED"

echo "== done  $(date)"
