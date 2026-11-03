#!/bin/bash
# Builds PETSc 3.25.5 + Kokkos/Kokkos-Kernels 5.1.0 (CUDA + OpenMP) + HDF5
# 1.14.6 for SuperfluidGPU on the bwUniCluster 3.0 H100 nodes (CUDA arch 90)
# and installs it to $HOME/opt/petsc-3.25.5-gnu14.2-cuda-int64.
# Runs on a GPU node so `make check` exercises CUDA/Kokkos on a real H100.
# For A100 nodes (gpu_a100_il) use --with-cuda-arch=80 and another PREFIX.
#
# Fetch the tarballs in $PKGS on a login node first, then
#   sbatch build-petsc-gpu.sh
#
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=96G
#SBATCH -J petsc-gpu-build
#SBATCH -o build-petsc-gpu-%j.log

source "$SLURM_SUBMIT_DIR/modules.sh"
# OpenMPI 5.0-gnu-14.2 is built CUDA-aware against this CUDA
module load devel/cuda/12.8
module list 2>&1

# After the module loads: Lmod scripts are not safe under set -u
set -euo pipefail
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi -L

WS=/pfs/work9/workspace/scratch/hd_pn277-super
PREFIX=$HOME/opt/petsc-3.25.5-gnu14.2-cuda-int64
export PETSC_DIR=$WS/petsc-build/petsc-3.25.5
export PETSC_ARCH=arch-bwuni3-h100-int64
# Tarballs fetched on a login node, so the build needs no network access
PKGS=$WS/petsc-build/pkgs
cd $PETSC_DIR

echo "== configure  $(date)"
./configure --prefix=$PREFIX \
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
  --with-64-bit-indices --with-debugging=0 \
  --with-cuda --with-cuda-dir=$CUDA_HOME --with-cuda-arch=90 \
  --with-openmp \
  --download-kokkos=$PKGS/kokkos-5.1.0.tar.gz \
  --download-kokkos-kernels=$PKGS/kokkos-kernels-5.1.0.tar.gz \
  --download-f2cblaslapack=$PKGS/f2cblaslapack-3.8.0.q2.tar.gz \
  --download-hdf5=$PKGS/hdf5-1.14.6.tar.gz \
  --with-make-np=${SLURM_CPUS_PER_TASK:-8} \
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3 CUDAOPTFLAGS=-O3

echo "== make all  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

echo "== make install  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH install

echo "== make check (installed)  $(date)"
make PETSC_DIR=$PREFIX PETSC_ARCH="" check || echo "make check FAILED"

echo "== done  $(date)"
