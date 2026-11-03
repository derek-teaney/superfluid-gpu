# GPU (H100) environment for building/running SuperPions on bwUniCluster 3.0:
#   source platforms/bwuni3/setupbwuni3-gpu.sh
#   cmake -B build-gpu -DSFGPU_KOKKOS=ON && cmake --build build-gpu -j
# For CPU-only builds use setupbwuni3.sh instead.
source "$(dirname "${BASH_SOURCE[0]}")/modules.sh"
module load devel/cuda/12.8
export PETSC_DIR=$HOME/opt/petsc-3.25.5-gnu14.2-cuda-int64
export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$PETSC_DIR:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH
