# Environment for building/running SuperPions on bwUniCluster 3.0:
#   source platforms/bwuni3/setupbwuni3.sh
source "$(dirname "${BASH_SOURCE[0]}")/modules.sh"
export PETSC_DIR=$HOME/opt/petsc-3.25.5-gnu14.2-int64
export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH
