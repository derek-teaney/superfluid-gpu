#!/bin/bash

# Usage: ./build.sh [-c|--cpu] [-g|--gpu] [-d|--debug] [-j N] [-h]

# --- default ---
TARGET=gpu              # gpu or cpu
BUILD_TYPE=Release      # release or debug
JOBS=4
CLEAN=0

# routine printing how to use
usage() {
    echo "Usage: $0 [-c|--cpu] [-g|--gpu] [-d|--debug] [-j N] [--clean]"
    echo "  default: $TARGET, $BUILD_TYPE, -j $JOBS"
}

# ---- parse options ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cpu)   TARGET=cpu ;;
        -g|--gpu)   TARGET=gpu ;;
        -d|--debug) BUILD_TYPE=Debug ;;
        -j)         JOBS="$2"; shift ;;   # option that takes a value
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# ---- choose settings ----
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# gpu 
if [[ $TARGET == gpu ]]; then
    SETUP=$ROOT/platforms/bwuni3/setupbwuni3-gpu.sh
    BUILD_DIR=$ROOT/build-gpu
    EXTRA=(-DSFGPU_KOKKOS=ON)
# cpu
else
    SETUP=$ROOT/platforms/bwuni3/setupbwuni3.sh
    BUILD_DIR=$ROOT/build-cpu
    EXTRA=()
fi

# ---- environment (source it BEFORE set -u: Lmod isn't -u safe) ----
source "$SETUP"
set -euo pipefail

# --- build ---
echo "Building $TARGET ($BUILD_TYPE) in $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "${EXTRA[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"
