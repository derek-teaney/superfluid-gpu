# Building on a new machine / cluster

Worked examples: `platforms/perlmutter/` (NERSC, Cray) and `platforms/bwuni3/`
(bwUniCluster 3.0, plain Lmod + OpenMPI).

## What the build actually needs

`CMakeLists.txt` finds every dependency through **pkg-config**, so the whole
job is: make these three `.pc` files findable via `PKG_CONFIG_PATH`.

| pkg-config name | Provided by | Notes |
|---|---|---|
| `PETSc` | PETSc install (`$PETSC_DIR/lib/pkgconfig/PETSc.pc`) | 64-bit indices, HDF5 enabled |
| `hdf5`  | HDF5 install | **parallel** HDF5, same MPI as PETSc |
| `fftw3` | FFTW3 (serial is enough) | often a module, or the OS package |

Plus: CMake ≥ 3.15, a C++17 compiler (C++20 for the Kokkos/GPU build), MPI, git.

The C/C++ compilers are taken from `PETSc.pc` (`ccompiler`/`cxxcompiler`)
before `project()`, so the code is always built with the same MPI wrappers
PETSc was built with. You don't set compilers yourself.

## Step by step

1. **Survey the machine.**
   ```bash
   module avail            # compilers, MPI, hdf5, fftw, petsc, cuda
   pkg-config --modversion PETSc hdf5 fftw3
   nproc; echo $SLURM_JOB_ID   # login nodes are often limited to 1-few cores
   ```
   Pick **one** toolchain (compiler + MPI) and use the modules built for it
   everywhere.

2. **Check each module actually ships a `.pc` file.** Load it, then
   `pkg-config --modversion <name>`. Cluster HDF5 modules often do *not*
   (bwUniCluster doesn't). If one is missing, either let PETSc build that
   package (below) or write a small `.pc` file yourself.

3. **Get PETSc.** Use a site module/spack install if one exists with 64-bit
   indices + HDF5. Otherwise build it from source. Copy
   `platforms/bwuni3/build-petsc-cpu.sh` and adapt it:
   ```bash
   ./configure --prefix=$PREFIX \
     --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
     --with-64-bit-indices --with-debugging=0 \
     --download-hdf5 --download-f2cblaslapack \
     COPTFLAGS="-O3 -march=native" CXXOPTFLAGS="-O3 -march=native"
   make all && make install && make check
   ```
   - `--download-hdf5` gives an HDF5 built with the same MPI and installs
     `hdf5.pc` next to `PETSc.pc`. That solves the missing-`.pc` problem.
   - If the site has a good BLAS/LAPACK (MKL, OpenBLAS), use `--with-blaslapack-dir=`
     instead of f2cblaslapack. Performance hardly matters here (stencil code).
   - **Offline compute nodes:** run `./configure ... --with-packages-download-dir=../pkgs`
     on the login node. It prints the tarball URLs. Download them, then
     pass `--download-hdf5=/path/hdf5-x.tar.gz` etc. in the batch job.
   - **Build on a compute node** (`sbatch`, ~30-60 min with 32-48 cores);
     login nodes are usually core-limited.
   - **Install somewhere durable** (`$HOME/opt`, a project directory) rather than
     scratch/workspaces that expire.

4. **Write a setup script** `platforms/<machine>/setup<machine>.sh` that loads
   the modules and exports the paths:
   ```bash
   module load <compiler> <mpi> <fftw>
   export PETSC_DIR=$HOME/opt/petsc-3.25.5-...-int64
   export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
   export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH
   ```
   Source the same script in job scripts before running.

5. **Build the code.**
   ```bash
   source platforms/<machine>/setup<machine>.sh
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ```
   Check the configure output. The "Shell override: Set C Compiler to ..." line
   must show PETSc's MPI wrapper.

## GPU (Kokkos) build

The same pattern, but PETSc must be configured with Kokkos:
`--with-cuda --download-kokkos --download-kokkos-kernels` (see
`platforms/perlmutter/build-petsc-gpu.sh`, or `platforms/bwuni3/build-petsc-gpu.sh`
for a plain OpenMPI + CUDA module cluster). Build PETSc on a GPU node with the
CUDA module matching the MPI's GPU support, and add
`export CMAKE_PREFIX_PATH=$PETSC_DIR:$CMAKE_PREFIX_PATH` so `find_package(Kokkos)`
works. Then:
```bash
cmake -B build-gpu -DSFGPU_KOKKOS=ON && cmake --build build-gpu -j
```

## Common pitfalls

- Mixing toolchains (PETSc built with one MPI, HDF5/FFTW from another): link
  errors or runtime crashes. Keep everything on one compiler+MPI.
- Serial HDF5: `petscviewerhdf5.h` needs parallel HDF5.
- Stale CMake cache after changing modules: delete the build directory.
- Missing `LD_LIBRARY_PATH` at run time: "cannot open shared object libpetsc.so".
