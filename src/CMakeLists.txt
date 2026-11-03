# Create your small core library
add_library(modelg
    measurer.cxx
    measurer_output.cxx
    Stepper.cxx
    NoiseGenerator.cxx
    initialize.cxx
)

# Attach all headers and external libraries to the core library
# PUBLIC means anything that links to modelg also gets these dependencies
target_include_directories(modelg PUBLIC
    ${CMAKE_SOURCE_DIR}/src # Your local .h files
    ${PETSC_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
)

target_link_options(modelg PUBLIC
    ${PETSC_LDFLAGS}
    ${HDF5_LDFLAGS}
    ${FFTW_LDFLAGS}
)

target_link_libraries(modelg PUBLIC
    ${PETSC_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${FFTW_LIBRARIES}
)


# Build SuperPions and link it to the core library
add_executable(SuperPions.exe SuperPions.cxx)
target_link_libraries(SuperPions.exe PRIVATE modelg)

if(SFGPU_KOKKOS)
    target_link_libraries(modelg PUBLIC Kokkos::kokkos)
    target_compile_definitions(modelg PUBLIC SFGPU_KOKKOS)
    # No fused multiply-add, on the host (the Cray CC wrapper adds
    # -march=znver3) or on the device (nvcc fuses by default). Every other
    # operation used by the stencils is correctly rounded on both, so ported
    # kernels can be checked bit-for-bit against the CPU build. The code is
    # bandwidth bound, so this costs essentially nothing.
    target_compile_options(modelg PUBLIC -ffp-contract=off)
    if(Kokkos_ENABLE_CUDA)
        target_compile_options(modelg PUBLIC -fmad=false)
        # cuFFT for GPU-accelerated 3D FFTs (topological charge measurement)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(modelg PUBLIC CUDA::cufft)
        target_compile_definitions(modelg PUBLIC SFGPU_CUFFT)
    endif()
endif()
