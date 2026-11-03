#ifndef MEASURER
#define MEASURER

#include "ModelA.h"
#include "nvector.h"
#include <array>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef SFGPU_CUFFT
#include <cufft.h>
#endif

////////////////////////////////////////////////////////////////////////
// Class to compute the fourier transform of the 1d array. The fourier transform
// is defined as 1/N \sum_x e^{ikx} W(x) = W(k) where W(x) is the array of size
// N and W(k) is the complex array of size N/2 + 1
class measurer_fft {

private:
  // Create the storage space for the plan
  std::vector<double> in_ptr;
  std::vector<std::complex<double>> out_ptr;
  fftw_plan plan;

public:
  measurer_fft(const size_t N) : in_ptr(N, 0.), out_ptr(N / 2 + 1, 0.) {
    // Create the plane for the fourier transform
    plan = fftw_plan_dft_r2c_1d(
        N, in_ptr.data(), reinterpret_cast<fftw_complex *>(out_ptr.data()),
        FFTW_MEASURE);
  }
  ~measurer_fft() { fftw_destroy_plan(plan); }

  void execute_row(double *in, std::complex<double> *out) {
    for (size_t i = 0; i < in_ptr.size(); i++) {
      in_ptr[i] = in[i];
    }
    fftw_execute(plan);
    for (size_t i = 0; i < out_ptr.size(); i++) {
      out[i] = out_ptr[i] / static_cast<double>(in_ptr.size());
    }
  }

  // Compute the fourier transform of an array in
  void execute(nvector<double, 2> &in, nvector<std::complex<double>, 2> &out) {
    for (size_t i = 0; i < in.N[0]; i++) {
      execute_row(&in(i, 0), &out(i, 0));
    }
  }
};

////////////////////////////////////////////////////////////////////////
// Class to compute a real-to-complex 3D fourier transform of a scalar field
// on a periodic box.  The transform is unnormalized,
//
//   W(k) = \sum_x e^{-i k.x} W(x),
//
// and callers divide by n0*n1*n2 (see Measurer::radialSpectrum).  The
// dimensions are (n0, n1, n2) in row-major (C) order, so that the last index
// n2 is contiguous in memory.  The real input has size n0*n1*n2 and the
// complex output (r2c half spectrum) has size n0*n1*(n2/2+1).
//
// execute_host transforms host memory with FFTW. With SFGPU_CUFFT,
// execute_device transforms device memory with cuFFT and leaves the result on
// the device. The plans and buffers are made on first use, so a GPU run never
// allocates the FFTW arrays and vice versa.
class measurer_fft_3d {

private:
  size_t n0, n1, n2;
  size_t out_size;
  // FFTW (host)
  std::vector<double> in_ptr;
  std::vector<std::complex<double>> out_ptr;
  fftw_plan fftw_plan_handle = nullptr;
  fftw_plan fftw_inverse_plan = nullptr; // c2r on the same buffers
#ifdef SFGPU_CUFFT
  // cuFFT (device)
  cufftHandle cufft_plan;
  bool have_cufft = false;
  cufftDoubleComplex *d_out = nullptr;
  cufftHandle cufft_inverse_plan;
  bool have_cufft_inverse = false;
#endif

public:
  measurer_fft_3d(const size_t N0, const size_t N1, const size_t N2)
      : n0(N0), n1(N1), n2(N2), out_size(N0 * N1 * (N2 / 2 + 1)) {}

  ~measurer_fft_3d() {
    if (fftw_plan_handle) {
      fftw_destroy_plan(fftw_plan_handle);
    }
    if (fftw_inverse_plan) {
      fftw_destroy_plan(fftw_inverse_plan);
    }
#ifdef SFGPU_CUFFT
    if (have_cufft) {
      cufftDestroy(cufft_plan);
      cudaFree(d_out);
    }
    if (have_cufft_inverse) {
      cufftDestroy(cufft_inverse_plan);
    }
#endif
  }

  size_t size_in() const { return n0 * n1 * n2; }
  size_t size_out() const { return out_size; }

  // Transform the real volume `in` (host memory, size n0*n1*n2). Returns the
  // half spectrum in host memory, valid until the next call.
  const std::complex<double> *execute_host(const double *in) {
    if (!fftw_plan_handle) {
      in_ptr.assign(size_in(), 0.);
      out_ptr.assign(out_size, 0.);
      fftw_plan_handle = fftw_plan_dft_r2c_3d(
          static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
          in_ptr.data(), reinterpret_cast<fftw_complex *>(out_ptr.data()),
          FFTW_MEASURE);
    }
    std::copy(in, in + size_in(), in_ptr.begin());
    fftw_execute(fftw_plan_handle);
    return out_ptr.data();
  }

  // Inverse (c2r, unnormalized) of execute_host: the half spectrum `in`
  // (host memory, size size_out()) to the real volume `out` (host memory,
  // size size_in()). `in` is not modified.
  void execute_inverse_host(const std::complex<double> *in, double *out) {
    if (!fftw_inverse_plan) {
      in_ptr.resize(size_in());
      out_ptr.resize(out_size);
      fftw_inverse_plan = fftw_plan_dft_c2r_3d(
          static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
          reinterpret_cast<fftw_complex *>(out_ptr.data()), in_ptr.data(),
          FFTW_MEASURE);
    }
    // c2r overwrites its input, so transform a copy
    std::copy(in, in + out_size, out_ptr.begin());
    fftw_execute(fftw_inverse_plan);
    std::copy(in_ptr.begin(), in_ptr.end(), out);
  }

#ifdef SFGPU_CUFFT
  // Transform the real volume `d_in` (device memory, size n0*n1*n2; not
  // modified). Returns the half spectrum in device memory, valid until the
  // next call.
  const cufftDoubleComplex *execute_device(const double *d_in) {
    if (!have_cufft) {
      if (cufftPlan3d(&cufft_plan, static_cast<int>(n0), static_cast<int>(n1),
                      static_cast<int>(n2), CUFFT_D2Z) != CUFFT_SUCCESS) {
        throw std::runtime_error("measurer_fft_3d: cufftPlan3d failed");
      }
      if (cudaMalloc(&d_out, out_size * sizeof(cufftDoubleComplex)) !=
          cudaSuccess) {
        cufftDestroy(cufft_plan);
        throw std::runtime_error("measurer_fft_3d: cudaMalloc failed");
      }
      have_cufft = true;
    }
    // An out-of-place D2Z transform leaves its input unchanged
    cufftExecD2Z(cufft_plan, const_cast<double *>(d_in), d_out);
    cudaDeviceSynchronize();
    return d_out;
  }

  // Inverse (Z2D, unnormalized) of execute_device: the half spectrum d_in
  // (device memory, size size_out(); overwritten, as cuFFT may do for C2R)
  // to the real volume d_real (device memory, size size_in()).
  void execute_inverse_device(cufftDoubleComplex *d_in, double *d_real) {
    if (!have_cufft_inverse) {
      if (cufftPlan3d(&cufft_inverse_plan, static_cast<int>(n0),
                      static_cast<int>(n1), static_cast<int>(n2),
                      CUFFT_Z2D) != CUFFT_SUCCESS) {
        throw std::runtime_error("measurer_fft_3d: cufftPlan3d (Z2D) failed");
      }
      have_cufft_inverse = true;
    }
    cufftExecZ2D(cufft_inverse_plan, d_in, d_real);
    cudaDeviceSynchronize();
  }
#endif
};

////////////////////////////////////////////////////////////////////////
// Computes slices of the fields and their fourier transforms
//
// The fields are labelled by U = phi, A, V, and phi2 and hence has dimensions
// NObs = Nphi +  NA +  NV  + 1 respectively.  The field phi2 is the square of
// the  field phi_a.
//
// Our fourier transform conventions are  (1/N_x) \sum_x e^{ikx} W(x) = W(k)
// where W(x)  is the average of the fields over the y and z directions, i.e.
// W(x) = 1/N_y * 1/N_z \sum_{y,z} U(x,y,z)
//
// The array wallX has dimensions NObs x N and contains the slices of the fields
// in the x direction. The array wallX_k has dimensions NObs x N/2 + 1 and
// contains the fourier transform of the slices of the fields in the x
// direction.
//
// The fields are rotated in the direction of the vev and the rotated fields are
// stored in the array wallX_rotated. The array wallXPhase has dimensions
// NObsPhase x N and contains the slices of the fields in the x direction of the
// rotated fields.  Finally the array wallXPhase_k has dimensions NObsPhase x
// N/2 + 1 and contains the fourier transform of the slices of the fields in the
// x direction of the rotated fields.
//
// The rotated fields are define as follows: The zero mode of the field phi
// defines a unit four vector n_a. The rotated fields are defined as follows:
//
// sigma = n_a phi_a,
// pi_b = phi_a - sigma n_b,
// A_a = rho_{ab} n_b,
// V_a = rhotilde_{ab} n_b,
// phi2.
//
// Here rho_{ab} is the field A_a and V_a are the
// axial and vector densities respectively and rhotilde_{ab} is the dual of the
// field rho_{ab}, i.e. multiplied by the epsilon tensor epsilon_{abcd} which
// swaps the vector and axial vector pieces.  Thus the dimension of the rotated
// fields is NObsPhase = 3*Nphi + 2 where Nphi is the number of scalar fields.
class Measurer {
public:
  // Arrays of size Nobs contain X=(phi[1..Nphi], q[1...Nq], phi2)
  static const PetscInt NObs =
      ModelAData::Nphi + ModelAData::NA + ModelAData::NV + 1;

  // Scalar data this is k=0 mode by itself
  static const PetscInt NScalars = NObs;
  std::vector<PetscScalar> OAverage;

  // Total energy of the ideal (Hamiltonian) sector,
  //   E = sum_x [ (A^2 + V^2) / (2 chi) + (grad phi)^2 / 2 - H phi_0 ],
  // with forward differences for grad phi. In superfluid mode H is the
  // effective field H * sigma/f used by the ideal step. Valid on rank 0 only.
  PetscScalar energy = 0.;

  // Spherically-averaged Fourier readout of the topological charge density
  // q(x) of the O(4) field (see computeTopChargeFourier in measurer.cxx).
  int NtopchargeBins = 0;
  double topcharge_dk = 0.;             // radial bin width = 2*pi/L
  std::vector<double> topcharge_Sk;     // S(k) = <|qtilde|^2> per radial shell
  std::vector<double> topcharge_kbins;  // |k| bin centers, = m*dk
  std::vector<double> topcharge_Nshell; // number of modes per shell
  std::complex<double> topcharge_zero{0., 0.}; // qtilde(k=0), DC component

  // Radially binned 3D power spectra of phi_a(x) and the locally-rotated
  // A_a(x) = contract_rho(n(x), rho(x))[a] with n(x) = phi(x)/|phi(x)|,
  // matching computeSliceAveragePhase.  Uses hybrid exact/log binning: one
  // bin per lattice shell |n|^2 = 1..10, then 20 bins per decade up to k_Nyq.
  // All quantities are traces over the internal index a.
  // Convention: f(k) = (1/N^3) Σ_x e^{-ikx} f(x), matching wallX_k.
  int NFieldBins = 0;
  std::vector<double> field_bin_edges; // NFieldBins+1 edges in |k|
  std::vector<double> field_kbins;    // mean |k| per bin (from mode sum)
  std::vector<double> field_Nshell;   // mode count per bin (full k-space weight)
  std::vector<double> field_Pphi;     // mean |phi(k)|^2 per bin, summed over a
  std::vector<double> field_PAA;      // mean |A(k)|^2 per bin, summed over a
  std::vector<double> field_ReC;      // mean Re[phi_a(k) A_a(k)*] per bin
  std::vector<double> field_PVV;      // mean |V_a(k)|^2 per bin (only with
                                      // vector_spectra)

  // Precomputed per-mode bin index, weight, and |k| for the half-spectrum
  // (size N*N*(N/2+1), built once in the constructor).
  std::vector<int> field_bin_map;
  std::vector<double> field_weight;
  std::vector<double> field_kmag;

  // First dimension is NObs, last is spatial index x=0...N
  nvector<PetscScalar, 2> wallX;
  nvector<PetscScalar, 2> wallY;
  nvector<PetscScalar, 2> wallZ;

  // First dimension is NObs, last dimension is fourier index k=0..N/2+1
  nvector<std::complex<double>, 2> wallX_k;
  nvector<std::complex<double>, 2> wallY_k;
  nvector<std::complex<double>, 2> wallZ_k;

  // First dimension is NObsRotated, last dimension is fourier index.
  // Array of size NobsRotate contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  static const PetscInt NObsRotated = 3 * ModelAData::Nphi + 2;
  nvector<std::complex<double>, 2> wallX_k_rotated;
  nvector<std::complex<double>, 2> wallY_k_rotated;
  nvector<std::complex<double>, 2> wallZ_k_rotated;

  // Array of size NobsPhase contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  // The first dimension is NObsPhase, the second domenions is the spatial index
  static const PetscInt NObsPhase = 3 * ModelAData::Nphi + 2;
  nvector<PetscScalar, 2> wallXPhase;
  nvector<PetscScalar, 2> wallYPhase;
  nvector<PetscScalar, 2> wallZPhase;

  // First dimension is NObsPhase, last dimension is the fourier index 0..N/2+1
  nvector<std::complex<double>, 2> wallXPhase_k;
  nvector<std::complex<double>, 2> wallYPhase_k;
  nvector<std::complex<double>, 2> wallZPhase_k;

public:
  Measurer(ModelA *ptr) : model(ptr) {
    N = model->data.NX;
    if (model->data.NX != model->data.NY || model->data.NX != model->data.NZ) {
      PetscPrintf(
          PETSC_COMM_WORLD,
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
      throw(
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }

    // Only need the fft for derived observables
    int rank = -1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
       fftw = std::make_unique<measurer_fft>(N);
    }


    wallX.resize(NObs, N);
    wallY.resize(NObs, N);
    wallZ.resize(NObs, N);

    wallX_k.resize(NObs, N / 2 + 1);
    wallY_k.resize(NObs, N / 2 + 1);
    wallZ_k.resize(NObs, N / 2 + 1);

    wallX_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallY_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallZ_k_rotated.resize(NObsRotated, N / 2 + 1);

    wallXPhase.resize(NObsPhase, N);
    wallYPhase.resize(NObsPhase, N);
    wallZPhase.resize(NObsPhase, N);

    wallXPhase_k.resize(NObsPhase, N / 2 + 1);
    wallYPhase_k.resize(NObsPhase, N / 2 + 1);
    wallZPhase_k.resize(NObsPhase, N / 2 + 1);

    // Set up the radial binning for the spherically averaged topological
    // charge spectrum.  The bin width is the reciprocal-lattice spacing
    // dk = 2*pi/L (LX == LY == LZ here since NX == NY == NZ).  The largest
    // possible |k| integer magnitude is sqrt(3)*(N/2), so we need
    // floor(sqrt(3)*N/2 + 0.5)+1 bins with center |k|_m = m*dk.
    topcharge_dk = 2.0 * M_PI / model->data.LX;
    const double rmax = sqrt(3.0) * (0.5 * static_cast<double>(N));
    NtopchargeBins = static_cast<int>(floor(rmax + 0.5)) + 1;
    topcharge_Sk.assign(NtopchargeBins, 0.);
    topcharge_kbins.assign(NtopchargeBins, 0.);
    topcharge_Nshell.assign(NtopchargeBins, 0.);
    for (int m = 0; m < NtopchargeBins; m++) {
      topcharge_kbins[m] = m * topcharge_dk;
    }
    // Field-spectra hybrid exact/log bin edges (pure geometry, no FFT needed)
    {
      const double kf = 2.0 * M_PI / model->data.LX;
      const double a_lat = model->data.LX / N;
      const double kNyq = M_PI / a_lat;
      const double ratio = pow(10.0, 1.0 / 20.0); // 20 bins per decade
      // Exact part: one bin per lattice shell |n|^2 = 1..10 (7 is not a sum
      // of three squares). Since |n|^2 is an integer, the edge
      // sqrt(|n|^2 + 0.5)*kf lies strictly between neighbouring shells. The
      // first edge 0.5*kf excludes k = 0.
      const int exact_n2[] = {1, 2, 3, 4, 5, 6, 8, 9, 10};
      field_bin_edges.push_back(0.5 * kf);
      for (int n2 : exact_n2)
        field_bin_edges.push_back(sqrt(n2 + 0.5) * kf);
      // Log part: multiply by ratio until kNyq
      double edge = field_bin_edges.back() * ratio;
      while (edge < kNyq) {
        field_bin_edges.push_back(edge);
        edge *= ratio;
      }
      field_bin_edges.push_back(kNyq);
      // Merge final bin if narrower than half a log step
      if (field_bin_edges.size() >= 3) {
        const double last = field_bin_edges[field_bin_edges.size() - 1];
        const double prev = field_bin_edges[field_bin_edges.size() - 2];
        if (last / prev < sqrt(ratio))
          field_bin_edges.erase(field_bin_edges.end() - 2);
      }
      NFieldBins = static_cast<int>(field_bin_edges.size()) - 1;
      field_kbins.assign(NFieldBins, 0.0);
      field_Nshell.assign(NFieldBins, 0.0);
      field_Pphi.assign(NFieldBins, 0.0);
      field_PAA.assign(NFieldBins, 0.0);
      field_ReC.assign(NFieldBins, 0.0);
      field_PVV.assign(NFieldBins, 0.0);

      // Precompute the bin index and weight for every mode in the half-spectrum.
      // This avoids a binary search per mode inside the binning loop.
      const int nxh = N / 2 + 1;
      const size_t nhalf = static_cast<size_t>(N) * N * nxh;
      field_bin_map.resize(nhalf);
      field_weight.resize(nhalf);
      field_kmag.resize(nhalf);
      const int half = N / 2;
      for (int kz = 0; kz < N; kz++) {
        const int mz = (kz <= half) ? kz : kz - N;
        for (int ky = 0; ky < N; ky++) {
          const int my = (ky <= half) ? ky : ky - N;
          for (int kx = 0; kx < nxh; kx++) {
            const size_t idx =
                (static_cast<size_t>(kz) * N + ky) * nxh + kx;
            const double km =
                kf * sqrt(static_cast<double>(kx * kx + my * my + mz * mz));
            field_kmag[idx] = km;
            // Binary search for bin
            int b = -1;
            if (km >= field_bin_edges[0] && km < field_bin_edges[NFieldBins]) {
              int lo = 0, hi = NFieldBins;
              while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (km >= field_bin_edges[mid]) lo = mid; else hi = mid;
              }
              b = lo;
            }
            field_bin_map[idx] = b;
            field_weight[idx] = (kx > 0 && kx < half) ? 2.0 : 1.0;
          }
        }
      }
      // Precompute per-bin Nshell and mean |k| (these are time-independent)
      for (size_t idx = 0; idx < nhalf; idx++) {
        const int b = field_bin_map[idx];
        if (b >= 0) {
          field_Nshell[b] += field_weight[idx];
          field_kbins[b] += field_weight[idx] * field_kmag[idx];
        }
      }
      for (int b = 0; b < NFieldBins; b++) {
        if (field_Nshell[b] > 0.0)
          field_kbins[b] /= field_Nshell[b];
      }
    }
    // The 3D FFT machinery is set up on first use (setupSpectra / setupFieldBins)
  }

  virtual ~Measurer() {
    finalizeAnalyticalCoarsening();
#ifdef SFGPU_CUFFT
    if (d_field_ft_buf) cudaFree(d_field_ft_buf);
    if (d_field_bin_map) cudaFree(d_field_bin_map);
    if (d_field_weight) cudaFree(d_field_weight);
#endif
    VecScatterDestroy(&toZero);
    VecDestroy(&qseq);
    VecDestroy(&qnat);
    VecDestroy(&qglob);
    DMDestroy(&da1);
  }

  // Takes the vector, solution,  and computes the walls and their fourier
  // transforms and the rotated versions.  On output the arrays wallX, wallX_k,
  // wallX_rotated, as well as Y and Z are filled with the data. This can be
  // accessed to write the data to disk.
  void measure(Vec *solution) {
    int rank = -1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    computeSliceAverages(solution);

    // Take the FFT and other steps based on the data collected
    if (rank == 0) {
      computeDerivedObs();
    }
  }

  // Computes the topological charge density q(x), its 3D FFT and the
  // spherically averaged power spectrum S(k) together with the zero mode.
  // computeTopChargeFourier is collective (it reduces q(x) to rank 0) so this
  // must be called on all ranks; the FFT/binning happens internally on rank 0.
  void measure_topcharge(Vec *solution) {
    computeTopChargeFourier(solution);
  }

  // Computes radially binned 3D power spectra of phi_a and the locally
  // rotated A_a (and V_a with vector_spectra). Collective, like
  // measure_topcharge.
  void measure_field_spectra(Vec *solution) { computeFieldSpectra(solution); }

  // Exact coarsening for simple dissipation (preserve_norm = false). The
  // coarsening diffusion is then linear: every Fourier mode of a field
  // component decays as exp(-c lambda(k) tau), with the lattice Laplacian
  // lambda(k) = sum_d (2 - 2 cos k_d)/h_d^2 of the diffusion stencil and
  // c = Gamma for phi, D for the charges; the source Gamma H sigma/f moves
  // the zero mode of phi_0 linearly in tau. setupAnalyticalCoarsening stores
  // the spectra of all Ndof components of U on rank 0 (in device memory if U
  // is a Kokkos vector); analyticalCoarsenedField writes the exactly diffused
  // field at diffusion time tau into out. All three are collective.
  void setupAnalyticalCoarsening(Vec *U);
  void analyticalCoarsenedField(double tau, Vec *out);
  void finalizeAnalyticalCoarsening();

  ModelA *getModel() { return model; }
  PetscInt getN() { return N; }
  int getNTopchargeBins() { return NtopchargeBins; }
  int getNFieldBins() { return NFieldBins; }
  double getTopchargeDk() { return topcharge_dk; }

private:
  // Slice averages, phase slice averages and energy: on the device if the
  // solution is a Kokkos vector, otherwise with the host loops below
  void computeSliceAverages(Vec *solution);
  void computeSliceAverage(Vec *solution);
  void computeEnergy(G_node ***fld);
  void computeSliceAveragePhase(Vec *solution);
  void computeDerivedObs();
  void computeTopChargeFourier(Vec *solution);
  void computeFieldSpectra(Vec *solution);
  void setupFieldBins();

  // Temporary host buffer for one half-spectrum (rank 0, used by
  // computeFieldSpectra to hold pi_a(k) while computing A_a(k))
  std::vector<std::complex<double>> field_ft_buf;
#ifdef SFGPU_CUFFT
  cufftDoubleComplex *d_field_ft_buf = nullptr;
  // field_bin_map and field_weight on the device, uploaded once
  int *d_field_bin_map = nullptr;
  double *d_field_weight = nullptr;
#endif
  // Gather qglob to rank 0 in natural ordering; returns the vector holding
  // all of q on rank 0 (qglob itself with one rank). Collective.
  Vec gatherQToZero();

  // 3D spectra of a scalar field q(x). q lives in qglob, a global vector of
  // the 1-dof DMDA da1 with the same layout as the model's domain.
  // radialSpectrum brings q to rank 0 in natural order (only a copy for one
  // rank), takes its FFT there (cuFFT if q is on the device) and bins
  // |q(k)|^2 into the |k| shells; Sk, Nshell and zero are set on rank 0.
  PetscErrorCode setupSpectra();
  PetscErrorCode radialSpectrum(double *Sk, double *Nshell,
                                std::complex<double> *zero);
  DM da1 = nullptr;
  Vec qglob = nullptr;
  Vec qnat = nullptr;           // natural ordering (more than one rank)
  Vec qseq = nullptr;           // all of q on rank 0 (more than one rank)
  VecScatter toZero = nullptr;

  // Analytical coarsening (rank 0): the half spectra of all Ndof components
  // at tau = 0, one after the other, plus one work spectrum; on the host or,
  // if coarse_on_device, on the device
  std::vector<std::complex<double>> coarse_ft;
  std::vector<std::complex<double>> coarse_work;
  bool coarse_on_device = false;
#ifdef SFGPU_CUFFT
  cufftDoubleComplex *d_coarse_ft = nullptr;
  cufftDoubleComplex *d_coarse_work = nullptr;
#endif

  ModelA *model;
  PetscInt N;

  // FFT engine using the fftw3 library
  std::unique_ptr<measurer_fft> fftw;
  // 3D FFT engine for the spectra (rank 0 only, made by setupSpectra)
  std::unique_ptr<measurer_fft_3d> fftw3d;
};

#endif
