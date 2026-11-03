// First: PETSc requires its Kokkos headers before any other PETSc header
#include "KokkosTypes.h"
#include "measurer.h"
#include "O4AlgebraHelper.h"
#include "Profiling.h"
#include "nvector.h"
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

////////////////////////////////////////////////////////////////////////

void Measurer::computeSliceAverage(Vec *solution) {
  // Get the local information and store in info

  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  // take the global vector U and distribute to the local vector localU
  ghost_exchange(da, *solution, localU);

  // From the vector define the pointer for the field phi
  G_node ***fld;
  DMDAVecGetArrayRead(da, localU, &fld);

  // Set up the slize averages initialized to zero in c++11
  std::fill(wallX.v.begin(), wallX.v.end(), 0.);
  std::fill(wallY.v.begin(), wallY.v.end(), 0.);
  std::fill(wallZ.v.begin(), wallZ.v.end(), 0.);

  // Local arrays with same characteristic dimensions
  nvector<PetscScalar, 2> wallLocalX(NObs, N);
  nvector<PetscScalar, 2> wallLocalY(NObs, N);
  nvector<PetscScalar, 2> wallLocalZ(NObs, N);

  std::fill(wallLocalX.v.begin(), wallLocalX.v.end(), 0.);
  std::fill(wallLocalY.v.begin(), wallLocalY.v.end(), 0.);
  std::fill(wallLocalZ.v.begin(), wallLocalZ.v.end(), 0.);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);
  PetscReal phi2;

  int actualInd = 0;

  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {
        phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallLocalX(l, i) += fld[k][j][i].f[l];
          wallLocalY(l, j) += fld[k][j][i].f[l];
          wallLocalZ(l, k) += fld[k][j][i].f[l];
          phi2 += pow(fld[k][j][i].f[l], 2);
        }
        for (int l = ModelAData::Nphi; l < ModelAData::Nphi + ModelAData::NA;
             l++) {
          actualInd = l - ModelAData::Nphi;
          wallLocalX(l, i) += fld[k][j][i].A[actualInd];
          wallLocalY(l, j) += fld[k][j][i].A[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].A[actualInd];
        }
        for (int l = ModelAData::Nphi + ModelAData::NA;
             l < ModelAData::Nphi + ModelAData::NA + ModelAData::NV; l++) {
          actualInd = l - ModelAData::Nphi - ModelAData::NA;
          wallLocalX(l, i) += fld[k][j][i].V[actualInd];
          wallLocalY(l, j) += fld[k][j][i].V[actualInd];
          wallLocalZ(l, k) += fld[k][j][i].V[actualInd];
        }
        int l = ModelAData::Nphi + ModelAData::NA + ModelAData::NV;
        wallLocalX(l, i) += phi2;
        wallLocalY(l, j) += phi2;
        wallLocalZ(l, k) += phi2;
      }
    }
  }

  for (int l = 0; l < NObs; ++l) {
    for (int i = 0; i < N; ++i) {
      wallLocalX(l, i) /= PetscReal(N * N);
      wallLocalY(l, i) /= PetscReal(N * N);
      wallLocalZ(l, i) /= PetscReal(N * N);
    }
  }

  // The energy needs the ghost cells filled above
  if (model->data.ahandler.compute_energy) {
    computeEnergy(fld);
  }

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  for (int l = 0; l < NObs; l++) {
    MPI_Reduce(&wallLocalX(l, 0), &wallX(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalY(l, 0), &wallY(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
    MPI_Reduce(&wallLocalZ(l, 0), &wallZ(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
  }
}

// fld is a ghosted local array. The sum is reduced to rank 0 in energy.
void Measurer::computeEnergy(G_node ***fld) {
  DM &da = model->domain;
  const auto &data = model->data;
  const auto &coeff = data.acoefficients;
  // Same effective field H * sigma/f as IdealPV2::step_no_reject
  const PetscReal H[4] = {coeff.H * coeff.sigmabyf(data.atime.t()), 0., 0.,
                          0.};

  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  PetscScalar phimid = 0, grad2 = 0, nab2 = 0, hEn = 0;
  for (PetscInt k = izs; k < izs + nz; k++) {
    for (PetscInt j = iys; j < iys + ny; j++) {
      for (PetscInt i = ixs; i < ixs + nx; i++) {
        for (PetscInt s = 0; s < ModelAData::Nphi; ++s) {
          phimid = fld[k][j][i].f[s];

          grad2 += pow(fld[k + 1][j][i].f[s] - phimid, 2);
          grad2 += pow(fld[k][j + 1][i].f[s] - phimid, 2);
          grad2 += pow(fld[k][j][i + 1].f[s] - phimid, 2);

          hEn += phimid * H[s];
        }
        for (PetscInt s = 0; s < ModelAData::NV; s++) {
          nab2 += pow(fld[k][j][i].V[s], 2);
        }
        for (PetscInt s = 0; s < ModelAData::NA; s++) {
          nab2 += pow(fld[k][j][i].A[s], 2);
        }
      }
    }
  }

  PetscScalar localEnergy = 0.5 / coeff.chi * nab2 + 0.5 * grad2 - hEn;
  MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
             PETSC_COMM_WORLD);
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 4> contract_rho(const std::array<T, 4> &n,
                                     const std::array<T, 6> &rho) {
  return {-(n[1] * rho[0]) - n[2] * rho[1] - n[3] * rho[2],
          n[0] * rho[0] + n[3] * rho[4] - n[2] * rho[5],
          n[0] * rho[1] - n[3] * rho[3] + n[1] * rho[5],
          n[0] * rho[2] + n[2] * rho[3] - n[1] * rho[4]};
}

// Helper function for slice averages works for double and complex types
template <typename T>
inline std::array<T, 6> dualize_rho(const std::array<T, 6> &rho) {
  return std::array<T, 6>{rho[3], rho[4], rho[5], rho[0], rho[1], rho[2]};
}

void Measurer::computeSliceAveragePhase(Vec *solution) {
  // Get the local information and store in info

  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  // take the global vector U and distribute to the local vector localU
  ghost_exchange(da, *solution, localU);

  // From the vector define the pointer for the field phi
  G_node ***fld;
  DMDAVecGetArrayRead(da, localU, &fld);

  // Set up the slize averages initialized to zero in c++11
  std::fill(wallXPhase.v.begin(), wallXPhase.v.end(), 0.);
  std::fill(wallYPhase.v.begin(), wallYPhase.v.end(), 0.);
  std::fill(wallZPhase.v.begin(), wallZPhase.v.end(), 0.);

  // Local arrays with same dimensions initialized to zero
  nvector<double, 2> wallXPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallYPhaseLocal(NObsPhase, N);
  nvector<double, 2> wallZPhaseLocal(NObsPhase, N);

  // Get the ranges
  PetscInt ixs, iys, izs, nx, ny, nz;
  DMDAGetCorners(da, &ixs, &iys, &izs, &nx, &ny, &nz);

  double sigma;
  std::array<double, 4> n{};
  std::array<double, 4> phi{};
  std::array<double, 6> rho{};
  std::array<double, 4> V{};
  std::array<double, 4> A{};
  // Store the local averages
  for (int k = izs; k < izs + nz; k++) {
    for (int j = iys; j < iys + ny; j++) {
      for (int i = ixs; i < ixs + nx; i++) {

        double phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          phi[l] = fld[k][j][i].f[l];
          phi2 += pow(phi[l], 2);
        }
        sigma = sqrt(phi2);

        if (phi2 > 100. * std::numeric_limits<double>::min()) {
          for (int l = 0; l < ModelAData::Nphi; l++) {
            n[l] = phi[l] / sigma;
          }
        } else {
          n = {1, 0, 0, 0};
        }

        for (int l = 0; l < ModelAData::NA; l++) {
          rho[l] = fld[k][j][i].A[l];
          rho[l + ModelAData::NA] = fld[k][j][i].V[l];
        }
        A = contract_rho(n, rho);
        V = contract_rho(n, dualize_rho(rho));

        wallXPhaseLocal(0, i) += sigma;
        wallYPhaseLocal(0, j) += sigma;
        wallZPhaseLocal(0, k) += sigma;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          wallXPhaseLocal(1 + l, i) += n[l];
          wallYPhaseLocal(1 + l, j) += n[l];
          wallZPhaseLocal(1 + l, k) += n[l];

          wallXPhaseLocal(5 + l, i) += A[l];
          wallYPhaseLocal(5 + l, j) += A[l];
          wallZPhaseLocal(5 + l, k) += A[l];

          wallXPhaseLocal(9 + l, i) += V[l];
          wallYPhaseLocal(9 + l, j) += V[l];
          wallZPhaseLocal(9 + l, k) += V[l];
        }
        wallXPhaseLocal(Measurer::NObsPhase - 1, i) += sigma * sigma;
        wallYPhaseLocal(Measurer::NObsPhase - 1, j) += sigma * sigma;
        wallZPhaseLocal(Measurer::NObsPhase - 1, k) += sigma * sigma;
      }
    }
  }
  for (int l = 0; l < NObsPhase; ++l) {
    for (int i = 0; i < N; ++i) {
      wallXPhaseLocal(l, i) /= PetscReal(N * N);
      wallYPhaseLocal(l, i) /= PetscReal(N * N);
      wallZPhaseLocal(l, i) /= PetscReal(N * N);
    }
  }

  // Retstore the array
  DMDAVecRestoreArrayRead(da, localU, &fld);
  DMRestoreLocalVector(da, &localU);

  // Bring all the data x data into one
  for (int l = 0; l < NObsPhase; l++) {
    MPI_Reduce(&wallXPhaseLocal(l, 0), &wallXPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallYPhaseLocal(l, 0), &wallYPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&wallZPhaseLocal(l, 0), &wallZPhase(l, 0), N, MPIU_SCALAR,
               MPI_SUM, 0, PETSC_COMM_WORLD);
  }
}

#ifdef SFGPU_KOKKOS
// Per-site observables summed into the slice averages: entries 0..NObs-1
// are X = (phi[4], A[3], V[3], phi2) as in computeSliceAverage, and entries
// NObs..NObs+NObsPhase-1 are (sigma, n[4], A'[4], V'[4], sigma^2) as in
// computeSliceAveragePhase.
struct SliceObs {
  static const int NTot = Measurer::NObs + Measurer::NObsPhase;
  double v[NTot];
  KOKKOS_INLINE_FUNCTION SliceObs() {
    for (int o = 0; o < NTot; o++) v[o] = 0.;
  }
  KOKKOS_INLINE_FUNCTION SliceObs &operator+=(const SliceObs &b) {
    for (int o = 0; o < NTot; o++) v[o] += b.v[o];
    return *this;
  }
};

namespace Kokkos {
template <> struct reduction_identity<SliceObs> {
  KOKKOS_FORCEINLINE_FUNCTION static SliceObs sum() { return SliceObs(); }
};
} // namespace Kokkos

// Adds the observables of site (k, j, i) to acc. Same formulas as the host
// loops (contract_rho and dualize_rho written out).
KOKKOS_INLINE_FUNCTION void add_site_obs(const ConstNodeView &u, PetscInt k,
                                         PetscInt j, PetscInt i,
                                         SliceObs &acc) {
  const int NObs = Measurer::NObs;
  constexpr PetscInt Nphi = ModelAData::Nphi, NA = ModelAData::NA;
  double phi[Nphi], rho[NA + ModelAData::NV];
  double phi2 = 0.;
  for (int l = 0; l < Nphi; l++) {
    phi[l] = u(k, j, i, l);
    acc.v[l] += phi[l];
    phi2 += phi[l] * phi[l];
  }
  for (int l = 0; l < NA; l++) {
    rho[l] = u(k, j, i, ModelAData::OffsetA + l);      // A
    rho[l + NA] = u(k, j, i, ModelAData::OffsetV + l); // V
    acc.v[ModelAData::OffsetA + l] += rho[l];
    acc.v[ModelAData::OffsetV + l] += rho[l + NA];
  }
  acc.v[ModelAData::Ndof] += phi2;

  // Rotated to the local direction n = phi / sigma
  const double sigma = Kokkos::sqrt(phi2);
  double n[4] = {1., 0., 0., 0.};
  if (phi2 > 100. * Kokkos::Experimental::norm_min_v<double>) {
    for (int l = 0; l < 4; l++) n[l] = phi[l] / sigma;
  }
  const double *r = rho;
  const double rd[6] = {rho[3], rho[4], rho[5], rho[0], rho[1], rho[2]};
  const double A[4] = {-(n[1] * r[0]) - n[2] * r[1] - n[3] * r[2],
                       n[0] * r[0] + n[3] * r[4] - n[2] * r[5],
                       n[0] * r[1] - n[3] * r[3] + n[1] * r[5],
                       n[0] * r[2] + n[2] * r[3] - n[1] * r[4]};
  const double V[4] = {-(n[1] * rd[0]) - n[2] * rd[1] - n[3] * rd[2],
                       n[0] * rd[0] + n[3] * rd[4] - n[2] * rd[5],
                       n[0] * rd[1] - n[3] * rd[3] + n[1] * rd[5],
                       n[0] * rd[2] + n[2] * rd[3] - n[1] * rd[4]};
  acc.v[NObs + 0] += sigma;
  for (int l = 0; l < 4; l++) {
    acc.v[NObs + 1 + l] += n[l];
    acc.v[NObs + 5 + l] += A[l];
    acc.v[NObs + 9 + l] += V[l];
  }
  acc.v[NObs + Measurer::NObsPhase - 1] += sigma * sigma;
}

// Sums the site observables over each plane of one direction (dir = 0, 1, 2
// for x, y, z walls) with one team per plane. Deterministic: no atomics.
// Returns W(p, o) on the host for the local planes p = 0..nplanes-1.
static Kokkos::View<double **>::host_mirror_type
plane_sums(const ConstNodeView &u, int dir, PetscInt xs, PetscInt ys,
           PetscInt zs, PetscInt xm, PetscInt ym, PetscInt zm) {
  const PetscInt nplanes = dir == 0 ? xm : (dir == 1 ? ym : zm);
  // The two other directions, fastest (contiguous in memory) last
  const PetscInt n1 = dir == 2 ? ym : zm;
  const PetscInt n2 = dir == 0 ? ym : xm;
  Kokkos::View<double **> W("plane_sums", nplanes, SliceObs::NTot);

  using Team = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "SlicePlanes", Kokkos::TeamPolicy<>(nplanes, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const PetscInt p = team.league_rank();
        SliceObs sum;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n1 * n2),
            [&](PetscInt idx, SliceObs &acc) {
              const PetscInt a = idx / n2, b = idx % n2;
              PetscInt k, j, i;
              if (dir == 0) {
                k = zs + a, j = ys + b, i = xs + p;
              } else if (dir == 1) {
                k = zs + a, j = ys + p, i = xs + b;
              } else {
                k = zs + p, j = ys + a, i = xs + b;
              }
              add_site_obs(u, k, j, i, acc);
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          for (int o = 0; o < SliceObs::NTot; o++) W(p, o) = sum.v[o];
        });
      });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W);
}

// The three sums of the energy, same terms as Measurer::computeEnergy
static void energy_terms(const ConstNodeView &u, PetscInt xs, PetscInt ys,
                         PetscInt zs, PetscInt xm, PetscInt ym, PetscInt zm,
                         PetscReal H0, PetscScalar &grad2, PetscScalar &nab2,
                         PetscScalar &hEn) {
  Kokkos::parallel_reduce(
      "Energy", SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i, PetscScalar & g,
                    PetscScalar & q, PetscScalar & h) {
        for (int s = 0; s < ModelAData::Nphi; s++) {
          const PetscScalar mid = u(k, j, i, s);
          const PetscScalar dz = u(k + 1, j, i, s) - mid;
          const PetscScalar dy = u(k, j + 1, i, s) - mid;
          const PetscScalar dx = u(k, j, i + 1, s) - mid;
          g += dz * dz + dy * dy + dx * dx;
        }
        h += u(k, j, i, 0) * H0;
        for (int c = ModelAData::OffsetA; c < ModelAData::Ndof; c++) {
          q += u(k, j, i, c) * u(k, j, i, c);
        }
      },
      grad2, nab2, hEn);
}
#endif

// Sums the local walls of all ranks into global on rank 0
static void reduce_walls(nvector<PetscScalar, 2> &local,
                         nvector<PetscScalar, 2> &global, PetscInt nobs,
                         PetscInt N) {
  for (int l = 0; l < nobs; l++) {
    MPI_Reduce(&local(l, 0), &global(l, 0), N, MPIU_SCALAR, MPI_SUM, 0,
               PETSC_COMM_WORLD);
  }
}

void Measurer::computeSliceAverages(Vec *solution) {
#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(*solution)) {
    DM &da = model->domain;
    Vec localU;
    DMGetLocalVector(da, &localU);
    // Ghost cells are needed only for the gradient in the energy
    ghost_exchange(da, *solution, localU);

    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

    ConstNodeView u;
    DMDAVecGetKokkosOffsetViewDOF(da, localU, &u);

    // Slice averages: plane sums on the device, then the same
    // normalization and reduction as the host loops
    nvector<PetscScalar, 2> wallLocal[3], wallPhaseLocal[3];
    const PetscInt start[3] = {xs, ys, zs};
    for (int dir = 0; dir < 3; dir++) {
      wallLocal[dir].resize(NObs, N);
      wallPhaseLocal[dir].resize(NObsPhase, N);
      std::fill(wallLocal[dir].v.begin(), wallLocal[dir].v.end(), 0.);
      std::fill(wallPhaseLocal[dir].v.begin(), wallPhaseLocal[dir].v.end(),
                0.);
      auto W = plane_sums(u, dir, xs, ys, zs, xm, ym, zm);
      for (PetscInt p = 0; p < (PetscInt)W.extent(0); p++) {
        for (int o = 0; o < NObs; o++) {
          wallLocal[dir](o, start[dir] + p) = W(p, o) / PetscReal(N * N);
        }
        for (int o = 0; o < NObsPhase; o++) {
          wallPhaseLocal[dir](o, start[dir] + p) =
              W(p, NObs + o) / PetscReal(N * N);
        }
      }
    }

    // Energy (optional)
    const auto &data = model->data;
    const auto &coeff = data.acoefficients;
    if (data.ahandler.compute_energy) {
      const PetscReal H0 = coeff.H * coeff.sigmabyf(data.atime.t());
      PetscScalar grad2 = 0, nab2 = 0, hEn = 0;
      energy_terms(u, xs, ys, zs, xm, ym, zm, H0, grad2, nab2, hEn);
      // Reduce after releasing the view below
      DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u);
      DMRestoreLocalVector(da, &localU);
      PetscScalar localEnergy = 0.5 / coeff.chi * nab2 + 0.5 * grad2 - hEn;
      MPI_Reduce(&localEnergy, &energy, 1, MPIU_SCALAR, MPI_SUM, 0,
                 PETSC_COMM_WORLD);
    } else {
      DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u);
      DMRestoreLocalVector(da, &localU);
    }

    std::fill(wallX.v.begin(), wallX.v.end(), 0.);
    std::fill(wallY.v.begin(), wallY.v.end(), 0.);
    std::fill(wallZ.v.begin(), wallZ.v.end(), 0.);
    std::fill(wallXPhase.v.begin(), wallXPhase.v.end(), 0.);
    std::fill(wallYPhase.v.begin(), wallYPhase.v.end(), 0.);
    std::fill(wallZPhase.v.begin(), wallZPhase.v.end(), 0.);
    reduce_walls(wallLocal[0], wallX, NObs, N);
    reduce_walls(wallLocal[1], wallY, NObs, N);
    reduce_walls(wallLocal[2], wallZ, NObs, N);
    reduce_walls(wallPhaseLocal[0], wallXPhase, NObsPhase, N);
    reduce_walls(wallPhaseLocal[1], wallYPhase, NObsPhase, N);
    reduce_walls(wallPhaseLocal[2], wallZPhase, NObsPhase, N);
    return;
  }
#endif
  computeSliceAverage(solution);
  computeSliceAveragePhase(solution);
}

////////////////////////////////////////////////////////////////////////
// Topological charge density computation
//
// q(x) = 1/(12 pi^2) eps^{abcd} phi_a (dx phi_b)(dy phi_c)(dz phi_d)
//       = 1/(12 pi^2) det[ phi , dx phi , dy phi , dz phi ]
//
// with centered finite differences d_i phi = (phi(x+e_i) - phi(x-e_i))/(2 h_i)

// Determinant of a 4x4 matrix whose rows are (r0, r1, r2, r3).
// This equals the fully-antisymmetric contraction
//   eps^{abcd} r0_a r1_b r2_c r3_d   with eps^{0123} = +1.
#ifdef SFGPU_KOKKOS
O4_FUNCTION
#else
static inline
#endif
double det4x4(const double r0[4], const double r1[4],
              const double r2[4], const double r3[4]) {
  // 3x3 minor of rows (r1, r2, r3) over columns (c0, c1, c2).
  // Written out explicitly (no lambda) so it compiles on CUDA devices.
  const double m123 = r1[1] * (r2[2] * r3[3] - r2[3] * r3[2]) -
                      r1[2] * (r2[1] * r3[3] - r2[3] * r3[1]) +
                      r1[3] * (r2[1] * r3[2] - r2[2] * r3[1]);
  const double m023 = r1[0] * (r2[2] * r3[3] - r2[3] * r3[2]) -
                      r1[2] * (r2[0] * r3[3] - r2[3] * r3[0]) +
                      r1[3] * (r2[0] * r3[2] - r2[2] * r3[0]);
  const double m013 = r1[0] * (r2[1] * r3[3] - r2[3] * r3[1]) -
                      r1[1] * (r2[0] * r3[3] - r2[3] * r3[0]) +
                      r1[3] * (r2[0] * r3[1] - r2[1] * r3[0]);
  const double m012 = r1[0] * (r2[1] * r3[2] - r2[2] * r3[1]) -
                      r1[1] * (r2[0] * r3[2] - r2[2] * r3[0]) +
                      r1[2] * (r2[0] * r3[1] - r2[1] * r3[0]);
  return r0[0] * m123 - r0[1] * m023 + r0[2] * m013 - r0[3] * m012;
}

// Radial shell m of the mode (kz, ky, kx) of the r2c half spectrum: modes
// with |k| = m*dk (rounded) share a shell, the last shell collects the corners
#ifdef SFGPU_KOKKOS
O4_FUNCTION
#else
static inline
#endif
int shell_index(int kz, int ky, int kx, int Nlat, int nbins) {
  const int half = Nlat / 2;
  const int mz = (kz <= half) ? kz : kz - Nlat;
  const int my = (ky <= half) ? ky : ky - Nlat;
  const double r = sqrt(static_cast<double>(kx * kx + my * my + mz * mz));
  const int m = static_cast<int>(floor(r + 0.5));
  return (m < nbins) ? m : nbins - 1;
}

#ifdef SFGPU_KOKKOS
// One value per owned site of the 1-dof DMDA used for the spectra
using SiteValueView =
    Kokkos::Experimental::OffsetView<PetscScalar ***, Kokkos::LayoutRight,
                                     KokkosMemSpace>;

// GPU kernel: compute q(x) at each owned site into q (a global vector of
// da1). localU is the ghosted solution.
static PetscErrorCode topcharge_kernel_kokkos(DM da, Vec localU, DM da1,
                                              Vec q, double prefac,
                                              double inv2hx, double inv2hy,
                                              double inv2hz) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  ConstNodeView u;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, localU, &u));
  SiteValueView qv;
  PetscCall(DMDAVecGetKokkosOffsetViewWrite(da1, q, &qv));

  Kokkos::parallel_for(
      "TopChargeDensity",
      SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        constexpr int Nphi = ModelAData::Nphi;
        double phi[Nphi], Dx[Nphi], Dy[Nphi], Dz[Nphi];
        for (int a = 0; a < Nphi; a++) {
          phi[a] = u(k, j, i, a);
          Dx[a] = (u(k, j, i + 1, a) - u(k, j, i - 1, a)) * inv2hx;
          Dy[a] = (u(k, j + 1, i, a) - u(k, j - 1, i, a)) * inv2hy;
          Dz[a] = (u(k + 1, j, i, a) - u(k - 1, j, i, a)) * inv2hz;
        }
        qv(k, j, i) = prefac * det4x4(phi, Dx, Dy, Dz);
      });
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewWrite(da1, q, &qv));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// GPU kernel: copy the field component c of the solution U into q (a global
// vector of da1)
static PetscErrorCode field_component_kokkos(DM da, Vec U, PetscInt c, DM da1,
                                             Vec q) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  ConstNodeView u;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, U, &u));
  SiteValueView qv;
  PetscCall(DMDAVecGetKokkosOffsetViewWrite(da1, q, &qv));

  Kokkos::parallel_for(
      "FieldComponent",
      SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        qv(k, j, i) = u(k, j, i, c);
      });
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewWrite(da1, q, &qv));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#ifdef SFGPU_CUFFT
// GPU kernel: bin scale*|ft(k)|^2 of the half spectrum ft (device memory)
// into the radial shells with atomic adds. A free function because nvcc
// rejects device lambdas in private members.
static void radial_binning_kokkos(const cufftDoubleComplex *ft, int Nlat,
                                  int nbins, double scale,
                                  const Kokkos::View<double *> &Sk,
                                  const Kokkos::View<double *> &Ns) {
  const int nxh = Nlat / 2 + 1;
  Kokkos::parallel_for(
      "RadialBinning",
      Kokkos::MDRangePolicy<
          Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
          {0, 0, 0}, {Nlat, Nlat, nxh}),
      KOKKOS_LAMBDA(int kz, int ky, int kx) {
        const int m = shell_index(kz, ky, kx, Nlat, nbins);
        const size_t idx =
            (static_cast<size_t>(kz) * Nlat + static_cast<size_t>(ky)) * nxh +
            static_cast<size_t>(kx);
        const cufftDoubleComplex z = ft[idx];
        Kokkos::atomic_add(&Sk(m), scale * (z.x * z.x + z.y * z.y));
        Kokkos::atomic_add(&Ns(m), 1.0);
      });
  Kokkos::fence();
}
#endif

// Bin scale*|ft(k)|^2 of the half spectrum ft (host memory) into the radial
// shells
static void radial_binning_host(const std::complex<double> *ft, int Nlat,
                                int nbins, double scale, double *Sk,
                                double *Ns) {
  const int nxh = Nlat / 2 + 1;
  for (int kz = 0; kz < Nlat; kz++) {
    for (int ky = 0; ky < Nlat; ky++) {
      for (int kx = 0; kx < nxh; kx++) {
        const int m = shell_index(kz, ky, kx, Nlat, nbins);
        const size_t idx =
            (static_cast<size_t>(kz) * Nlat + static_cast<size_t>(ky)) * nxh +
            static_cast<size_t>(kx);
        Sk[m] += scale * std::norm(ft[idx]);
        Ns[m] += 1.0;
      }
    }
  }
}

PetscErrorCode Measurer::setupSpectra() {
  PetscFunctionBeginUser;
  if (da1) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  // A 1-dof DMDA with the same layout as the domain, and the same vector
  // type (Kokkos vectors stay on the device)
  DM &da = model->domain;
  VecType vtype;
  PetscCall(DMDACreateCompatibleDMDA(da, 1, &da1));
  PetscCall(DMGetVecType(da, &vtype));
  PetscCall(DMSetVecType(da1, vtype));
  PetscCall(DMCreateGlobalVector(da1, &qglob));

  PetscMPIInt size, rank;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (size > 1) {
    PetscCall(DMDACreateNaturalVector(da1, &qnat));
    PetscCall(VecScatterCreateToZero(qnat, &toZero, &qseq));
  }
  if (rank == 0) {
    fftw3d = std::make_unique<measurer_fft_3d>(N, N, N);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Measurer::radialSpectrum(double *Sk, double *Nshell,
                                        std::complex<double> *zero) {
  PetscFunctionBeginUser;
  // Bring q to rank 0 in natural ordering, idx = i + N*j + N*N*k. With one
  // rank the global ordering of the DMDA already is the natural one.
  Vec qfull = qglob;
  if (toZero) {
    PetscCall(DMDAGlobalToNaturalBegin(da1, qglob, INSERT_VALUES, qnat));
    PetscCall(DMDAGlobalToNaturalEnd(da1, qglob, INSERT_VALUES, qnat));
    PetscCall(
        VecScatterBegin(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(
        VecScatterEnd(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD));
    qfull = qseq;
  }
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // S(k) = <|q(k)|^2> per shell with q(k) = (1/N^3) \sum_x e^{-ikx} q(x)
  const int nbins = NtopchargeBins;
  const double ntot = static_cast<double>(fftw3d->size_in());
  const double scale = 1.0 / (ntot * ntot);
  std::fill(Sk, Sk + nbins, 0.0);
  std::fill(Nshell, Nshell + nbins, 0.0);

  const PetscScalar *q;
  PetscMemType mtype;
  PetscCall(VecGetArrayReadAndMemType(qfull, &q, &mtype));
#ifdef SFGPU_CUFFT
  if (PetscMemTypeDevice(mtype)) {
    // FFT and binning on the device; only the shells come back to the host
    Kokkos::fence();
    const cufftDoubleComplex *ft = fftw3d->execute_device(q);
    Kokkos::View<double *> Sk_dev("Sk", nbins);
    Kokkos::View<double *> Ns_dev("Nshell", nbins);
    radial_binning_kokkos(ft, N, nbins, scale, Sk_dev, Ns_dev);
    auto Sk_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Sk_dev);
    auto Ns_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ns_dev);
    for (int m = 0; m < nbins; m++) {
      Sk[m] = Sk_host(m);
      Nshell[m] = Ns_host(m);
    }
    cufftDoubleComplex z0;
    cudaMemcpy(&z0, ft, sizeof(z0), cudaMemcpyDeviceToHost);
    *zero = std::complex<double>(z0.x, z0.y) / ntot;
  } else
#endif
  {
    (void)mtype;
    const std::complex<double> *ft = fftw3d->execute_host(q);
    radial_binning_host(ft, N, nbins, scale, Sk, Nshell);
    *zero = ft[0] / ntot;
  }
  PetscCall(VecRestoreArrayReadAndMemType(qfull, &q));

  for (int m = 0; m < nbins; m++) {
    if (Nshell[m] > 0.0) {
      Sk[m] /= Nshell[m];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

void Measurer::computeTopChargeFourier(Vec *solution) {
  setupSpectra();

  DM &da = model->domain;
  Vec localU;
  DMGetLocalVector(da, &localU);
  ghost_exchange(da, *solution, localU);

  const auto &data = model->data;
  const double inv2hx = 1.0 / (2.0 * data.hX());
  const double inv2hy = 1.0 / (2.0 * data.hY());
  const double inv2hz = 1.0 / (2.0 * data.hZ());
  const double prefac = 1.0 / (12.0 * M_PI * M_PI);

#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(*solution)) {
    topcharge_kernel_kokkos(da, localU, da1, qglob, prefac, inv2hx, inv2hy,
                            inv2hz);
  } else
#endif
  {
    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
    G_node ***fld;
    DMDAVecGetArrayRead(da, localU, &fld);
    PetscScalar ***q;
    DMDAVecGetArray(da1, qglob, &q);
    for (PetscInt k = zs; k < zs + zm; k++) {
      for (PetscInt j = ys; j < ys + ym; j++) {
        for (PetscInt i = xs; i < xs + xm; i++) {
          double phi[ModelAData::Nphi], Dx[ModelAData::Nphi],
              Dy[ModelAData::Nphi], Dz[ModelAData::Nphi];
          for (int a = 0; a < ModelAData::Nphi; a++) {
            phi[a] = fld[k][j][i].f[a];
            Dx[a] = (fld[k][j][i + 1].f[a] - fld[k][j][i - 1].f[a]) * inv2hx;
            Dy[a] = (fld[k][j + 1][i].f[a] - fld[k][j - 1][i].f[a]) * inv2hy;
            Dz[a] = (fld[k + 1][j][i].f[a] - fld[k - 1][j][i].f[a]) * inv2hz;
          }
          q[k][j][i] = prefac * det4x4(phi, Dx, Dy, Dz);
        }
      }
    }
    DMDAVecRestoreArray(da1, qglob, &q);
    DMDAVecRestoreArrayRead(da, localU, &fld);
  }
  DMRestoreLocalVector(da, &localU);

  radialSpectrum(topcharge_Sk.data(), topcharge_Nshell.data(),
                 &topcharge_zero);
}

// -----------------------------------------------------------------------
// Hybrid exact/log binning for the field spectra
// -----------------------------------------------------------------------

// Allocate the FFT buffer for the field spectra (bin edges are set up in the
// constructor).  Called on first use so that the 3D FFT plan exists.
void Measurer::setupFieldBins() {
  if (!field_ft_buf.empty()) return; // already done
  setupSpectra();

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    field_ft_buf.resize(fftw3d->size_out());
#ifdef SFGPU_CUFFT
    cudaMalloc(&d_field_ft_buf,
               fftw3d->size_out() * sizeof(cufftDoubleComplex));
#endif
  }
}

// -----------------------------------------------------------------------
// Host binning: accumulate |pi|^2, |A|^2, Re[pi*conj(A)] using precomputed
// bin_map and weight arrays.  Im[pi*conj(A)] is exactly zero for real fields
// (Hermitian symmetry) and is not stored.
// -----------------------------------------------------------------------
static void field_binning_host(const std::complex<double> *ft_pi,
                               const std::complex<double> *ft_A,
                               size_t nmodes, double scale,
                               const int *bin_map, const double *weight,
                               double *Ppipi, double *PAA, double *ReC) {
  for (size_t idx = 0; idx < nmodes; idx++) {
    const int b = bin_map[idx];
    if (b < 0) continue;
    const double w = weight[idx] * scale;
    Ppipi[b] += w * std::norm(ft_pi[idx]);
    PAA[b]   += w * std::norm(ft_A[idx]);
    const std::complex<double> c = ft_pi[idx] * std::conj(ft_A[idx]);
    ReC[b]   += w * c.real();
  }
}

// Host binning of one spectrum: P += scale * weight * |ft|^2 per bin
static void field_binning_single_host(const std::complex<double> *ft,
                                      size_t nmodes, double scale,
                                      const int *bin_map,
                                      const double *weight, double *P) {
  for (size_t idx = 0; idx < nmodes; idx++) {
    const int b = bin_map[idx];
    if (b < 0) continue;
    P[b] += weight[idx] * scale * std::norm(ft[idx]);
  }
}

#ifdef SFGPU_CUFFT
// -----------------------------------------------------------------------
// GPU binning kernel for the field spectra (precomputed bin map on device)
// -----------------------------------------------------------------------
static void field_binning_kokkos(const cufftDoubleComplex *ft_pi,
                                 const cufftDoubleComplex *ft_A,
                                 size_t nmodes, double scale,
                                 const int *bin_map_d, const double *weight_d,
                                 const Kokkos::View<double *> &Ppipi,
                                 const Kokkos::View<double *> &PAA,
                                 const Kokkos::View<double *> &ReC) {
  Kokkos::parallel_for(
      "FieldBinning", static_cast<int>(nmodes),
      KOKKOS_LAMBDA(int idx) {
        const int b = bin_map_d[idx];
        if (b < 0) return;
        const double w = weight_d[idx] * scale;
        const cufftDoubleComplex zp = ft_pi[idx];
        const cufftDoubleComplex za = ft_A[idx];
        Kokkos::atomic_add(&Ppipi(b), w * (zp.x * zp.x + zp.y * zp.y));
        Kokkos::atomic_add(&PAA(b),   w * (za.x * za.x + za.y * za.y));
        Kokkos::atomic_add(&ReC(b),   w * (zp.x * za.x + zp.y * za.y));
      });
  Kokkos::fence();
}

// GPU binning of one spectrum: P += scale * weight * |ft|^2 per bin
static void field_binning_single_kokkos(const cufftDoubleComplex *ft,
                                        size_t nmodes, double scale,
                                        const int *bin_map_d,
                                        const double *weight_d,
                                        const Kokkos::View<double *> &P) {
  Kokkos::parallel_for(
      "FieldBinningSingle", static_cast<int>(nmodes),
      KOKKOS_LAMBDA(int idx) {
        const int b = bin_map_d[idx];
        if (b < 0) return;
        const cufftDoubleComplex z = ft[idx];
        Kokkos::atomic_add(&P(b), weight_d[idx] * scale * (z.x * z.x + z.y * z.y));
      });
  Kokkos::fence();
}
#endif

// -----------------------------------------------------------------------
// GPU kernel for A_a(x) = contract_rho(n(x), rho(x))[a] with local
// n_a(x) = phi_a(x)/|phi(x)|, matching computeSliceAveragePhase. With dual,
// V_a(x) = contract_rho(n(x), dualize_rho(rho(x)))[a] instead.
// -----------------------------------------------------------------------
#ifdef SFGPU_KOKKOS
static PetscErrorCode Avec_component_local_kokkos(DM da, Vec U, PetscInt a,
                                                  bool dual, DM da1, Vec q) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  ConstNodeView u;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, U, &u));
  SiteValueView qv;
  PetscCall(DMDAVecGetKokkosOffsetViewWrite(da1, q, &qv));

  constexpr PetscInt oA = ModelAData::OffsetA;
  constexpr PetscInt Nphi = ModelAData::Nphi;
  Kokkos::parallel_for(
      "AvecComponentLocal",
      SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        // Local unit vector n = phi/|phi|
        double n[Nphi];
        double phi2 = 0.0;
        for (int l = 0; l < Nphi; l++) {
          n[l] = u(k, j, i, l);
          phi2 += n[l] * n[l];
        }
        if (phi2 > 100.0 * 2.2250738585072014e-308) { // ~numeric_limits<double>::min()
          const double inv = 1.0 / sqrt(phi2);
          for (int l = 0; l < Nphi; l++) n[l] *= inv;
        } else {
          n[0] = 1.0; n[1] = 0.0; n[2] = 0.0; n[3] = 0.0;
        }
        // rho stored as A[3],V[3] at offsets oA..oA+5; dualize_rho swaps
        // the two halves
        const PetscInt o1 = oA + (dual ? 3 : 0), o2 = oA + (dual ? 0 : 3);
        const double r0 = u(k, j, i, o1 + 0);
        const double r1 = u(k, j, i, o1 + 1);
        const double r2 = u(k, j, i, o1 + 2);
        const double r3 = u(k, j, i, o2 + 0);
        const double r4 = u(k, j, i, o2 + 1);
        const double r5 = u(k, j, i, o2 + 2);
        double result;
        switch (a) {
        case 0: result = -(n[1]*r0) - n[2]*r1 - n[3]*r2;  break;
        case 1: result = n[0]*r0 + n[3]*r4 - n[2]*r5;     break;
        case 2: result = n[0]*r1 - n[3]*r3 + n[1]*r5;     break;
        default: result = n[0]*r2 + n[2]*r3 - n[1]*r4;    break;
        }
        qv(k, j, i) = result;
      });
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewWrite(da1, q, &qv));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// q = A_a(x) = contract_rho(n(x), rho(x))[a] with the local direction
// n(x) = phi(x)/|phi(x)| of the solution-like vector U, or with dual the
// vector density V_a(x) = contract_rho(n(x), dualize_rho(rho(x)))[a]
static void rotated_charge_component(DM da, Vec U, PetscInt a, bool dual,
                                     DM da1, Vec q) {
#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(U)) {
    Avec_component_local_kokkos(da, U, a, dual, da1, q);
    return;
  }
#endif
  PetscInt xs, ys, zs, xm, ym, zm;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
  PetscScalar ****u;
  DMDAVecGetArrayDOFRead(da, U, &u);
  PetscScalar ***qa;
  DMDAVecGetArray(da1, q, &qa);
  const PetscInt o1 = ModelAData::OffsetA + (dual ? 3 : 0);
  const PetscInt o2 = ModelAData::OffsetA + (dual ? 0 : 3);
  for (PetscInt k = zs; k < zs + zm; k++) {
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        // Local unit vector n = phi/|phi|
        double n[ModelAData::Nphi];
        double phi2 = 0.0;
        for (int l = 0; l < ModelAData::Nphi; l++) {
          n[l] = u[k][j][i][l];
          phi2 += n[l] * n[l];
        }
        if (phi2 > 100.0 * std::numeric_limits<double>::min()) {
          const double inv = 1.0 / sqrt(phi2);
          for (int l = 0; l < ModelAData::Nphi; l++) n[l] *= inv;
        } else {
          n[0] = 1.0; n[1] = 0.0; n[2] = 0.0; n[3] = 0.0;
        }
        const double r0 = u[k][j][i][o1 + 0];
        const double r1 = u[k][j][i][o1 + 1];
        const double r2 = u[k][j][i][o1 + 2];
        const double r3 = u[k][j][i][o2 + 0];
        const double r4 = u[k][j][i][o2 + 1];
        const double r5 = u[k][j][i][o2 + 2];
        double result;
        switch (a) {
        case 0: result = -(n[1]*r0) - n[2]*r1 - n[3]*r2;  break;
        case 1: result = n[0]*r0 + n[3]*r4 - n[2]*r5;     break;
        case 2: result = n[0]*r1 - n[3]*r3 + n[1]*r5;     break;
        default: result = n[0]*r2 + n[2]*r3 - n[1]*r4;    break;
        }
        qa[k][j][i] = result;
      }
    }
  }
  DMDAVecRestoreArray(da1, q, &qa);
  DMDAVecRestoreArrayDOFRead(da, U, &u);
}

Vec Measurer::gatherQToZero() {
  if (!toZero) {
    return qglob;
  }
  DMDAGlobalToNaturalBegin(da1, qglob, INSERT_VALUES, qnat);
  DMDAGlobalToNaturalEnd(da1, qglob, INSERT_VALUES, qnat);
  VecScatterBegin(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
  return qseq;
}

// -----------------------------------------------------------------------
// computeFieldSpectra: power spectra of phi_a and locally-rotated A_a
// with hybrid exact/log binning.
// phi_a(x) is the raw field component.
// A_a(x) = contract_rho(n(x), rho(x))[a] with n(x) = phi(x)/|phi(x)|,
// matching the local decomposition in computeSliceAveragePhase.
// -----------------------------------------------------------------------
void Measurer::computeFieldSpectra(Vec *solution) {
  setupFieldBins();

  DM &da = model->domain;

  // 1. Zero accumulators (kbins and Nshell are time-independent, set in ctor)
  const int nb = NFieldBins;
  std::fill(field_Pphi.begin(), field_Pphi.end(), 0.0);
  std::fill(field_PAA.begin(), field_PAA.end(), 0.0);
  std::fill(field_ReC.begin(), field_ReC.end(), 0.0);
  std::fill(field_PVV.begin(), field_PVV.end(), 0.0);

  const double ntot = static_cast<double>(N) * N * N;
  const double scale = 1.0 / (ntot * ntot);
  // fftw3d exists on rank 0 only
  const size_t nmodes = fftw3d ? fftw3d->size_out() : 0;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // 2. Loop over components a = 0..Nphi-1
  for (int a = 0; a < ModelAData::Nphi; a++) {

    // --- Copy phi_a(x) into qglob ---
#ifdef SFGPU_KOKKOS
    if (is_kokkos_vec(*solution)) {
      field_component_kokkos(da, *solution, a, da1, qglob);
    } else
#endif
    {
      PetscInt xs, ys, zs, xm, ym, zm;
      DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
      PetscScalar ****u;
      DMDAVecGetArrayDOFRead(da, *solution, &u);
      PetscScalar ***q;
      DMDAVecGetArray(da1, qglob, &q);
      for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
          for (PetscInt i = xs; i < xs + xm; i++) {
            q[k][j][i] = u[k][j][i][a];
          }
        }
      }
      DMDAVecRestoreArray(da1, qglob, &q);
      DMDAVecRestoreArrayDOFRead(da, *solution, &u);
    }

    // --- Gather phi_a to rank 0 and FFT; save result in buffer ---
    Vec qfull = qglob;
    if (toZero) {
      DMDAGlobalToNaturalBegin(da1, qglob, INSERT_VALUES, qnat);
      DMDAGlobalToNaturalEnd(da1, qglob, INSERT_VALUES, qnat);
      VecScatterBegin(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      VecScatterEnd(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      qfull = qseq;
    }
    if (rank == 0) {
      const PetscScalar *qdata;
      PetscMemType mtype;
      VecGetArrayReadAndMemType(qfull, &qdata, &mtype);
#ifdef SFGPU_CUFFT
      if (PetscMemTypeDevice(mtype)) {
        Kokkos::fence();
        const cufftDoubleComplex *ft = fftw3d->execute_device(qdata);
        cudaMemcpy(d_field_ft_buf, ft, nmodes * sizeof(cufftDoubleComplex),
                   cudaMemcpyDeviceToDevice);
      } else
#endif
      {
        (void)mtype;
        const std::complex<double> *ft = fftw3d->execute_host(qdata);
        std::copy(ft, ft + nmodes, field_ft_buf.begin());
      }
      VecRestoreArrayReadAndMemType(qfull, &qdata);
    }

    // --- Build A_a(x) = contract_rho(n(x), rho(x))[a] with local n(x) ---
    rotated_charge_component(da, *solution, a, false, da1, qglob);

    // --- Gather A_a to rank 0, FFT, and bin ---
    qfull = qglob;
    if (toZero) {
      DMDAGlobalToNaturalBegin(da1, qglob, INSERT_VALUES, qnat);
      DMDAGlobalToNaturalEnd(da1, qglob, INSERT_VALUES, qnat);
      VecScatterBegin(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      VecScatterEnd(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      qfull = qseq;
    }
    if (rank == 0) {
      const PetscScalar *qdata;
      PetscMemType mtype;
      VecGetArrayReadAndMemType(qfull, &qdata, &mtype);
#ifdef SFGPU_CUFFT
      if (PetscMemTypeDevice(mtype)) {
        Kokkos::fence();
        const cufftDoubleComplex *ft_A = fftw3d->execute_device(qdata);
        // Precomputed bin map and weight on the device, uploaded once
        if (!d_field_bin_map) {
          if (cudaMalloc(&d_field_bin_map, nmodes * sizeof(int)) !=
                  cudaSuccess ||
              cudaMalloc(&d_field_weight, nmodes * sizeof(double)) !=
                  cudaSuccess) {
            throw std::runtime_error(
                "computeFieldSpectra: cudaMalloc of the bin map failed");
          }
          cudaMemcpy(d_field_bin_map, field_bin_map.data(),
                     nmodes * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(d_field_weight, field_weight.data(),
                     nmodes * sizeof(double), cudaMemcpyHostToDevice);
        }

        Kokkos::View<double *> Ppipi_d("Ppipi", nb);
        Kokkos::View<double *> PAA_d("PAA", nb);
        Kokkos::View<double *> ReC_d("ReC", nb);
        if (a > 0) {
          auto Ppipi_h = Kokkos::create_mirror_view(Ppipi_d);
          auto PAA_h = Kokkos::create_mirror_view(PAA_d);
          auto ReC_h = Kokkos::create_mirror_view(ReC_d);
          for (int b = 0; b < nb; b++) {
            Ppipi_h(b) = field_Pphi[b];
            PAA_h(b) = field_PAA[b];
            ReC_h(b) = field_ReC[b];
          }
          Kokkos::deep_copy(Ppipi_d, Ppipi_h);
          Kokkos::deep_copy(PAA_d, PAA_h);
          Kokkos::deep_copy(ReC_d, ReC_h);
        }

        field_binning_kokkos(d_field_ft_buf, ft_A, nmodes, scale,
                             d_field_bin_map, d_field_weight, Ppipi_d, PAA_d,
                             ReC_d);

        auto Ppipi_h2 =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ppipi_d);
        auto PAA_h2 =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), PAA_d);
        auto ReC_h2 =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ReC_d);
        for (int b = 0; b < nb; b++) {
          field_Pphi[b] = Ppipi_h2(b);
          field_PAA[b] = PAA_h2(b);
          field_ReC[b] = ReC_h2(b);
        }
      } else
#endif
      {
        (void)mtype;
        const std::complex<double> *ft_A = fftw3d->execute_host(qdata);
        field_binning_host(field_ft_buf.data(), ft_A, nmodes, scale,
                           field_bin_map.data(), field_weight.data(),
                           field_Pphi.data(), field_PAA.data(),
                           field_ReC.data());
      }
      VecRestoreArrayReadAndMemType(qfull, &qdata);
    }

    // --- With vector_spectra: V_a(x) with the same local n(x), |V_a(k)|^2 ---
    if (model->data.ahandler.vector_spectra) {
      rotated_charge_component(da, *solution, a, true, da1, qglob);
      qfull = gatherQToZero();
      if (rank == 0) {
        const PetscScalar *qdata;
        PetscMemType mtype;
        VecGetArrayReadAndMemType(qfull, &qdata, &mtype);
#ifdef SFGPU_CUFFT
        if (PetscMemTypeDevice(mtype)) {
          Kokkos::fence();
          const cufftDoubleComplex *ft_V = fftw3d->execute_device(qdata);
          Kokkos::View<double *> PVV_d("PVV", nb);
          auto PVV_h = Kokkos::create_mirror_view(PVV_d);
          for (int b = 0; b < nb; b++) PVV_h(b) = field_PVV[b];
          Kokkos::deep_copy(PVV_d, PVV_h);
          field_binning_single_kokkos(ft_V, nmodes, scale, d_field_bin_map,
                                      d_field_weight, PVV_d);
          Kokkos::deep_copy(PVV_h, PVV_d);
          for (int b = 0; b < nb; b++) field_PVV[b] = PVV_h(b);
        } else
#endif
        {
          (void)mtype;
          const std::complex<double> *ft_V = fftw3d->execute_host(qdata);
          field_binning_single_host(ft_V, nmodes, scale, field_bin_map.data(),
                                    field_weight.data(), field_PVV.data());
        }
        VecRestoreArrayReadAndMemType(qfull, &qdata);
      }
    }
  } // end loop over a

  // 4. Normalize: divide by the (time-independent) full k-space mode count
  //    per bin, summed over Nphi components.  Nshell was precomputed in the
  //    constructor per component; multiply by Nphi for the trace.
  if (rank == 0) {
    for (int b = 0; b < nb; b++) {
      const double Nb = field_Nshell[b] * ModelAData::Nphi;
      if (Nb > 0.0) {
        field_Pphi[b] /= Nb;
        field_PAA[b] /= Nb;
        field_ReC[b] /= Nb;
        field_PVV[b] /= Nb;
      }
    }
  }
}

// Given the fourier transform of phi_a(t,k) and other fields
// stored in the wallk strucutre, compute the rotated fields
// The zero mode has phi(t,0) defines a unit four vector n_a
//
// wallk_rotated contains  sigma = n_a phi_a and pi_b = rho_{ab} n_b
// as well as the axial vector density  A_a = rho_{ab} n_b and
// corresponding vector density V_a. Finally phi2 is also
// stored in this structure
// -----------------------------------------------------------------------
// Analytical coarsening for simple dissipation (preserve_norm = false), see
// Measurer::setupAnalyticalCoarsening
// -----------------------------------------------------------------------

// Eigenvalues (2 - 2 cos(2 pi m / N)) / h^2, m = 0..N-1, of the 3-point
// second difference along one axis, as in the diffusion stencil
static std::vector<double> laplacian_1d(PetscInt N, double h) {
  std::vector<double> lam(N);
  for (PetscInt m = 0; m < N; m++) {
    lam[m] = (2. - 2. * cos(2. * M_PI * static_cast<double>(m) / N)) / (h * h);
  }
  return lam;
}

// work = (ft(k) exp(-coef_tau lambda(k)) + zero_shift [k = 0]) * inv_ntot on
// the half spectrum, idx = (kz * N + ky) * (N/2+1) + kx. Host version.
static void decay_spectrum_host(const std::complex<double> *ft,
                                std::complex<double> *work, PetscInt N,
                                const std::vector<double> &lx,
                                const std::vector<double> &ly,
                                const std::vector<double> &lz,
                                double coef_tau, double zero_shift,
                                double inv_ntot) {
  const PetscInt nxh = N / 2 + 1;
  for (PetscInt kz = 0; kz < N; kz++) {
    for (PetscInt ky = 0; ky < N; ky++) {
      for (PetscInt kx = 0; kx < nxh; kx++) {
        const size_t idx = (static_cast<size_t>(kz) * N + ky) * nxh + kx;
        const double f = exp(-coef_tau * (lx[kx] + ly[ky] + lz[kz]));
        work[idx] = ft[idx] * (f * inv_ntot);
      }
    }
  }
  work[0] += zero_shift * inv_ntot;
}

#ifdef SFGPU_CUFFT
// Device version of decay_spectrum_host. A free function because nvcc does
// not allow lambdas in private member functions.
static void decay_spectrum_kokkos(const cufftDoubleComplex *ft,
                                  cufftDoubleComplex *work, PetscInt N,
                                  const std::vector<double> &lx_h,
                                  const std::vector<double> &ly_h,
                                  const std::vector<double> &lz_h,
                                  double coef_tau, double zero_shift,
                                  double inv_ntot) {
  Kokkos::View<double *> lx("lx", N), ly("ly", N), lz("lz", N);
  {
    auto hx = Kokkos::create_mirror_view(lx);
    auto hy = Kokkos::create_mirror_view(ly);
    auto hz = Kokkos::create_mirror_view(lz);
    for (PetscInt m = 0; m < N; m++) {
      hx(m) = lx_h[m];
      hy(m) = ly_h[m];
      hz(m) = lz_h[m];
    }
    Kokkos::deep_copy(lx, hx);
    Kokkos::deep_copy(ly, hy);
    Kokkos::deep_copy(lz, hz);
  }
  const size_t nxh = N / 2 + 1;
  const size_t nmodes = static_cast<size_t>(N) * N * nxh;
  const size_t nN = N;
  Kokkos::parallel_for(
      "DecaySpectrum", Kokkos::RangePolicy<Kokkos::IndexType<size_t>>(0, nmodes),
      KOKKOS_LAMBDA(size_t idx) {
        const size_t kx = idx % nxh;
        const size_t ky = (idx / nxh) % nN;
        const size_t kz = idx / (nxh * nN);
        const double f =
            Kokkos::exp(-coef_tau * (lx(kx) + ly(ky) + lz(kz))) * inv_ntot;
        cufftDoubleComplex w;
        w.x = ft[idx].x * f;
        w.y = ft[idx].y * f;
        if (idx == 0) {
          w.x += zero_shift * inv_ntot;
        }
        work[idx] = w;
      });
  Kokkos::fence();
}
#endif

// Copy component c of the solution-like vector U into the 1-dof vector q
static void get_field_component(DM da, Vec U, PetscInt c, DM da1, Vec q) {
#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(U)) {
    field_component_kokkos(da, U, c, da1, q);
    return;
  }
#endif
  PetscInt xs, ys, zs, xm, ym, zm;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
  PetscScalar ****u;
  DMDAVecGetArrayDOFRead(da, U, &u);
  PetscScalar ***qa;
  DMDAVecGetArray(da1, q, &qa);
  for (PetscInt k = zs; k < zs + zm; k++)
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++)
        qa[k][j][i] = u[k][j][i][c];
  DMDAVecRestoreArray(da1, q, &qa);
  DMDAVecRestoreArrayDOFRead(da, U, &u);
}

#ifdef SFGPU_KOKKOS
// GPU kernel: write the 1-dof vector q into component c of U, leaving the
// other components unchanged
static PetscErrorCode set_field_component_kokkos(DM da, Vec U, PetscInt c,
                                                 DM da1, Vec q) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  NodeView u;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, U, &u));
  Kokkos::Experimental::OffsetView<const PetscScalar ***, Kokkos::LayoutRight,
                                   KokkosMemSpace>
      qv;
  PetscCall(DMDAVecGetKokkosOffsetView(da1, q, &qv));
  Kokkos::parallel_for(
      "SetFieldComponent",
      SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        u(k, j, i, c) = qv(k, j, i);
      });
  Kokkos::fence();
  PetscCall(DMDAVecRestoreKokkosOffsetView(da1, q, &qv));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// Write the 1-dof vector q into component c of U
static void set_field_component(DM da, Vec U, PetscInt c, DM da1, Vec q) {
#ifdef SFGPU_KOKKOS
  if (is_kokkos_vec(U)) {
    set_field_component_kokkos(da, U, c, da1, q);
    return;
  }
#endif
  PetscInt xs, ys, zs, xm, ym, zm;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
  PetscScalar ****u;
  DMDAVecGetArrayDOF(da, U, &u);
  const PetscScalar ***qa;
  DMDAVecGetArrayRead(da1, q, &qa);
  for (PetscInt k = zs; k < zs + zm; k++)
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++)
        u[k][j][i][c] = qa[k][j][i];
  DMDAVecRestoreArrayRead(da1, q, &qa);
  DMDAVecRestoreArrayDOF(da, U, &u);
}

void Measurer::setupAnalyticalCoarsening(Vec *U) {
  setupSpectra();
  finalizeAnalyticalCoarsening();
  DM &da = model->domain;
  const size_t nmodes = fftw3d ? fftw3d->size_out() : 0;
  constexpr PetscInt Ndof = ModelAData::Ndof;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (PetscInt c = 0; c < Ndof; c++) {
    // Component c to rank 0 in natural ordering (as in computeFieldSpectra)
    get_field_component(da, *U, c, da1, qglob);
    Vec qfull = qglob;
    if (toZero) {
      DMDAGlobalToNaturalBegin(da1, qglob, INSERT_VALUES, qnat);
      DMDAGlobalToNaturalEnd(da1, qglob, INSERT_VALUES, qnat);
      VecScatterBegin(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      VecScatterEnd(toZero, qnat, qseq, INSERT_VALUES, SCATTER_FORWARD);
      qfull = qseq;
    }
    if (rank == 0) {
      const PetscScalar *qdata;
      PetscMemType mtype;
      VecGetArrayReadAndMemType(qfull, &qdata, &mtype);
#ifdef SFGPU_CUFFT
      if (PetscMemTypeDevice(mtype)) {
        if (!d_coarse_ft) {
          if (cudaMalloc(&d_coarse_ft,
                         Ndof * nmodes * sizeof(cufftDoubleComplex)) !=
                  cudaSuccess ||
              cudaMalloc(&d_coarse_work,
                         nmodes * sizeof(cufftDoubleComplex)) != cudaSuccess) {
            throw std::runtime_error(
                "analytical coarsening: cudaMalloc of the spectra failed");
          }
          coarse_on_device = true;
        }
        Kokkos::fence();
        const cufftDoubleComplex *ft = fftw3d->execute_device(qdata);
        cudaMemcpy(d_coarse_ft + c * nmodes, ft,
                   nmodes * sizeof(cufftDoubleComplex),
                   cudaMemcpyDeviceToDevice);
      } else
#endif
      {
        (void)mtype;
        if (coarse_ft.empty()) {
          coarse_ft.resize(Ndof * nmodes);
          coarse_work.resize(nmodes);
        }
        const std::complex<double> *ft = fftw3d->execute_host(qdata);
        std::copy(ft, ft + nmodes, coarse_ft.begin() + c * nmodes);
      }
      VecRestoreArrayReadAndMemType(qfull, &qdata);
    }
  }
}

void Measurer::analyticalCoarsenedField(double tau, Vec *out) {
  DM &da = model->domain;
  const auto &data = model->data;
  const auto &coeff = data.acoefficients;
  const size_t nmodes = fftw3d ? fftw3d->size_out() : 0;
  constexpr PetscInt Ndof = ModelAData::Ndof;
  const double ntot = static_cast<double>(N) * N * N;
  const std::vector<double> lx = laplacian_1d(N, data.hX());
  const std::vector<double> ly = laplacian_1d(N, data.hY());
  const std::vector<double> lz = laplacian_1d(N, data.hZ());

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (PetscInt c = 0; c < Ndof; c++) {
    // phi diffuses with Gamma and is driven by Gamma H sigma/f (phi_0 only),
    // the charges diffuse with D, as in ModelGExplicitDiffusionStep::step_on
    const bool is_phi = c < ModelAData::Nphi;
    const double coef_tau = (is_phi ? coeff.Gamma() : coeff.D()) * tau;
    const double zero_shift =
        (c == 0) ? ntot * coeff.Gamma() * coeff.H *
                       coeff.sigmabyf(data.atime.t()) * tau
                 : 0.;

    Vec qfull = toZero ? qseq : qglob;
    if (rank == 0) {
      PetscScalar *qdata;
      PetscMemType mtype;
      VecGetArrayWriteAndMemType(qfull, &qdata, &mtype);
#ifdef SFGPU_CUFFT
      if (coarse_on_device) {
        decay_spectrum_kokkos(d_coarse_ft + c * nmodes, d_coarse_work, N, lx,
                              ly, lz, coef_tau, zero_shift, 1. / ntot);
        if (PetscMemTypeDevice(mtype)) {
          fftw3d->execute_inverse_device(d_coarse_work, qdata);
        } else {
          throw std::runtime_error(
              "analytical coarsening: spectra on the device but the field "
              "vector on the host");
        }
      } else
#endif
      {
        (void)mtype;
        decay_spectrum_host(coarse_ft.data() + c * nmodes, coarse_work.data(),
                            N, lx, ly, lz, coef_tau, zero_shift, 1. / ntot);
        fftw3d->execute_inverse_host(coarse_work.data(), qdata);
      }
      VecRestoreArrayWriteAndMemType(qfull, &qdata);
    }
    // Back from rank 0 in natural ordering to the distributed global vector
    if (toZero) {
      VecScatterBegin(toZero, qseq, qnat, INSERT_VALUES, SCATTER_REVERSE);
      VecScatterEnd(toZero, qseq, qnat, INSERT_VALUES, SCATTER_REVERSE);
      DMDANaturalToGlobalBegin(da1, qnat, INSERT_VALUES, qglob);
      DMDANaturalToGlobalEnd(da1, qnat, INSERT_VALUES, qglob);
    }
    set_field_component(da, *out, c, da1, qglob);
  }
}

void Measurer::finalizeAnalyticalCoarsening() {
  coarse_ft.clear();
  coarse_ft.shrink_to_fit();
  coarse_work.clear();
  coarse_work.shrink_to_fit();
#ifdef SFGPU_CUFFT
  if (d_coarse_ft) {
    cudaFree(d_coarse_ft);
    d_coarse_ft = nullptr;
  }
  if (d_coarse_work) {
    cudaFree(d_coarse_work);
    d_coarse_work = nullptr;
  }
#endif
  coarse_on_device = false;
}

void rotate_to_vev(nvector<std::complex<double>, 2> &wallk,
                   nvector<std::complex<double>, 2> &wallk_rotated) {
  std::array<std::complex<double>, 4> n{};
  std::array<std::complex<double>, 4> phi{};
  std::array<std::complex<double>, 4> phir{};
  std::array<std::complex<double>, 6> rho{};
  std::array<std::complex<double>, 4> V{};
  std::array<std::complex<double>, 4> A{};

  // Extract the vev direction
  double norm = 0.;
  for (int a = 0; a < 4; a++) {
    n[a] = wallk(a, 0); // Extract the zero mode  of phi_a
    norm += std::norm(n[a]);
  }

  norm = sqrt(norm);
  if (norm < std::numeric_limits<double>::min()) {
    n = std::array<std::complex<double>, 4>{1, 0, 0, 0};
  } else {
    for (size_t a = 0; a < 4; a++) {
      n[a] /= norm;
    }
  }

  // Loop over momenta and do the rotation to vev
  for (size_t k = 0; k < wallk.N[1]; k++) {
    for (size_t a = 0; a < 4; a++) {
      phi[a] = wallk(a, k);
    }

    for (size_t ab = 0; ab < 6; ab++) {
      rho[ab] = wallk(4 + ab, k);
    }
    std::complex<double> phi2k = wallk(10, k);

    std::complex<double> phis(0.);
    for (size_t a = 0; a < 4; a++) {
      phis += n[a] * phi[a];
    }
    for (size_t a = 0; a < 4; a++) {
      phir[a] = phi[a] - n[a] * phis;
    }

    A = contract_rho(n, rho);
    V = contract_rho(n, dualize_rho(rho));

    wallk_rotated(0, k) = phis;
    for (size_t a = 0; a < 4; a++) {
      wallk_rotated(1 + a, k) = phir[a];
      wallk_rotated(5 + a, k) = A[a];
      wallk_rotated(9 + a, k) = V[a];
    }
    wallk_rotated(13, k) = phi2k;
  }
}

void Measurer::computeDerivedObs() {
  // NB: the intent is that this is to be called only
  // from the rank=0

  // Initializing the average to zero
  OAverage = std::vector<PetscScalar>(NScalars, 0.);

  // Compute the spatial correlation function of wall averages
  for (int l = 0; l < NObs; l++) {
    for (int i = 0; i < N; i++) {
      OAverage[l] += wallX(l, i);
    }
  }
  // Compute <X>
  for (int l = 0; l < NObs; l++) {
    OAverage[l] /= PetscReal(N);
  }

  fftw->execute(wallX, wallX_k);
  fftw->execute(wallY, wallY_k);
  fftw->execute(wallZ, wallZ_k);

  rotate_to_vev(wallX_k, wallX_k_rotated);
  rotate_to_vev(wallY_k, wallY_k_rotated);
  rotate_to_vev(wallZ_k, wallZ_k_rotated);

  fftw->execute(wallXPhase, wallXPhase_k);
  fftw->execute(wallYPhase, wallYPhase_k);
  fftw->execute(wallZPhase, wallZPhase_k);
}
