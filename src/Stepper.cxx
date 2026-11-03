// First: PETSc requires its Kokkos headers before any other PETSc header
#include "KokkosTypes.h"
#include "Stepper.h"
#include "ModelA.h"
#include "O4AlgebraHelper.h"
#include <algorithm>
#include <cstdio>

///////////////////////////////////////////////////////////////////////////

IdealPV2::IdealPV2(ModelA &in)
    : model(&in), da(model->domain), data(model->data) {

  // Store for later use
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);
#ifdef SFGPU_KOKKOS
  use_kokkos = is_kokkos_vec(model->solution);
#endif
}

// Rotates phi at one site by the charges A, V (converted to chemical
// potentials). Shared by the host loops and the Kokkos kernels.
O4_FUNCTION void rotate_by_charges(PetscScalar *f, const PetscScalar *A,
                                   const PetscScalar *V, PetscReal chi,
                                   PetscReal dt) {
  PetscScalar axialmu[ModelAData::NA], vectormu[ModelAData::NV];

  for (PetscInt s = 0; s < ModelAData::NV; s++) {
    vectormu[s] = -V[s] / chi * dt;
  }

  for (PetscInt s = 0; s < ModelAData::NA; s++) {
    axialmu[s] = -A[s] / chi * dt;
  }

  O4AlgebraHelper::O4Rotation(vectormu, axialmu, f);
}

#ifdef SFGPU_KOKKOS
// Device version of IdealPV2::step_no_reject: first half-drift, ghost
// exchange, then the kick fused with the second half-drift. Same arithmetic
// in the same order as the host loops; only sin/cos differ in the last bit.
static PetscErrorCode ideal_step_kokkos(DM da, Vec solution, PetscReal dt,
                                        PetscReal chi, PetscReal H0,
                                        PetscReal axx, PetscReal ayy,
                                        PetscReal azz) {
  const auto &prof = Profiling::get();
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  const SitePolicy owned({zs, ys, xs}, {zs + zm, ys + ym, xs + xm});
  const PetscReal hdt = dt / 2.0;
  constexpr PetscInt Nphi = ModelAData::Nphi, NA = ModelAData::NA,
                     NV = ModelAData::NV;
  constexpr PetscInt OffsetA = ModelAData::OffsetA,
                     OffsetV = ModelAData::OffsetV;

  // drifts the solution by dt / 2.0
  PetscCall(PetscLogEventBegin(prof.ideal_rotate, 0, 0, 0, 0));
  {
    NodeView u;
    PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, solution, &u));
    Kokkos::parallel_for(
        "IdealRotate", owned, KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
          PetscScalar f[Nphi], A[NA], V[NV];
          for (int c = 0; c < Nphi; c++) f[c] = u(k, j, i, c);
          for (int c = 0; c < NA; c++) A[c] = u(k, j, i, OffsetA + c);
          for (int c = 0; c < NV; c++) V[c] = u(k, j, i, OffsetV + c);
          rotate_by_charges(f, A, V, chi, hdt);
          for (int c = 0; c < Nphi; c++) u(k, j, i, c) = f[c];
        });
    Kokkos::fence();
    PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, solution, &u));
  }
  PetscCall(PetscLogEventEnd(prof.ideal_rotate, 0, 0, 0, 0));

  // Fill in the ghost cells
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(ghost_exchange(da, solution, localU));

  PetscCall(PetscLogEventBegin(prof.ideal_kick, 0, 0, 0, 0));
  {
    ConstNodeView phi;
    NodeView phinew;
    PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, localU, &phi));
    PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, solution, &phinew));
    Kokkos::parallel_for(
        "IdealKick", owned, KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
          // phi components of the site and its six neighbors
          PetscScalar c[Nphi], xp[Nphi], xm[Nphi], yp[Nphi], ym[Nphi],
              zp[Nphi], zm[Nphi];
          for (int l = 0; l < Nphi; l++) {
            c[l] = phi(k, j, i, l);
            xp[l] = phi(k, j, i + 1, l);
            xm[l] = phi(k, j, i - 1, l);
            yp[l] = phi(k, j + 1, i, l);
            ym[l] = phi(k, j - 1, i, l);
            zp[l] = phi(k + 1, j, i, l);
            zm[l] = phi(k - 1, j, i, l);
          }
          PetscScalar A[NA], V[NV], f[Nphi];
          for (int l = 0; l < NA; l++) A[l] = phinew(k, j, i, OffsetA + l);
          for (int l = 0; l < NV; l++) V[l] = phinew(k, j, i, OffsetV + l);
          for (int l = 0; l < Nphi; l++) f[l] = phinew(k, j, i, l);

          PetscScalar advxx, advyy, advzz;
          for (PetscInt l = 0; l < NA; l++) {
            advxx = (-xp[0] * c[l + 1] + c[0] * xp[l + 1] + c[0] * xm[l + 1] -
                     xm[0] * c[l + 1]) *
                    axx;
            advyy = (-yp[0] * c[l + 1] + c[0] * yp[l + 1] + c[0] * ym[l + 1] -
                     ym[0] * c[l + 1]) *
                    ayy;
            advzz = (-zp[0] * c[l + 1] + c[0] * zp[l + 1] + c[0] * zm[l + 1] -
                     zm[0] * c[l + 1]) *
                    azz;
            A[l] += dt * (advxx + advyy + advzz - H0 * c[l + 1]);
          }

          PetscInt s1, s2, epsilon;
          for (PetscInt s = 0; s < NV; s++) {
            s1 = (s + 1) % 3;
            s2 = (s + 2) % 3;
            epsilon = ((PetscScalar)(s - s1) * (s1 - s2) * (s2 - s)) / 2.;
            advxx = epsilon *
                    (xp[1 + s2] * c[1 + s1] - c[1 + s2] * xm[1 + s1] -
                     xp[1 + s1] * c[1 + s2] + c[1 + s1] * xm[1 + s2]) *
                    axx;
            advyy = epsilon *
                    (yp[1 + s2] * c[1 + s1] - c[1 + s2] * ym[1 + s1] -
                     yp[1 + s1] * c[1 + s2] + c[1 + s1] * ym[1 + s2]) *
                    ayy;
            advzz = epsilon *
                    (zp[1 + s2] * c[1 + s1] - c[1 + s2] * zm[1 + s1] -
                     zp[1 + s1] * c[1 + s2] + c[1 + s1] * zm[1 + s2]) *
                    azz;
            V[s] += dt * (advxx + advyy + advzz);
          }

          // drifts dt / 2.0 (fused, as in the host loop)
          rotate_by_charges(f, A, V, chi, hdt);

          for (int l = 0; l < NA; l++) phinew(k, j, i, OffsetA + l) = A[l];
          for (int l = 0; l < NV; l++) phinew(k, j, i, OffsetV + l) = V[l];
          for (int l = 0; l < Nphi; l++) phinew(k, j, i, l) = f[l];
        });
    Kokkos::fence();
    PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, solution, &phinew));
    PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &phi));
  }
  PetscCall(PetscLogFlops(Profiling::ideal_kick_flops * xm * ym * zm));
  PetscCall(PetscLogEventEnd(prof.ideal_kick, 0, 0, 0, 0));

  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// Position-Verlet update for the ideal (advective) sector. Linearized about
// phi_0 = f this is a wave equation with omega^2 = v^2 khat^2 + m^2, where
// v^2 = f^2 / chi and khat^2 <= 12 / a^2 for the 3D nearest-neighbor
// Laplacian. Verlet is stable for omega_max * dt < 2, i.e. (for small m)
//     v * dt < a / sqrt(3) ~ 0.577 a
// We conventionally take v^2 = f^2 / chi = 1 and a = 1, so dt < 0.577.
bool IdealPV2::step(const double &dt) { return step_no_reject(dt); }

bool IdealPV2::step_no_reject(const double &dt) {
  const auto &prof = Profiling::get();

#ifdef SFGPU_KOKKOS
  if (use_kokkos) {
    const auto &coeff = data.acoefficients;
    PetscCall(ideal_step_kokkos(
        da, model->solution, dt, coeff.chi,
        coeff.H * coeff.sigmabyf(data.atime.t()), pow(1. / data.hX(), 2),
        pow(1. / data.hY(), 2), pow(1. / data.hZ(), 2)));
    return true;
  }
#endif

  // drifts the solution by dt / 2.0
  G_node ***phinew;
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));
  PetscLogEventBegin(prof.ideal_rotate, 0, 0, 0, 0);
  rotatePhi(phinew, dt / 2.0);
  PetscLogEventEnd(prof.ideal_rotate, 0, 0, 0, 0);
  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));

  // Get a local vector with ghost cells
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));

  // Fill in the ghost celss with mpicalls
  PetscCall(ghost_exchange(da, model->solution, localU));

  // Get Access to arrays with drifted solution
  G_node ***phi;
  PetscCall(DMDAVecGetArrayRead(da, localU, &phi));
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));

  const auto &coeff = data.acoefficients;
  // Superfluid mode: the field couples as H * sigma/f
  const PetscReal H[4] = {coeff.H * coeff.sigmabyf(data.atime.t()), 0., 0.,
                          0.};
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);

  PetscScalar advxx, advyy, advzz;
  PetscInt s1, s2, epsilon;

  // Loop over central elements
  PetscLogEventBegin(prof.ideal_kick, 0, 0, 0, 0);
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // First evolve the momenta nab

        G_node &centralPhi = phi[k][j][i];
        G_node &phixplus = phi[k][j][i + 1];
        G_node &phixminus = phi[k][j][i - 1];
        G_node &phiyplus = phi[k][j + 1][i];
        G_node &phiyminus = phi[k][j - 1][i];
        G_node &phizplus = phi[k + 1][j][i];
        G_node &phizminus = phi[k - 1][j][i];

        for (PetscInt l = 0; l < ModelAData::NA; l++) {
          advxx = (-phixplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phixplus.f[l + 1] +
                   centralPhi.f[0] * phixminus.f[l + 1] -
                   phixminus.f[0] * centralPhi.f[l + 1]) *
                  axx;

          advyy = (-phiyplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phiyplus.f[l + 1] +
                   centralPhi.f[0] * phiyminus.f[l + 1] -
                   phiyminus.f[0] * centralPhi.f[l + 1]) *
                  ayy;

          advzz = (-phizplus.f[0] * centralPhi.f[l + 1] +
                   centralPhi.f[0] * phizplus.f[l + 1] +
                   centralPhi.f[0] * phizminus.f[l + 1] -
                   phizminus.f[0] * centralPhi.f[l + 1]) *
                  azz;

          phinew[k][j][i].A[l] +=
              dt * (advxx + advyy + advzz - H[0] * centralPhi.f[l + 1]);
        }

        for (PetscInt s = 0; s < ModelAData::NV; s++) {

          s1 = (s + 1) % 3;
          s2 = (s + 2) % 3;
          epsilon = ((PetscScalar)(s - s1) * (s1 - s2) * (s2 - s)) / 2.;
          // advection term with epsilon
          advxx = epsilon *
                  (phixplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phixminus.f[1 + s1] -
                   phixplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phixminus.f[1 + s2]) *
                  axx;

          advyy = epsilon *
                  (phiyplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phiyminus.f[1 + s1] -
                   phiyplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phiyminus.f[1 + s2]) *
                  ayy;

          advzz = epsilon *
                  (phizplus.f[1 + s2] * centralPhi.f[1 + s1] -
                   centralPhi.f[1 + s2] * phizminus.f[1 + s1] -
                   phizplus.f[1 + s1] * centralPhi.f[1 + s2] +
                   centralPhi.f[1 + s1] * phizminus.f[1 + s2]) *
                  azz;

          phinew[k][j][i].V[s] += dt * (advxx + advyy + advzz);
        }

        // drifts dt / 2.0. Safe inside the kick loop: the kick reads phi
        // only from the ghosted copy, and the drift of this site needs only
        // its own kicked charges.
        rotateSite(phinew[k][j][i], dt / 2.0);
      }
    }
  }

  PetscLogFlops(Profiling::ideal_kick_flops * xdimension * ydimension *
                zdimension);
  PetscLogEventEnd(prof.ideal_kick, 0, 0, 0, 0);

  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &phi));
  PetscCall(DMRestoreLocalVector(da, &localU));

  return true;
}

void IdealPV2::rotatePhi(G_node ***phinew, double dt) {
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        rotateSite(phinew[k][j][i], dt);
      }
    }
  }
}

// Rotates phi at one site by the charges (converted to chemical potentials)
void IdealPV2::rotateSite(G_node &node, double dt) const {
  rotate_by_charges(node.f, node.A, node.V, data.acoefficients.chi, dt);
}

void IdealPV2::finalize() { ; }

/////////////////////////////////////////////////////////////////////////

// Explicit forward-Euler step for the deterministic diffusion equations of
// Model G (one ghost exchange, one stencil pass). Each lattice mode is
// multiplied by 1 - dt * Gamma * khat^2 with khat^2 <= 12 (a = 1), so
//     dt * Gamma < 1/12   no mode changes sign (non-oscillatory)
//     dt * Gamma < 1/6    stability
// and likewise with D for the charges. We conventionally keep
// D = Gamma / 3, so the phi bound is the binding constraint.
ModelGExplicitDiffusionStep::ModelGExplicitDiffusionStep(ModelA &in)
    : model(&in) {
#ifdef SFGPU_KOKKOS
  use_kokkos = is_kokkos_vec(model->solution);
#endif
}

#ifdef SFGPU_KOKKOS
// Device version of the stencil loop in ModelGExplicitDiffusionStep::step:
// the same arithmetic in the same order, so with -fmad=false it matches the
// host loop bit-for-bit. The center values are read from localU, which the
// ghost exchange has just filled from the solution, so the solution itself is
// only written: one pass over the field less than phinew += ...
static PetscErrorCode diffusion_stencil_kokkos(
    DM da, Vec localU, Vec solution, const PetscReal (&Hin)[4], PetscReal axx,
    PetscReal ayy, PetscReal azz, PetscReal dtG, PetscReal dtD, PetscReal f,
    bool preserve_norm) {
  PetscFunctionBeginUser;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  ConstNodeView phi;
  NodeView phinew;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, localU, &phi));
  PetscCall(DMDAVecGetKokkosOffsetViewDOFWrite(da, solution, &phinew));

  const Kokkos::Array<PetscReal, 4> H = {Hin[0], Hin[1], Hin[2], Hin[3]};
  const PetscReal tiny = std::numeric_limits<double>::min();
  const PetscInt Nphi = ModelAData::Nphi;

  Kokkos::parallel_for(
      "DiffusionStencil", SitePolicy({zs, ys, xs}, {zs + zm, ys + ym, xs + xm}),
      KOKKOS_LAMBDA(PetscInt k, PetscInt j, PetscInt i) {
        // phi: nabla^2 phi, then the mass term (H[l] is H * sigma/f)
        PetscReal p[4];
        for (PetscInt l = 0; l < Nphi; l++) {
          p[l] = phi(k, j, i, l);
          p[l] += dtG * (axx * (phi(k, j, i + 1, l) + phi(k, j, i - 1, l)) +
                         ayy * (phi(k, j + 1, i, l) + phi(k, j - 1, i, l)) +
                         azz * (phi(k + 1, j, i, l) + phi(k - 1, j, i, l)) -
                         2. * (axx + ayy + azz) * phi(k, j, i, l));
          p[l] += dtG * H[l];
        }
        // Normalize phi to f (same as G_node::normalize_phi)
        if (preserve_norm) {
          PetscReal norm = 0.;
          for (PetscInt l = 0; l < Nphi; l++) {
            norm += p[l] * p[l];
          }
          if (norm > tiny) {
            norm = Kokkos::sqrt(norm);
            for (PetscInt l = 0; l < Nphi; l++) {
              p[l] *= f / norm;
            }
          } else {
            p[0] = f;
          }
        }
        for (PetscInt l = 0; l < Nphi; l++) {
          phinew(k, j, i, l) = p[l];
        }

        // Charges A (c = 4..6) then V (c = 7..9): nabla^2 with D
        for (PetscInt c = ModelAData::OffsetA; c < ModelAData::Ndof; c++) {
          phinew(k, j, i, c) =
              phi(k, j, i, c) +
              dtD * (axx * (phi(k, j, i + 1, c) + phi(k, j, i - 1, c)) +
                     ayy * (phi(k, j + 1, i, c) + phi(k, j - 1, i, c)) +
                     azz * (phi(k + 1, j, i, c) + phi(k - 1, j, i, c)) -
                     2. * (axx + ayy + azz) * phi(k, j, i, c));
        }
      });
  // So the DiffusionStencil event times the kernel, not just its launch
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewDOFWrite(da, solution, &phinew));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, localU, &phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Single-rank fast path of ghost_exchange (see Profiling.h). With one rank and
// periodic boundaries every local site, ghosts included, is a copy of the
// global site at its periodically wrapped position, so one kernel fills local
// with one read and one write of the field. PETSc's DMGlobalToLocal does the
// same copy as an index-list gather at a fraction of the memory bandwidth.
// Sets *done to PETSC_FALSE and does nothing when it does not apply (more
// than one rank, other boundaries, host vectors, -use_petsc_ghost_exchange).
PetscErrorCode ghost_fill_periodic_kokkos(DM da, Vec global, Vec local,
                                          PetscBool *done) {
  PetscFunctionBeginUser;
  *done = PETSC_FALSE;
  static PetscBool use_petsc = PETSC_FALSE, checked = PETSC_FALSE;
  if (!checked) {
    PetscCall(PetscOptionsHasName(NULL, NULL, "-use_petsc_ghost_exchange",
                                  &use_petsc));
    checked = PETSC_TRUE;
  }
  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
  DMBoundaryType bx, by, bz;
  PetscInt M, N, P;
  PetscCall(DMDAGetInfo(da, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL,
                        &bx, &by, &bz, NULL));
  if (use_petsc || size != 1 || bx != DM_BOUNDARY_PERIODIC ||
      by != DM_BOUNDARY_PERIODIC || bz != DM_BOUNDARY_PERIODIC ||
      !is_kokkos_vec(global) || !is_kokkos_vec(local)) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetGhostCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  ConstNodeView g;
  NodeView l;
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da, global, &g));
  PetscCall(DMDAVecGetKokkosOffsetViewDOFWrite(da, local, &l));
  const int ndof = static_cast<int>(l.extent(3));

  // One thread per value, the (i, c) index of a row fastest: neighbouring
  // threads then copy neighbouring doubles. (One thread per site of ndof
  // values made each warp store touch ndof times more memory sectors.)
  using RowPolicy = Kokkos::MDRangePolicy<
      Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
      Kokkos::IndexType<int>>;
  const int x0 = static_cast<int>(xs), y0 = static_cast<int>(ys),
            z0 = static_cast<int>(zs);
  const int nx = static_cast<int>(M), ny = static_cast<int>(N),
            nz = static_cast<int>(P);
  Kokkos::parallel_for(
      "GhostFill",
      RowPolicy({z0, y0, 0},
                {z0 + static_cast<int>(zm), y0 + static_cast<int>(ym),
                 static_cast<int>(xm) * ndof},
                {1, 1, 256}),
      KOKKOS_LAMBDA(int k, int j, int ic) {
        const int i = x0 + ic / ndof;
        const int c = ic - (i - x0) * ndof;
        // The ghost width is smaller than the box, so one wrap suffices
        const int sk = k < 0 ? k + nz : (k >= nz ? k - nz : k);
        const int sj = j < 0 ? j + ny : (j >= ny ? j - ny : j);
        const int si = i < 0 ? i + nx : (i >= nx ? i - nx : i);
        l(k, j, i, c) = g(sk, sj, si, c);
      });
  Kokkos::fence();

  PetscCall(DMDAVecRestoreKokkosOffsetViewDOFWrite(da, local, &l));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da, global, &g));
  *done = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

bool ModelGExplicitDiffusionStep::step_on(const double &dt, Vec &vec,
                                          bool preserve_norm) {

  auto &data = model->data;
  DM da = model->domain;
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(ghost_exchange(da, vec, localU));

  const auto &coeff = data.acoefficients;
  const PetscReal H[4] = {coeff.H * coeff.sigmabyf(data.atime.t()), 0., 0.,
                          0.};
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);
  PetscReal dtG = dt * coeff.Gamma();
  PetscReal dtD = dt * coeff.D();
  PetscReal f = sqrt(data.f2());

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

#ifdef SFGPU_KOKKOS
  if (use_kokkos) {
    PetscCall(diffusion_stencil_kokkos(da, localU, vec, H, axx, ayy, azz,
                                       dtG, dtD, f, preserve_norm));
  } else
#endif
  {
    G_node ***phi;
    PetscCall(DMDAVecGetArrayRead(da, localU, &phi));
    G_node ***phinew;
    PetscCall(DMDAVecGetArray(da, vec, &phinew));
    for (PetscInt k = zstart; k < zstart + zdimension; k++) {
      for (PetscInt j = ystart; j < ystart + ydimension; j++) {
        for (PetscInt i = xstart; i < xstart + xdimension; i++) {
          G_node &centralPhi = phi[k][j][i];
          G_node &phixplus = phi[k][j][i + 1];
          G_node &phixminus = phi[k][j][i - 1];
          G_node &phiyplus = phi[k][j + 1][i];
          G_node &phiyminus = phi[k][j - 1][i];
          G_node &phizplus = phi[k + 1][j][i];
          G_node &phizminus = phi[k - 1][j][i];
          for (PetscInt l = 0; l < ModelAData::Nphi; l++) {
            phinew[k][j][i].f[l] +=
                dtG * (axx * (phixplus.f[l] + phixminus.f[l]) +
                       ayy * (phiyplus.f[l] + phiyminus.f[l]) +
                       azz * (phizplus.f[l] + phizminus.f[l]) -
                       2. * (axx + ayy + azz) * centralPhi.f[l]);
            phinew[k][j][i].f[l] += dtG * H[l];
          }
          if (preserve_norm) {
            G_node::normalize_phi(phinew[k][j][i].f, f);
          }
          for (PetscInt l = 0; l < ModelAData::NA; l++) {
            phinew[k][j][i].A[l] +=
                dtD * (axx * (phixplus.A[l] + phixminus.A[l]) +
                       ayy * (phiyplus.A[l] + phiyminus.A[l]) +
                       azz * (phizplus.A[l] + phizminus.A[l]) -
                       2. * (axx + ayy + azz) * centralPhi.A[l]);
          }
          for (PetscInt l = 0; l < ModelAData::NV; l++) {
            phinew[k][j][i].V[l] +=
                dtD * (axx * (phixplus.V[l] + phixminus.V[l]) +
                       ayy * (phiyplus.V[l] + phiyminus.V[l]) +
                       azz * (phizplus.V[l] + phizminus.V[l]) -
                       2. * (axx + ayy + azz) * centralPhi.V[l]);
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da, vec, &phinew));
    PetscCall(DMDAVecRestoreArrayRead(da, localU, &phi));
  }
  PetscCall(DMRestoreLocalVector(da, &localU));
  return true;
}

bool ModelGExplicitDiffusionStep::step(const double &dt) {

  auto &data = model->data;

  // Get a local vector with ghost cells
  DM da = model->domain;
  Vec localU;
  PetscCall(DMGetLocalVector(da, &localU));

  // Fill in the ghost celss with mpicalls
  PetscCall(ghost_exchange(da, model->solution, localU));

  const auto &coeff = data.acoefficients;
  // Superfluid mode: the field couples as H * sigma/f
  const PetscReal H[4] = {coeff.H * coeff.sigmabyf(data.atime.t()), 0., 0.,
                          0.};
  const PetscReal axx = pow(1. / data.hX(), 2);
  const PetscReal ayy = pow(1. / data.hY(), 2);
  const PetscReal azz = pow(1. / data.hZ(), 2);

  PetscReal dtG = dt * coeff.Gamma();
  PetscReal dtD = dt * coeff.D();
  PetscReal f = sqrt(data.f2());

  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;
  DMDAGetCorners(da, &xstart, &ystart, &zstart, &xdimension, &ydimension,
                 &zdimension);

  const auto &prof = Profiling::get();
  PetscLogEventBegin(prof.diffusion_stencil, 0, 0, 0, 0);
#ifdef SFGPU_KOKKOS
  if (use_kokkos) {
    PetscCall(diffusion_stencil_kokkos(da, localU, model->solution, H, axx,
                                       ayy, azz, dtG, dtD, f, true));
  } else
#endif
  {
  // Get Access to arrays with drifted solution
  G_node ***phi;
  PetscCall(DMDAVecGetArrayRead(da, localU, &phi));
  G_node ***phinew;
  PetscCall(DMDAVecGetArray(da, model->solution, &phinew));

  // Loop over central elements
  for (PetscInt k = zstart; k < zstart + zdimension; k++) {
    for (PetscInt j = ystart; j < ystart + ydimension; j++) {
      for (PetscInt i = xstart; i < xstart + xdimension; i++) {
        // First evolve the momenta nab

        G_node &centralPhi = phi[k][j][i];
        G_node &phixplus = phi[k][j][i + 1];
        G_node &phixminus = phi[k][j][i - 1];
        G_node &phiyplus = phi[k][j + 1][i];
        G_node &phiyminus = phi[k][j - 1][i];
        G_node &phizplus = phi[k + 1][j][i];
        G_node &phizminus = phi[k - 1][j][i];

        for (PetscInt l = 0; l < ModelAData::Nphi; l++) {

          // Compute the nabla^2 phi
          phinew[k][j][i].f[l] +=
              dtG * (axx * (phixplus.f[l] + phixminus.f[l]) +
                     ayy * (phiyplus.f[l] + phiyminus.f[l]) +
                     azz * (phizplus.f[l] + phizminus.f[l]) -
                     2. * (axx + ayy + azz) * centralPhi.f[l]);

          // Add the mass term (H[l]  is  H * sigma/f)
          phinew[k][j][i].f[l] += dtG * H[l];
        }
        // Normalize the first four components of the solution to f
        G_node::normalize_phi(phinew[k][j][i].f, f);

        for (PetscInt l = 0; l < ModelAData::NA; l++) {
          // Compute the nabla^2 A
          phinew[k][j][i].A[l] +=
              dtD * (axx * (phixplus.A[l] + phixminus.A[l]) +
                     ayy * (phiyplus.A[l] + phiyminus.A[l]) +
                     azz * (phizplus.A[l] + phizminus.A[l]) -
                     2. * (axx + ayy + azz) * centralPhi.A[l]);
        }

        for (PetscInt l = 0; l < ModelAData::NV; l++) {
          phinew[k][j][i].V[l] +=
              dtD * (axx * (phixplus.V[l] + phixminus.V[l]) +
                     ayy * (phiyplus.V[l] + phiyminus.V[l]) +
                     azz * (phizplus.V[l] + phizminus.V[l]) -
                     2. * (axx + ayy + azz) * centralPhi.V[l]);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da, model->solution, &phinew));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &phi));
  }
  PetscLogFlops(Profiling::diffusion_stencil_flops * xdimension * ydimension *
                zdimension);
  PetscLogEventEnd(prof.diffusion_stencil, 0, 0, 0, 0);

  PetscCall(DMRestoreLocalVector(da, &localU));

  return true;
}
