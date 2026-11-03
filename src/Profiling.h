#ifndef PROFILING_H
#define PROFILING_H

#include <petscdmda.h>

// PETSc log stages and events for -log_view. Registered once, on first use.
//
// Stages split the -log_view table by substep type, so PETSc's own events
// (VecScatterBegin/End, SFPack/Unpack) are attributed to A, B, or the
// measurements separately. Events inside a stage split exchange from compute.
struct Profiling {
  PetscLogStage ideal_stage, diffusion_stage, measure_stage;

  // IdealKick includes the second half-drift, fused into the kick loop;
  // IdealRotate is the first half-drift only.
  PetscLogEvent ideal_step, diffusion_step;
  PetscLogEvent ghost_exchange, ideal_kick, ideal_rotate, diffusion_stencil;
  PetscLogEvent ref_copy;

  // Approximate flops per site, counted by hand from the loop bodies in
  // Stepper.cxx. Used only for the Mflop/s column of -log_view.
  static constexpr PetscLogDouble ideal_kick_flops = 183;
  static constexpr PetscLogDouble diffusion_stencil_flops = 141;

  static const Profiling &get() {
    static const Profiling p;
    return p;
  }

private:
  Profiling() {
    PetscLogStageRegister("IdealA", &ideal_stage);
    PetscLogStageRegister("DiffusionB", &diffusion_stage);
    PetscLogStageRegister("Measure", &measure_stage);

    PetscLogEventRegister("IdealStep", 0, &ideal_step);
    PetscLogEventRegister("ExplicitDiffusion", 0, &diffusion_step);
    PetscLogEventRegister("GhostExchange", 0, &ghost_exchange);
    PetscLogEventRegister("IdealKick", 0, &ideal_kick);
    PetscLogEventRegister("IdealRotate", 0, &ideal_rotate);
    PetscLogEventRegister("DiffusionStencil", 0, &diffusion_stencil);
    PetscLogEventRegister("RefVecCopy", 0, &ref_copy);
  }
};

#ifdef SFGPU_KOKKOS
// One kernel instead of DMGlobalToLocal for a single rank with periodic
// boundaries and Kokkos vectors (Stepper.cxx); *done says whether it applied
PetscErrorCode ghost_fill_periodic_kokkos(DM da, Vec global, Vec local,
                                          PetscBool *done);
#endif

// Fills the ghost cells of local from global, logged as GhostExchange.
inline PetscErrorCode ghost_exchange(DM da, Vec global, Vec local) {
  const auto &prof = Profiling::get();
  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(prof.ghost_exchange, 0, 0, 0, 0));
  PetscBool done = PETSC_FALSE;
#ifdef SFGPU_KOKKOS
  PetscCall(ghost_fill_periodic_kokkos(da, global, local, &done));
#endif
  if (!done) {
    PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));
  }
  PetscCall(PetscLogEventEnd(prof.ghost_exchange, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
