#ifndef STEPPER_H
#define STEPPER_H

#include "ModelA.h"
#include "Profiling.h"

class Stepper {
public:
  virtual bool step(const double &dt) = 0;
  virtual void finalize() = 0;
  virtual ~Stepper() = default;
};

/////////////////////////////////////////////////////////////////////////
// Ideal Position Verlet Stepper. The step is always accepted; the energy is
// recorded by the Measurer, not here.
class IdealPV2 : public Stepper {
public:
  IdealPV2(ModelA &in);
  void finalize() override;
  bool step(const double &dt) override;

  ~IdealPV2() { ; }

private:
  ModelA *model;

  DM da;
  const ModelAData &data;

  bool step_no_reject(const double &dt);
  void rotatePhi(G_node ***phinew, double dt);
  void rotateSite(G_node &node, double dt) const;

  // Minor optimization
  PetscInt xstart, ystart, zstart, xdimension, ydimension, zdimension;

  // Run the step as Kokkos kernels (solution is a Kokkos vector)
  bool use_kokkos = false;
};

/////////////////////////////////////////////////////////////////////////
// Explicit second-order Runge-Kutta step for the deterministic diffusion
// equation of Model G.
class ModelGExplicitDiffusionStep : public Stepper {
public:
  ModelGExplicitDiffusionStep(ModelA &in);
  bool step(const double &dt) override;
  // Diffusion step on an external vector (does not touch model->solution).
  // With preserve_norm = false, phi is not renormalized to |phi| = f
  // afterwards (simple dissipation instead of dissipation on the sphere).
  bool step_on(const double &dt, Vec &vec, bool preserve_norm = true);
  void finalize() override { ; }
  ~ModelGExplicitDiffusionStep() { ; }

private:
  ModelA *model;
  // Run the stencil as a Kokkos kernel (solution is a Kokkos vector)
  bool use_kokkos = false;
};

/////////////////////////////////////////////////////////////////////////
// Solve the deterministic superfluid equations of motion using operator
// splitting. 'A' = ideal Verlet step, 'B' = explicit diffusion step.
class SuperSplitStep : public Stepper {
public:
  SuperSplitStep(ModelA &in, const std::string &step_sequence = "ABBBBBB")
      : model(&in), pv2(in), diffusion(in), steps(step_sequence),
        A_count(0), B_count(0) {
    for (char s : steps) {
      if (s == 'A') {
        A_count++;
      } else if (s == 'B') {
        B_count++;
      }
    }
  }
  bool step(const double &dt) override {
    const auto &prof = Profiling::get();

    bool ok = true;
    for (char s : steps) {
      if (s == 'A') {
        PetscLogStagePush(prof.ideal_stage);
        PetscLogEventBegin(prof.ideal_step, 0, 0, 0, 0);
        ok = ok && pv2.step(dt / A_count);
        PetscLogEventEnd(prof.ideal_step, 0, 0, 0, 0);
        PetscLogStagePop();
      } else if (s == 'B') {
        PetscLogStagePush(prof.diffusion_stage);
        PetscLogEventBegin(prof.diffusion_step, 0, 0, 0, 0);
        ok = ok && diffusion.step(dt / B_count);
        PetscLogEventEnd(prof.diffusion_step, 0, 0, 0, 0);
        PetscLogStagePop();
      }
    }
    return ok;
  }
  void finalize() override {
    pv2.finalize();
    diffusion.finalize();
  }

private:
  ModelA *model;
  IdealPV2 pv2;
  ModelGExplicitDiffusionStep diffusion;
  std::string steps;  // The steps to take, e.g. "ABBB"
  double A_count = 0; // Number of A steps in the string steps
  double B_count = 0; // Number of B steps in the string steps
};

#endif
