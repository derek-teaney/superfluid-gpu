#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "ModelA.h"
#include "Profiling.h"
#include "Stepper.h"
#include "gitversion.h"
#include "initialize.h"

// Measurer, where the Petsc are included
#include "measurer.h"
#include "measurer_output.h"

void initialize_event(const int &ievent, ModelA *const model,
                      nlohmann::json &inputs) {
  const auto &ahandler = model->data.ahandler;
  std::string initialization = inputs["initialization"];
  bool zero_charges = inputs.value("zero_charges", true);

  if (initialization == "default" or initialization == "randomspins") {
    // Random spins on the sphere with zero or gaussian charges
    model->initialize_random_spins();
    if (zero_charges) {
        model->zero_charges();
    }
    else {
        model->initialize_gaussian_charges();
    }
  } else if (initialization == "loadfile") {
    // Load a standardized initial condition from an HDF5 snapshot. The
    // "loadfile" input is the filename stem; the file itself is
    // <loadfile>_save.h5 (same convention as ModelA::write()).
    std::string fname = inputs["loadfile"].get<std::string>();
    model->read(fname);
  } else if (initialization == "gaussians") {
    model->initialize(initialize_gaussians, &inputs["gaussians"]);
    model->write(inputs["outputfiletag"].get<std::string>() + "_initial");
  } else if (initialization == "spinwaves") {
    model->initialize(initialize_wave_spins, &inputs["spinwaves"]);
    model->write(inputs["outputfiletag"].get<std::string>() + "_initial");
  } else {
    throw std::runtime_error(
        "Unknown initialization type: " + initialization +
        ". Please use one of the following: default, loadfile, "
        "randomspins-gaussiancharges, gaussians, spinwaves.");
  }
}

// One coarsening, started from the current solution at the entry t_s of
// time_start_coarsening: duplicate the solution, apply pure diffusion to the
// copy, and measure field spectra at log-spaced diffusion times
// (ModelAData::coarsening_times). The solution itself is not changed.
static void run_coarsening(double t_s, ModelA *const model, Measurer &measurer,
                           measurer_output_fasthdf5 *measurer_output,
                           int rank) {
  const auto &ahandler = model->data.ahandler;
  const auto &atime = model->data.atime;

  // Largest diffusion substep: max(Gamma, D) * hmax = 1/24. Each lattice
  // mode is multiplied by 1 - Gamma h khat^2 with khat^2 <= 12, so
  // Gamma h < 1/12 keeps every mode from changing sign; this is half
  // of that bound.
  const auto &coeff = model->data.acoefficients;
  const double hmax = 1. / (24. * std::max(coeff.Gamma(), coeff.D()));
  double kGamma = 0.;
  const std::vector<double> tau_meas =
      model->data.coarsening_times(t_s, &kGamma);
  PetscPrintf(PETSC_COMM_WORLD,
              "Coarsening: t_start = %g, starting at t = %g, k_Gamma = %g, "
              "%zu spectra at diffusion times %g ... %g, %s, %s\n",
              t_s, (double)atime.t(), kGamma, tau_meas.size(),
              tau_meas.front(), tau_meas.back(),
              ahandler.preserve_norm_coarsening
                  ? "on the sphere (|phi| = f)"
                  : "simple dissipation (|phi| free)",
              ahandler.analytical_coarsening
                  ? "exact in Fourier space"
                  : ("substeps <= " + std::to_string(hmax)).c_str());

  Vec sol_coarse;
  VecDuplicate(model->solution, &sol_coarse);
  VecCopy(model->solution, sol_coarse);
  ModelGExplicitDiffusionStep diffuser(*model);
  if (ahandler.analytical_coarsening) {
    measurer.setupAnalyticalCoarsening(&model->solution);
  }

  // Diffuse from one readout time to the next in equal substeps of at most
  // hmax, so every spectrum is taken exactly at its tau. Or, with
  // analytical_coarsening, diffuse exactly from tau = 0 to each tau.
  double tau = 0.;
  for (const double tau_next : tau_meas) {
    const double span = tau_next - tau;
    if (ahandler.analytical_coarsening) {
      measurer.analyticalCoarsenedField(tau_next, &sol_coarse);
    } else if (span > 0.) {
      const int nstep =
          std::max(1, static_cast<int>(std::ceil(span / hmax - 1e-9)));
      for (int istep = 0; istep < nstep; istep++) {
        diffuser.step_on(span / nstep, sol_coarse,
                         ahandler.preserve_norm_coarsening);
      }
    }
    tau = tau_next;
    measurer.measure_field_spectra(&sol_coarse);
    if (rank == 0) {
      measurer_output->save_coarsening();
      measurer_output->save_coarsening_time(tau, tau / atime.dt(), t_s);
    }
  }
  measurer.finalizeAnalyticalCoarsening();
  VecDestroy(&sol_coarse);
  PetscPrintf(PETSC_COMM_WORLD, "Coarsening: t_start = %g done\n", t_s);
}

void run_event(const int &ievent, ModelA *const model, Stepper *const step,
               nlohmann::json &inputs) {

  const auto &ahandler = model->data.ahandler;
  auto &atime = model->data.atime;

  // Set up logging for PETSc so we can find out how much time
  // each part takes
  int steps = 0;
  PetscLogEvent measurements, stepmonitor, saving;
  PetscLogEventRegister("Measurements", 0, &measurements);
  PetscLogEventRegister("Saving the fields", 0, &saving);
  PetscLogEventRegister("Steps", 0, &stepmonitor);

  // Set filename for the hdf5 ouput file
  std::string filename;
  if (ahandler.eventmode) {
    std::stringstream namestream;
    namestream << ahandler.outputfiletag << "_" << std::setw(4)
               << std::setfill('0') << ahandler.current_event << ".h5";
    filename = namestream.str();
  } else {
    filename = ahandler.outputfiletag + ".h5";
  }
  PetscFileMode file_access = FILE_MODE_WRITE;
  // Open the file and create the measurement object
  Measurer measurer(model);

  // Creat the measurer output object, which will write the measurements
  // to the h5 file.
  int rank = -1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::unique_ptr<measurer_output_fasthdf5> measurer_output;
  if (rank == 0) {
    measurer_output = std::make_unique<measurer_output_fasthdf5>(
        &measurer, filename, file_access);
  }

  // With -profile_refcopy, time one VecCopy of the solution per step. It moves
  // the same volume as the owned-block copy in DMGlobalToLocal, as a
  // reference for the GhostExchange time.
  const auto &prof = Profiling::get();
  PetscBool refcopy = PETSC_FALSE;
  PetscOptionsHasName(NULL, NULL, "-profile_refcopy", &refcopy);
  Vec refvec = NULL;
  if (refcopy) {
    VecDuplicate(model->solution, &refvec);
  }

  // Start the loop
  size_t next_coarsening = 0;
  const double tiny = 1.e-10;
  while (atime.t() < atime.tfinal() - tiny) {

    // measure the solution every saveFrequency
    if (steps % ahandler.saveFrequency == 0) {
      PetscLogStagePush(prof.measure_stage);
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure(&model->solution);
      if (rank == 0) {
        measurer_output->save();
      }
      PetscPrintf(PETSC_COMM_WORLD,
                  "Event/Timestep %d/%d: step size = %g, time = %g, final = "
                  "%g\n",
                  ahandler.current_event, steps, (double)atime.dt(),
                  (double)atime.t(), (double)atime.tfinal());
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
      PetscLogStagePop();
    }

    // on separate frequency: measure the topological charge Fourier readout
    if (ahandler.saveFrequencyTopcharge > 0 and
        steps % ahandler.saveFrequencyTopcharge == 0) {
      PetscLogStagePush(prof.measure_stage);
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure_topcharge(&model->solution);
      if (rank == 0) {
        measurer_output->save_topcharge();
      }
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
      PetscLogStagePop();
    }

    // on separate frequency: radially binned 3D power spectra of the fields
    if (ahandler.saveFrequencyFields > 0 and
        steps % ahandler.saveFrequencyFields == 0) {
      PetscLogStagePush(prof.measure_stage);
      PetscLogEventBegin(measurements, 0, 0, 0, 0);
      measurer.measure_field_spectra(&model->solution);
      if (rank == 0) {
        measurer_output->save_field_spectra();
      }
      PetscLogEventEnd(measurements, 0, 0, 0, 0);
      PetscLogStagePop();
    }

    // Coarsenings: one for each entry of time_start_coarsening (sorted)
    // that has been reached, each from the current solution
    while (next_coarsening < ahandler.time_start_coarsening.size() &&
           atime.t() >=
               ahandler.time_start_coarsening[next_coarsening] - tiny) {
      run_coarsening(ahandler.time_start_coarsening[next_coarsening], model,
                     measurer, measurer_output.get(), rank);
      next_coarsening++;
    }

    // Write the solution to tape if writeFrequency > 0. This is used for
    // plotting of the solution. It is normally not analyzed, or written.
    if (ahandler.writeFrequency > 0 and steps % ahandler.writeFrequency == 0) {
      PetscLogEventBegin(saving, 0, 0, 0, 0);
      std::ostringstream tString;
      // Set the precision to 2 digits after the decimal point filled with 0
      tString << std::fixed << std::setprecision(2) << "_t_" << atime.t();
      model->write(ahandler.outputfiletag + tString.str());
      PetscLogEventEnd(saving, 0, 0, 0, 0);
    }

    // Do the actual steps
    PetscLogEventBegin(stepmonitor, 0, 0, 0, 0);
    step->step(atime.dt());
    PetscLogEventEnd(stepmonitor, 0, 0, 0, 0);

    if (refcopy) {
      PetscLogEventBegin(prof.ref_copy, 0, 0, 0, 0);
      VecCopy(model->solution, refvec);
      PetscLogEventEnd(prof.ref_copy, 0, 0, 0, 0);
    }

    // Increment the clock
    steps++;
    atime += atime.dt();
  }
  VecDestroy(&refvec);
}

void Run(nlohmann::json &inputs) {
  // Digest the inputs, some fields may be modified on ouptut
  ModelAData inputdata(inputs);

  // allocate the grid and initialize
  ModelA model(inputdata);

  // Construct the stepper
  std::unique_ptr<Stepper> step;
  auto &etype = inputdata.ahandler.evolverType;
  if (etype == "SuperSplitStep") {
    std::string step_sequence =
        inputs["SuperSplitStep"].value<std::string>("step_sequence", "ABBBBBB");
    PetscPrintf(PETSC_COMM_WORLD,
                "Using the SuperSplitStep stepper step_sequence %s \n",
                step_sequence.c_str());
    step = std::make_unique<SuperSplitStep>(model, step_sequence);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unrecognized stepper type %s. Aborting...\n",
                etype.c_str());
    return;
  }

  auto &ahandler = model.data.ahandler;
  auto &atime = model.data.atime;
  int nevents = std::max(ahandler.nevents, 1);
  for (int i = 0; i < nevents; i++) {
    atime.reset();
    initialize_event(i, &model, inputs);
    run_event(i, &model, step.get(), inputs);
    ahandler.current_event++;
  }

  // Destroy everything
  step->finalize();
  model.finalize();
}

int main(int argc, char **argv) {

  // Initialization of PETSc universe
  PetscErrorCode ierr;
  std::string help =
      "Usage: \n\n\t ./SuperPions.exe -input input.json [options]\n\n";
  ierr = PetscInitialize(&argc, &argv, (char *)0, help.c_str());
  if (ierr) {
    return ierr;
  }

  // Open the input file and parse the inputs into the ModelAData
  char filename[PETSC_MAX_PATH_LEN] = "";
  ierr = PetscOptionsGetString(NULL, NULL, "-input", filename, sizeof(filename),
                               NULL);
  nlohmann::json inputs;
  std::ifstream ifs(filename);
  if (ifs) {
    ifs >> inputs;
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Unable to open input file %s. Aborting...\n",
                filename);
    return PetscFinalize();
  }
  PetscPrintf(PETSC_COMM_WORLD, "Current version: %s\n", gitversion);

  PetscPrintf(PETSC_COMM_WORLD, "Running SuperPions with input file %s\n",
              filename);

  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Input parameters:\n" << inputs.dump(2) << std::endl;
  }

  Run(inputs);

  return PetscFinalize();
}
