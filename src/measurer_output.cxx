#include "measurer_output.h"
#include <cstdio>

#ifndef MODELA_NO_HDF5
/////////////////////////////////////////////////////////////////////////

measurer_output_fasthdf5::measurer_output_fasthdf5(Measurer *in,
                                                   const std::string filename,
                                                   const PetscFileMode &mode)
    : measure(in) {

  if (mode == FILE_MODE_WRITE) {
    file_id =
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else if (mode == FILE_MODE_APPEND) {
    // Check if file exists
    PetscBool exists = PETSC_FALSE;
    PetscTestFile(filename.c_str(), '\0', &exists);
    if (exists) {
      // File exists and we are in restart mode,  so open it for readwrite
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
      // File either doesn't exist although we are in restart mode
      file_id =
          H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
  }

  std::array<size_t, 1> NN1{Measurer::NScalars};
  scalars = std::make_unique<ntuple<1>>(NN1, "phi", file_id);
  NN1 = {1};
  timeout = std::make_unique<ntuple<1>>(NN1, "timeout", file_id);
  energy = std::make_unique<ntuple<1>>(NN1, "energy", file_id);

  std::array<size_t, 2> NN2{Measurer::NObs,
                            static_cast<size_t>(measure->getN())};
  wallx = std::make_unique<ntuple<2>>(NN2, "wallx", file_id);
  wally = std::make_unique<ntuple<2>>(NN2, "wally", file_id);
  wallz = std::make_unique<ntuple<2>>(NN2, "wallz", file_id);

  std::array<size_t, 3> NN3 = {Measurer::NObs,
                               static_cast<size_t>(measure->getN()) / 2 + 1, 2};
  wallx_k = std::make_unique<ntuple<3>>(NN3, "wallx_k", file_id);
  wally_k = std::make_unique<ntuple<3>>(NN3, "wally_k", file_id);
  wallz_k = std::make_unique<ntuple<3>>(NN3, "wallz_k", file_id);

  NN3 = {Measurer::NObsRotated, static_cast<size_t>(measure->getN()) / 2 + 1,
         2};
  wallx_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wallx_k_rotated", file_id);
  wally_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wally_k_rotated", file_id);
  wallz_k_rotated =
      std::make_unique<ntuple<3>>(NN3, "wallz_k_rotated", file_id);

  NN2 = {Measurer::NObsPhase, static_cast<size_t>(measure->getN())};
  wallx_phase = std::make_unique<ntuple<2>>(NN2, "wallx_phase", file_id);
  wally_phase = std::make_unique<ntuple<2>>(NN2, "wally_phase", file_id);
  wallz_phase = std::make_unique<ntuple<2>>(NN2, "wallz_phase", file_id);

  NN3 = {Measurer::NObsPhase, static_cast<size_t>(measure->getN()) / 2 + 1, 2};
  wallx_phase_k = std::make_unique<ntuple<3>>(NN3, "wallx_phase_k", file_id);
  wally_phase_k = std::make_unique<ntuple<3>>(NN3, "wally_phase_k", file_id);
  wallz_phase_k = std::make_unique<ntuple<3>>(NN3, "wallz_phase_k", file_id);

  // Topological charge spherically-averaged Fourier readout.
  const size_t nbins = static_cast<size_t>(measure->getNTopchargeBins());
  NN1 = {nbins};
  topcharge_sk = std::make_unique<ntuple<1>>(NN1, "topcharge_Sk", file_id);
  topcharge_kbins = std::make_unique<ntuple<1>>(NN1, "topcharge_kbins", file_id);
  topcharge_nshell =
      std::make_unique<ntuple<1>>(NN1, "topcharge_Nshell", file_id);
  NN1 = {2};
  topcharge_zero =
      std::make_unique<ntuple<1>>(NN1, "topcharge_zeromode", file_id);

  // Radially binned 3D spectra of the transverse fields pi_a and A_a.
  const size_t nfb = static_cast<size_t>(measure->getNFieldBins());
  NN1 = {nfb};
  fields_Pphi = std::make_unique<ntuple<1>>(NN1, "fields_Pphi", file_id);
  fields_PAA = std::make_unique<ntuple<1>>(NN1, "fields_PAA", file_id);
  fields_ReC = std::make_unique<ntuple<1>>(NN1, "fields_ReC", file_id);
  const bool vector_spectra = measure->getModel()->data.ahandler.vector_spectra;
  if (vector_spectra) {
    fields_PVV = std::make_unique<ntuple<1>>(NN1, "fields_PVV", file_id);
  }
  fields_kbins = std::make_unique<ntuple<1>>(NN1, "fields_kbins", file_id);
  fields_nshell = std::make_unique<ntuple<1>>(NN1, "fields_Nshell", file_id);
  NN1 = {nfb + 1};
  fields_bin_edges =
      std::make_unique<ntuple<1>>(NN1, "fields_bin_edges", file_id);
  NN1 = {1};
  fields_time = std::make_unique<ntuple<1>>(NN1, "fields_time", file_id);

  // Coarsening output (same bins as field spectra)
  NN1 = {nfb};
  coarsen_Pphi = std::make_unique<ntuple<1>>(NN1, "coarsen_Pphi", file_id);
  coarsen_PAA = std::make_unique<ntuple<1>>(NN1, "coarsen_PAA", file_id);
  coarsen_ReC = std::make_unique<ntuple<1>>(NN1, "coarsen_ReC", file_id);
  if (vector_spectra) {
    coarsen_PVV = std::make_unique<ntuple<1>>(NN1, "coarsen_PVV", file_id);
  }
  coarsen_kbins = std::make_unique<ntuple<1>>(NN1, "coarsen_kbins", file_id);
  coarsen_nshell = std::make_unique<ntuple<1>>(NN1, "coarsen_Nshell", file_id);
  NN1 = {nfb + 1};
  coarsen_bin_edges =
      std::make_unique<ntuple<1>>(NN1, "coarsen_bin_edges", file_id);
  NN1 = {1};
  coarsen_step = std::make_unique<ntuple<1>>(NN1, "coarsen_step", file_id);
  coarsen_time = std::make_unique<ntuple<1>>(NN1, "coarsen_time", file_id);
  coarsen_tstart =
      std::make_unique<ntuple<1>>(NN1, "coarsen_tstart", file_id);
}

measurer_output_fasthdf5::~measurer_output_fasthdf5() { H5Fclose(file_id); }

void measurer_output_fasthdf5::save(const std::string &what) {

  timeout->row[0] = measure->getModel()->data.atime.t();
  timeout->fill();

  if (measure->getModel()->data.ahandler.compute_energy) {
    energy->row[0] = measure->energy;
    energy->fill();
  }

  scalars->row = measure->OAverage;
  scalars->fill();

  wallx->row = measure->wallX.v;
  wally->row = measure->wallY.v;
  wallz->row = measure->wallZ.v;

  // wallx->fill();
  // wally->fill();
  // wallz->fill();

  // hdf5 doesn't have complex data types uses arrays with extra dimension
  // [0],[1] sequentially in memory
  std::memcpy(wallx_k->row.data(), measure->wallX_k.v.data(),
              wallx_k->row.size() * sizeof(double));
  std::memcpy(wally_k->row.data(), measure->wallY_k.v.data(),
              wally_k->row.size() * sizeof(double));
  std::memcpy(wallz_k->row.data(), measure->wallZ_k.v.data(),
              wallz_k->row.size() * sizeof(double));

  wallx_k->fill();
  wally_k->fill();
  wallz_k->fill();

  std::memcpy(wallx_k_rotated->row.data(), measure->wallX_k_rotated.v.data(),
              wallx_k_rotated->row.size() * sizeof(double));
  std::memcpy(wally_k_rotated->row.data(), measure->wallY_k_rotated.v.data(),
              wally_k_rotated->row.size() * sizeof(double));
  std::memcpy(wallz_k_rotated->row.data(), measure->wallZ_k_rotated.v.data(),
              wallz_k_rotated->row.size() * sizeof(double));

  wallx_k_rotated->fill();
  wally_k_rotated->fill();
  wallz_k_rotated->fill();

  wallx_phase->row = measure->wallXPhase.v;
  wally_phase->row = measure->wallYPhase.v;
  wallz_phase->row = measure->wallZPhase.v;

  // wallx_phase->fill();
  // wally_phase->fill();
  // wallz_phase->fill();

  std::memcpy(wallx_phase_k->row.data(), measure->wallXPhase_k.v.data(),
              wallx_phase_k->row.size() * sizeof(double));
  std::memcpy(wally_phase_k->row.data(), measure->wallYPhase_k.v.data(),
              wally_phase_k->row.size() * sizeof(double));
  std::memcpy(wallz_phase_k->row.data(), measure->wallZPhase_k.v.data(),
              wallz_phase_k->row.size() * sizeof(double));

  wallx_phase_k->fill();
  wally_phase_k->fill();
  wallz_phase_k->fill();
}

void measurer_output_fasthdf5::save_topcharge(const std::string &what) {

  // Topological charge spherically-averaged Fourier readout.  The bin centers
  // and per-shell mode counts are time independent, so write them only once.
  if (!topcharge_static_written) {
    topcharge_kbins->row = measure->topcharge_kbins;
    topcharge_kbins->fill();
    topcharge_nshell->row = measure->topcharge_Nshell;
    topcharge_nshell->fill();
    topcharge_static_written = true;
  }
  topcharge_sk->row = measure->topcharge_Sk;
  topcharge_sk->fill();
  topcharge_zero->row[0] = measure->topcharge_zero.real();
  topcharge_zero->row[1] = measure->topcharge_zero.imag();
  topcharge_zero->fill();
}

void measurer_output_fasthdf5::save_field_spectra(const std::string &what) {

  // The bin edges and static info are time independent: write them once
  if (!fields_static_written) {
    fields_bin_edges->row = measure->field_bin_edges;
    fields_bin_edges->fill();
    fields_kbins->row = measure->field_kbins;
    fields_kbins->fill();
    fields_nshell->row = measure->field_Nshell;
    fields_nshell->fill();
    fields_static_written = true;
  }
  fields_time->row[0] = measure->getModel()->data.atime.t();
  fields_time->fill();
  fields_Pphi->row = measure->field_Pphi;
  fields_Pphi->fill();
  fields_PAA->row = measure->field_PAA;
  fields_PAA->fill();
  fields_ReC->row = measure->field_ReC;
  fields_ReC->fill();
  if (fields_PVV) {
    fields_PVV->row = measure->field_PVV;
    fields_PVV->fill();
  }
}

void measurer_output_fasthdf5::save_coarsening(const std::string &what) {
  if (!coarsen_static_written) {
    coarsen_bin_edges->row = measure->field_bin_edges;
    coarsen_bin_edges->fill();
    coarsen_kbins->row = measure->field_kbins;
    coarsen_kbins->fill();
    coarsen_nshell->row = measure->field_Nshell;
    coarsen_nshell->fill();
    coarsen_static_written = true;
  }
  coarsen_Pphi->row = measure->field_Pphi;
  coarsen_Pphi->fill();
  coarsen_PAA->row = measure->field_PAA;
  coarsen_PAA->fill();
  coarsen_ReC->row = measure->field_ReC;
  coarsen_ReC->fill();
  if (coarsen_PVV) {
    coarsen_PVV->row = measure->field_PVV;
    coarsen_PVV->fill();
  }
}

void measurer_output_fasthdf5::save_coarsening_time(double tau, double step,
                                                    double t_start) {
  coarsen_step->row[0] = step;
  coarsen_step->fill();
  coarsen_time->row[0] = tau;
  coarsen_time->fill();
  coarsen_tstart->row[0] = t_start;
  coarsen_tstart->fill();
}
#endif

/////////////////////////////////////////////////////////////////////////
//! Helper routine which opens the ASCII viewer with a
//! specific mode
PetscErrorCode PetscViewerASCIIOpenMode(MPI_Comm comm, const char *name,
                                        PetscFileMode mode,
                                        PetscViewer *viewer) {
  PetscViewerCreate(PETSC_COMM_SELF, viewer);
  PetscViewerSetType(*viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(*viewer, mode);
  PetscViewerFileSetName(*viewer, name);
  return (0);
}

measurer_output_txt::measurer_output_txt(Measurer *in,
                                         const std::string &filename)
    : measure(in) {

  //const auto &ahandler = measure->getModel()->data.ahandler;
  std::string name = filename + std::string("_averages.txt");
  PetscInt ierr = PetscViewerASCIIOpenMode(
      PETSC_COMM_SELF, name.c_str(), FILE_MODE_WRITE, &averages_asciiviewer);
  CHKERRV(ierr);
}

measurer_output_txt::~measurer_output_txt() {
  PetscViewerDestroy(&averages_asciiviewer);
}

void measurer_output_txt::save(const std::string &what) {

  if (what == "phi") {
    for (int iO = 0; iO < measure->NScalars; iO++) {
      PetscViewerASCIIPrintf(averages_asciiviewer, "%22.15e ",
                             measure->OAverage[iO]);
    }
    PetscViewerASCIIPrintf(averages_asciiviewer, "\n");
  }
}
