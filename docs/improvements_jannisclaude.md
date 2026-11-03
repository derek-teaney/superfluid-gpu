# Potential improvements, especially for runs on several GPUs

First collected 2026-10-20 (commit `8e8d5e0`), updated 2026-10-26 (commit
`0ea3bb7`) with what has been implemented since and with the measurements of
the 768³ test on 4 GPUs.

Measured costs at 512³ on one bwUniCluster H100 (one rank): 0.247 s per time
step (1 A + 6 B), 0.034 s per diffusion substep, 0.039 s per basic
measurement, 0.97 s per field spectrum (without `vector_spectra`).

## Done since 2026-10-20

| What | Commit | Effect |
|---|---|---|
| Crash of the field spectra on more than one rank | `4841738` | `computeFieldSpectra` called `fftw3d->size_out()` on every rank, but `fftw3d` exists on rank 0 only: every run with field spectra on ≥ 2 ranks or GPUs crashed at the first spectrum. Fixed. |
| Coarsening substep from Γ | `1642c06` | Largest substep h with max(Γ, D)·h = 1/24 instead of dt / (number of `B`): 4× fewer substeps at Γ = 0.125. |
| Exact coarsening (`analytical_coarsening`) | `1642c06` | For `preserve_norm = false`: FFT of all 10 components once at t_start, then each mode exp(−c λ(k) τ) with the lattice Laplacian, back transform per readout. τ_final = 1/(Γ k₁²). Cost independent of τ_final. Checked on CPU and GPU against the exact decay of the lowest shell to ≤ 10⁻¹⁵. |
| Several coarsenings per run | `5f528c2` | `time_start_coarsening` may be a list; `coarsen_tstart` labels the rows. `RunDatabase` selects one with `t_start=`. |
| Bin map uploaded once | `0ea3bb7` | The field-spectrum bin map and weights go to the GPU once instead of with a host loop over all modes for every component (the former `TODO` in `measurer.cxx`). The gain per spectrum was not measured separately. |
| Vector density spectra (`vector_spectra`) | `0ea3bb7` | Optional `fields_PVV` / `coarsen_PVV` (V_a in the local frame), about one more FFT per component. |

## Measured: 768³ on 4 GPUs (job 7890768, 2026-10-23)

One node `uc3n082`, 4 × H100 with **95.8 GB** each, `-use_gpu_aware_mpi 0`,
60 time steps, 12 basic measurements, 6 field spectra, numerical coarsening at
t = 10 (τ ≤ 50, 5 spectra). Total 157 s. Files:
`super-analysis/results/dcc/tests/NX768-4gpu/` (`-log_view` in the `.out`,
GPU memory every 5 s in `NX768-4gpu_gpumem.csv`).

| Per call | Total | of which ghost exchange |
|---|---|---|
| Time step (A + 6 B) | **0.449 s** | about 0.36 s (**80%**) |
| Ideal step A | 96 ms | 72 ms |
| Diffusion substep B | 63 ms | 48 ms (stencil kernel 16 ms) |
| Coarsening substep | about 70 ms | 54 ms |
| Field spectrum (gather of 8 × 3.6 GB to rank 0, FFTs on GPU 0) | **about 9 s** | — |

| Memory (peak, including the coarsening copy) | |
|---|---|
| GPU 0 (rank 0, spectrum buffers) | 47.9 GB (50%) |
| GPUs 1–3 | 29.8 GB (31%) |
| Host | 211 GB of 755 GB |

Conclusions:

- Without the exchange a step would take about 0.12 s, i.e. per site faster
  than 512³ on one GPU. The ghost exchange through host memory is the
  bottleneck on several GPUs.
- Analytical coarsening adds about 40 GB on GPU 0 (stored spectra of the 10
  components plus a work spectrum), i.e. about 88 GB of 95.8 GB: it fits on
  4 GPUs at 768³, with about 8% margin (estimate, not run).
- Each spectrum, and each analytical readout (spectrum plus 10 back
  transforms and redistribution of 10 components), costs ≳ 9 s at 768³: with
  100 readouts the analytical coarsening takes an estimated 30–50 min, so use
  fewer readouts there.

## 1. The biggest lever: one event per GPU

The events of a run are independent but run one after another on one GPU,
while a node has 4 GPUs.

- For 512³: run 4 jobs with 1 GPU each (or one 4-GPU job with 4 separate
  `srun` steps, each with its own seed). That gives about 4× more events per
  hour with no code change, because the GPUs never communicate.
- Splitting a single run across several GPUs is only worth it when the
  lattice does not fit on one GPU (768³ and above, with the coarsening copy)
  or when one run has to finish faster. On 4 GPUs a 768³ step currently
  takes 0.449 s, of which 80% is communication.

## 2. Specific to running on several GPUs

| What | Problem | Improvement | Effort |
|---|---|---|---|
| **GPU-aware MPI** (most important) | Measured at 768³: the ghost exchange through host memory is 80% of the step time (7 exchanges per step). The site modules `mpi/openmpi/5.0*` are not CUDA-aware (see `SuperfluidGPU_bwuni3_summary.pdf`, section 2.6), so `-use_gpu_aware_mpi 1` alone will not work with the current PETSc. | Build PETSc against a CUDA-aware MPI, e.g. the HPC-X Open MPI of the NVIDIA HPC SDK (`/software/bwhpc/common/toolkit/nvidia_hpc_sdk/25.1/.../comm_libs/12.6/hpcx`), and run with `-use_gpu_aware_mpi 1`. On Perlmutter with GPU-aware MPI a ghost exchange took about 2 ms (`docs/PLAN.md`) against 48–72 ms here. Possible gain: step 0.45 s → about 0.15 s at 768³ on 4 GPUs. | medium (PETSc rebuild, then a short test) |
| Spectra on one GPU | Each field spectrum gathers 8 full components to rank 0 (3.6 GB each at 768³; 12 with `vector_spectra`), and only GPU 0 does the FFTs. Measured about 9 s per spectrum at 768³. | (a) Short term: fewer spectra and coarsening readouts at large N. (b) Long term: a distributed FFT (cuFFTMp or heFFTe), so each GPU transforms its own part and nothing is gathered. Needed at 1024³, where GPU 0 needs about 41 GB just for the spectrum buffers. | (a) none, (b) large |
| Analytical coarsening memory | The stored spectra of the 10 components live on GPU 0: about 11 GB at 512³, 36 GB at 768³, 86 GB at 1024³. At 768³ on 4 GPUs GPU 0 is then about 92% full; at 1024³ it does not fit on any number of GPUs. | Keep the stored spectra in host memory of rank 0 and copy one component per back transform to the GPU (1–3 GB per component at 768³). | small |
| Memory: copy for coarsening | The coarsening keeps a third full copy of the fields (`sol_coarse`). At 1024³ that is 86 GB more in total. | Keep the original on the host instead of the GPU, diffuse in place and copy back. Or save a snapshot at t_start and run the coarsening as a separate job (`"initialization": "loadfile"` exists). | medium |
| Domain split | `DMDACreate3d` uses `PETSC_DECIDE`, so PETSc chooses how the lattice is divided among the ranks. | Try `-da_processors_x 1` (split only in y and z): the halos are then contiguous in memory and cheaper to pack. Measure. | trivial to test |
| Load on rank 0 | Rank 0 also holds the gathered field component, the FFT buffers and writes the HDF5 output: 47.9 GB on GPU 0 against 29.8 GB on the others at 768³. | Goes away with a distributed FFT; until then, include it in memory planning. | — |

## 3. Improvements on any number of GPUs

| What | Problem | Improvement | Estimated gain |
|---|---|---|---|
| Diffusion kernel | `DiffusionStencil` reaches about 320 GB/s against about 1.5 TB/s peak (A100 measurement in `docs/PLAN.md`); at 768³ on 4 GPUs the kernel itself is 16 ms of the 63 ms substep. | Ideas from `docs/PLAN.md`: store the fields per component (SoA) instead of per site (AoS, `G_node`), tune the tile sizes. | matters on one GPU; on several GPUs only after the exchange is fixed |
| Fuse B substeps | Each of the 6 B substeps does 1 ghost exchange and 1 full pass over memory. | With wider ghost layers, do several substeps per exchange: fewer exchanges (most valuable on several GPUs) and fewer passes over memory. Also listed in `docs/PLAN.md`. | medium to large; on several GPUs the exchange count goes down directly |
| Time-step stability check | Nothing checks the Verlet bound v·Δt < 1/√3 with v = √(f²/χ); with large f² and too large Δt a run blows up without a message. | Check at startup and stop with a message. | avoids wasted runs |
| Lattice momentum for the coarsening | τ_first = 1/(Γ (1.2 k_Γ)²) uses the continuum k²; at early t_start (k_Γ near the lattice cutoff) the lattice decay is 12–32% slower. | Write the lattice momentum k̂² = ⟨Σ_d 4 sin²(k_d/2)⟩ per bin (e.g. `coarsen_khat`), or choose τ_first from the lattice λ. | analysis accuracy at early times |

## Suggested order

1. Throughput at 512³: one event per GPU (only the runner script changes).
2. For runs on several GPUs: PETSc with a CUDA-aware MPI (HPC-X) and a short
   test with `-use_gpu_aware_mpi 1`; also `-da_processors_x 1`.
3. Small code changes: analytical-coarsening spectra in host memory,
   time-step stability check.
4. At 768³: fewer spectra and coarsening readouts; a short analytical
   coarsening test to confirm about 88 GB on GPU 0.
5. Larger projects: fused B substeps, kernel optimization, memory saving in
   the coarsening, then a distributed FFT (needed for 1024³).
