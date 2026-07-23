# Triton-MPI-Optimization

UCSD CSE234 (Winter 2025) — Programming Assignment 2

Distributed training and GPU kernel optimization: implementing model/data-parallel communication with MPI, custom collective operations, and a Triton matmul kernel.

## What's in here

- **`mpi_wrapper/comm.py`** — A `Communicator` wrapper around `mpi4py` (Allreduce, Allgather, Reduce_scatter, Alltoall, Split) that also tracks bytes transferred, plus custom `myAllreduce` / `myAlltoall` implementations.
- **`model/func_impl.py`** — Helpers for combining model parallelism (MP) and data parallelism (DP) in a Transformer's fully-connected layers (`fc_q`, `fc_k`, `fc_v`, `fc_o`), including collecting activations across MP shards in the forward and backward passes.
- **`data/data_parallel_preprocess.py`** — Splits a dataset across data-parallel ranks.
- **`matmul_triton.ipynb`** — A matrix multiplication kernel written in [Triton](https://github.com/openai/triton).
- **`mpi-test.py`** — A small CLI for trying out the different MPI collective operations (`--test_case allreduce`, `allgather`, `reduce_scatter`, `split`, `alltoall`, `myallreduce`, `myalltoall`).
- **`discussion2-1.txt`** — Notes comparing the performance of the custom `myAllreduce` / `myAlltoall` implementations to MPI's built-in versions.
- **`tests/`** — Test suite for the assignment.
- **`PA2_Instructions.md`** — Original assignment instructions.

## Requirements

```bash
pip install -r requirements.txt
```

Uses `mpi4py`, `numpy`, and `pytest` (with `pytest-mpi`). You'll also need an MPI implementation (e.g. OpenMPI or MPICH) installed on your system.

## Usage

Open `matmul_triton.ipynb` in Jupyter to run/inspect the Triton matmul kernel.

Run an MPI collective-communication example across 4 processes:

```bash
mpirun -np 4 python mpi-test.py --test_case allreduce
```

Swap `allreduce` for any of: `allgather`, `reduce_scatter`, `split`, `alltoall`, `myallreduce`, `myalltoall`.

Run the tests:

```bash
pytest tests/
```

## Notes

Built as coursework for UCSD CSE234, focused on implementing distributed communication primitives and GPU kernels by hand rather than relying on existing framework internals.
