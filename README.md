# prime_hunter

Distributed cluster-based search for probable prime pairs using the formula:

```
N     = 10^a - k·10^b - j·10^c - 1
rev(N) = decimal digit reversal of N
```

Only outputs pairs where **both N and rev(N) are probable primes** (Miller-Rabin).

Search targets numbers with **50 000 – 100 000 digits** (`a ≥ 50000`).

## Architecture

| Phase | What | Where |
|-------|------|-------|
| 0 | Sieve table (ord_p(10) algebraic, O(1) lookup) | every worker at startup |
| 1 | Trial division — eliminates composite (k,j) pairs | GPU worker (CUDA) or CPU fallback |
| 2 | Miller-Rabin on survivors — tests N and rev(N) | CPU worker (GMP) |

MPI distributes work across machines; each node auto-detects GPU/CPU role.

## Dependencies

| Library | Version | Notes |
|---------|---------|-------|
| CUDA Toolkit | ≥ 11.0 | GPU trial division (Phase 1) |
| OpenMPI | ≥ 4.0 | Multi-machine distribution |
| GMP | ≥ 6.0 | Miller-Rabin (Phase 2) |
| CMake | ≥ 3.18 | Build system |

### Install on Ubuntu/Debian

```bash
sudo apt install libopenmpi-dev libgmp-dev cmake build-essential
# CUDA: follow https://developer.nvidia.com/cuda-downloads
```

> **CUDA 11.x + GCC 12 incompatibility**: if cmake fails with *"unsupported GNU version"*, install gcc-11:
> ```bash
> sudo apt install gcc-11 g++-11
> ```
> CMakeLists.txt detects gcc-11 automatically and uses it as the CUDA host compiler.

### Install on Fedora/RHEL

```bash
sudo dnf install openmpi-devel gmp-devel cmake gcc
# load MPI module if needed: module load mpi/openmpi-x86_64
```

## Build

```bash
git clone <repo-url>
cd prime_hunter
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

If your GPU architecture is not detected automatically:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86  # e.g. RTX 30xx
```

## Generate prime files (run once)

```bash
cd build
./gen_primes 10000 small_primes.bin        # sieve table primes  (~78 KB)
./gen_primes 100000000 primes_1e8.bin      # trial division primes (~5.8M primes, ~22 MB)
```

## Run

```bash
# Single machine, 4 workers (1 master + 3 workers; first worker uses GPU if available)
mpirun -np 4 ./prime_hunter small_primes.bin primes_1e8.bin 50000 51000

# With explicit c_min (default 100)
mpirun -np 4 ./prime_hunter small_primes.bin primes_1e8.bin 50000 51000 100

# Multi-machine (create a hostfile first)
mpirun -np 8 --hostfile hosts.txt ./prime_hunter small_primes.bin primes_1e8.bin 50000 51000
```

**Hostfile example (`hosts.txt`):**
```
node1 slots=4
node2 slots=4
```

> Workers on the same node as the master automatically adjust GPU index to avoid conflicts.

## Output files

| File | Contents |
|------|----------|
| `phase1_survivors.txt` | All (a,b,c,k,j) that passed trial division |
| `phase2_primes.txt` | Pairs where both N and rev(N) are probable primes |
| `monitor.txt` | Live status updated every second |

## Monitor

While running, `watch -n1 cat monitor.txt` shows:

- Queue depths (Phase 1 / Phase 2 pending)
- Per-worker status (busy/idle, current task, elapsed time)
- Last 20 Phase 1 results (a, b, c, survivors)
- Last 20 Phase 2 results (a, b, c, pairs tested, probable primes, both-prime count)
- All candidates found so far (a, b, c, k, j)

## GPU benchmark

```bash
cd build
./bench_gpu small_primes.bin primes_1e8.bin 200
```

Tries different CUDA block sizes and prints timing so you can pick the fastest for your GPU.

## Tests

```bash
cd build
ctest -V
```
