# Project Code Directory

This directory contains all the project files that are supposed to be run on the NVIDIA GPU enabled machine. 

## How to compile 
### Load nvcc compiler 
```bash
module load cuda/12.6.2
```
### Use cmake to update the Makefile 
From inside the build directory run 
```bash 
cmake ..
```
Then run the makefile 
```bash 
make clean && make
```
## How to run 
### Serial Version
```bash
./serial-version/build/serial ./data/correctness/mdp.json
```
```bash
./serial-version/build/serial ./data/performance/gw_64x64/mdp.json
```
and see the outputed data
```bash
cat ./serial-version/build/results/summary.txt 
```

### Parallel Version

**Note**: The CUDA version requires significant GPU memory for large problems. For a 64x64 gridworld (4096 states), approximately 1GB of GPU memory is needed. If you encounter "out of memory" errors, try smaller problems first.

```bash
# Small test case (recommended first)
./cuda-code/build/cuda ./data/correctness/mdp.json
```

```bash
# Large performance test (requires >=1GB GPU memory)
./cuda-code/build/cuda ./data/performance/gw_64x64/mdp.json
```

If you get GPU memory errors with large problems, use smaller test cases or run the serial version instead:
```bash
# Alternative for memory-constrained GPUs
./serial-version/build/serial ./data/performance/gw_64x64/mdp.json
```

and see the results 
```bash
cat ./cuda-code/build/results/summary.txt 
```

## Memory Requirements

The CUDA implementation stores full transition and reward matrices in GPU memory:
- **Memory needed**: `S² × A × 16 bytes` (where S=states, A=actions)  
- **32x32 gridworld**: ~256MB GPU memory
- **64x64 gridworld**: ~1GB GPU memory
- **128x128 gridworld**: ~16GB GPU memory

For problems that exceed GPU memory, use the CPU serial version which uses sparse matrix representations.

### Correctness Comparison Script
```bash
# Run both versions and compare results
./serial-version/build/serial ./data/correctness/mdp.json
./cuda-code/build/cuda ./data/correctness/mdp.json

# Compare the outputs
```bash
python3 compare_results.py results/gw_32x32-cuda.txt serial-version/build/results/gw_32x32-serial.txt
```
```bash
python3 compare_results.py results/gw_64x64-cuda.txt results/gw_64x64-serial.txt
```
```bash
python3 compare_results.py results/mdp-cuda.txt results/gw_64x64-serial.txt
```