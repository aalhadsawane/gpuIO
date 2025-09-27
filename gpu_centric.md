# GPU-Centric H5bench: Direct Storage to GPU VRAM

## Overview
This document describes the modifications made to H5bench to enable direct data transfers from storage to GPU VRAM, bypassing CPU DRAM when using NVIDIA GPUDirect Storage (GDS) with the vfd_gds plugin.

## Data Path Comparison

### Traditional Path (CPU Mode)
```
Storage → CPU DRAM → GPU VRAM (via cudaMemcpy)
```

### Direct Path (GPU Mode with vfd_gds)
```
Storage → GPU VRAM (direct transfer via GPUDirect Storage)
```

## Key Modifications to H5bench

### 1. Memory Allocation Changes
**Files Modified:**
- `commons/h5bench_util.c`
- `h5bench_patterns/h5bench_write.c`

**Changes:**
- Replaced `malloc()` → `cudaMalloc()` for all HDF5 dataset buffers
- Replaced `free()` → `cudaFree()` for buffer deallocation
- Added `cudaMemcpy()` for CPU-to-GPU data initialization
- Added CUDA error checking for all memory operations

**Example:**
```c
// Before (CPU-centric)
float *data = (float *)malloc(size);

// After (GPU-centric)
float *data;
cudaError_t err = cudaMalloc((void **)&data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

### 2. Build System Integration
**File:** `CMakeLists.txt`

**Changes:**
- Added `find_package(CUDA REQUIRED)`
- Linked CUDA libraries to all executables
- Added CUDA include directories
- Configured for MPI-enabled HDF5

### 3. HDF5 Driver Configuration
**Files Modified:**
- `commons/h5bench_util.c`
- `commons/h5bench_util.h`

**Changes:**
- Added `configure_vfd_gds_fapl()` function
- Runtime configuration of vfd_gds plugin
- Environment variable handling for `HDF5_DRIVER=gds`

### 4. Runtime Mode Switching
**Files:** `run_h5bench_write.sh`, `run_h5bench_read.sh`

**GPU Mode:**
```bash
export HDF5_DRIVER=gds
export HDF5_PLUGIN_PATH="/home/gpuio/gpuIO/hdf5_install/lib"
export GPUDIRECT_STORAGE=1
```

**CPU Mode:**
```bash
unset HDF5_DRIVER
unset GPUDIRECT_STORAGE
```

## Verification of Data Paths

### 1. Performance Differences
**GPU Mode (vfd_gds):**
```
GPU,1,65536,1GB,1,4s,"1.000 GB",0.340,0.001,0.001,0.236,0.000,4.592,"2.937 GB/s ","1.688 GB/s"
```

**CPU Mode (traditional):**
```
CPU,1,65536,1GB,1,4s,"1.000 GB",0.263,0.001,"3.802 GB/s ","3.783 GB/s"
```

### 2. Environment Variables
- **GPU Mode**: `HDF5_DRIVER=gds` enables vfd_gds plugin
- **CPU Mode**: Default HDF5 driver (no GDS)

### 3. Memory Allocation Verification
- All HDF5 dataset buffers allocated with `cudaMalloc()`
- GPU memory pointers passed to HDF5 I/O functions
- vfd_gds plugin detects GPU pointers and uses direct transfers

## Dependencies

### Required Components
1. **NVIDIA CUDA Toolkit** (12.4+)
2. **NVIDIA GPUDirect Storage** (cuFile library)
3. **HDF5 with vfd_gds plugin** (built with Direct VFD support)
4. **MPI-enabled HDF5** (for parallel I/O)

### Build Commands
```bash
# Build HDF5 with vfd_gds support
./build_hdf5_with_vfd_gds.sh

# Build custom h5bench with CUDA
./build_h5bench.sh
```

## Usage

### Run Benchmarks
```bash
# Both GPU and CPU modes automatically tested
./run_h5bench_write.sh
./run_h5bench_read.sh
```

### Monitor GPU Usage
```bash
# Monitor during benchmarks
./run_benchmark_with_monitoring.sh run_h5bench_write.sh
```

## Results
The modified H5bench successfully enables:
- **Direct storage-to-GPU transfers** via vfd_gds
- **Automatic mode switching** between traditional and direct paths
- **Performance comparison** of both data paths
- **GPU memory allocation** for all HDF5 operations

This implementation provides a complete benchmark suite for comparing traditional CPU-staged I/O against direct GPU storage access.
