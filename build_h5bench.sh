#!/bin/bash

# Build custom h5bench with CUDA support for GPU Direct Storage benchmarking
echo "Building custom h5bench with CUDA memory allocation support..."

# Set the correct HDF5_HOME to use custom-built HDF5 with MPI and vfd_gds support
export HDF5_HOME=/home/gpuio/gpuIO/hdf5_install
echo "Using custom-built HDF5 with vfd_gds at: $HDF5_HOME"

# Navigate to h5bench directory
cd /home/gpuio/gpuIO/benchmarks/h5bench

# Create CUDA-enabled build directory
mkdir -p build_cuda
cd build_cuda

# Configure with CUDA support and custom-built HDF5 (with MPI support)
echo "Configuring CMake with CUDA and custom-built HDF5 support..."
cmake -DCMAKE_C_FLAGS="-I$HDF5_HOME/include" \
      -DCMAKE_CXX_FLAGS="-I$HDF5_HOME/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$HDF5_HOME/lib" \
      -DHDF5_ROOT=$HDF5_HOME \
      -DHDF5_DIR=$HDF5_HOME \
      -DHDF5_INCLUDE_DIRS="$HDF5_HOME/include" \
      -DHDF5_LIBRARIES="$HDF5_HOME/lib/libhdf5.so" \
      ..

# Build the project
echo "Building h5bench executables..."
make

# Add environment variables to .bashrc for persistence
echo "export PATH=$HDF5_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$HDF5_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo ""
echo "✅ Custom h5bench with CUDA support built successfully!"
echo "Build directory: $(pwd)"
echo ""
echo "Available executables:"
ls -la h5bench_*
echo ""
echo "Key features enabled:"
echo "  - GPU memory allocation (cudaMalloc) for all dataset buffers"
echo "  - Support for both traditional and GPU Direct Storage paths"
echo "  - Automatic vfd_gds detection and configuration"
echo "  - Ready for SSD→GPU VRAM direct transfers"

