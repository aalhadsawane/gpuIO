#!/bin/bash

# Clone and build h5bench
mkdir -p /home/gpuio/gpuIO/benchmarks/
cd /home/gpuio/gpuIO/benchmarks/
git clone --recurse-submodules https://github.com/hpc-io/h5bench
cd h5bench
mkdir -p build
cd build
cmake -DCMAKE_C_FLAGS="-I$HDF5_HOME/include" \
      -DCMAKE_CXX_FLAGS="-I$HDF5_HOME/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$HDF5_HOME/lib" \
      -DHDF5_ROOT=$HDF5_HOME ..
make

# Add environment variables to .bashrc for persistence
echo "export PATH=$HDF5_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$HDF5_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo "h5bench built successfully!"

