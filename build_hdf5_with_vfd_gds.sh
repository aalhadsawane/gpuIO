#!/bin/bash

# Build HDF5 with vfd_gds (GPU Direct Storage) support
# This script builds HDF5 with the Direct Virtual File Driver enabled

# Set install directory within workspace
export HDF5_HOME=/home/gpuio/gpuIO/hdf5_install
mkdir -p $HDF5_HOME

# Set build directory within workspace
BUILD_DIR=/home/gpuio/gpuIO/hdf5_build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Building HDF5 with vfd_gds support..."
echo "HDF5_HOME: $HDF5_HOME"
echo "BUILD_DIR: $BUILD_DIR"

# Clone HDF5 if not already present
if [ ! -d "hdf5" ]; then
    echo "Cloning HDF5 repository..."
    git clone https://github.com/HDFGroup/hdf5.git
fi

cd hdf5
git checkout hdf5-1_14_1-2

# Create build directory
mkdir -p build
cd build

echo "Configuring HDF5 with vfd_gds support..."

# Configure HDF5 with vfd_gds support
cmake -DCMAKE_INSTALL_PREFIX=$HDF5_HOME \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DHDF5_ENABLE_PARALLEL=ON \
      -DHDF5_ENABLE_THREADSAFE=ON \
      -DHDF5_ENABLE_DIRECT_VFD=ON \
      -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
      -DHDF5_ENABLE_SZIP_SUPPORT=ON \
      -DALLOW_UNSUPPORTED=ON \
      ..

echo "Building HDF5..."
make -j$(nproc)

echo "Installing HDF5..."
make install

# Add environment variables to .bashrc for persistence
echo "export PATH=$HDF5_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$HDF5_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export HDF5_HOME=$HDF5_HOME" >> ~/.bashrc

echo ""
echo "✅ HDF5 with vfd_gds support built successfully!"
echo "Installation directory: $HDF5_HOME"
echo ""
echo "Verifying installation..."
$HDF5_HOME/bin/h5cc -showconfig | grep -E "(Parallel|Direct|VFD)"

echo ""
echo "Environment variables added to ~/.bashrc:"
echo "  - PATH: $HDF5_HOME/bin"
echo "  - LD_LIBRARY_PATH: $HDF5_HOME/lib"
echo "  - HDF5_HOME: $HDF5_HOME"
echo ""
echo "To use the new HDF5 installation, run:"
echo "  source ~/.bashrc"
echo "  export HDF5_HOME=$HDF5_HOME"
