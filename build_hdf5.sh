#! /bin/bash

# set install directory
export HDF5_HOME=/home/gpuio/hdf5_install
mkdir -p $HDF5_HOME
mkdir -p /home/gpuio/hdf5_build
cd /home/gpuio/hdf5_build
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5-1_14_1-2
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HDF5_HOME \
      -DHDF5_ENABLE_PARALLEL=ON \
      -DHDF5_ENABLE_THREADSAFE=ON \
      -DALLOW_UNSUPPORTED=ON \
      -DCMAKE_C_COMPILER=mpicc ..
make -j && make install

# add environment variables to .bashrc for persistence
echo "export PATH=$HDF5_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$HDF5_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo "hdf5 built successfully!"