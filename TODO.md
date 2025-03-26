# TODO


# CRITICAL add cleanup in Results. DONE

## 0. Remove IO_mode from make decision tree. DONE

## Add chunking and chunk sizes with compression for write benchmarks.

## 1. Add support for MB as dataset size.

## Pick multiple single threaded processes vs multiple single threaded processes.

Currently we use multiple single threaded processes.

Current exp:
Will be a combination of run 4,5,6


##### Run 4:
```conf
IO_THREADS=1,2,4
# Block sizes in bytes (16KB, 32KB, 64KB, 128KB)
BLOCK_SIZES=65536,131072,262144,524288,1048576,2097152,4194304
# Dataset sizes in gigabytes
DATASET_SIZES=1GB,5GB,10GB,20GB
MODES=GPU
```

##### Run 5:
```conf
IO_THREADS=8,16,32
# Block sizes in bytes (16KB, 32KB, 64KB, 128KB)
BLOCK_SIZES=65536,131072,262144,524288,1048576,2097152,4194304
# Dataset sizes in gigabytes
DATASET_SIZES=1GB,5GB,10GB,20GB
MODES=GPU
```

##### Run 6:
```conf
IO_THREADS=1,2,4,8,16,32
# Block sizes in bytes (16KB, 32KB, 64KB, 128KB)
BLOCK_SIZES=65536,131072,262144,524288,1048576,2097152,4194304
# Dataset sizes in gigabytes
DATASET_SIZES=1GB,5GB,10GB,20GB
MODES=CPU
```