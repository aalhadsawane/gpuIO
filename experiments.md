# Results Run 1

## Benchmark Summary

### GPU SYNC Performance

#### Top 3 Raw Write Rates
- **1 thread, 1048576 block size, 1GB dataset**: 356.273 TB/s
- **1 thread, 262144 block size, 1GB dataset**: 355.470 TB/s
- **2 threads, 1048576 block size, 10GB dataset**: 350.719 TB/s

#### Top 3 Observed Write Rates
- **8 threads, 65536 block size, 10GB dataset**: 249.225 GB/s
- **1 thread, 1048576 block size, 10GB dataset**: 165.423 GB/s
- **8 threads, 65536 block size, 1GB dataset**: 165.116 GB/s

### GPU ASYNC Performance

#### Top 3 Raw Write Rates
- **4 threads, 262144 block size, 10GB dataset**: 347.622 TB/s
- **2 threads, 262144 block size, 1GB dataset**: 347.622 TB/s
- **4 threads, 1048576 block size, 1GB dataset**: 346.856 TB/s

#### Top 3 Observed Write Rates
- **4 threads, 262144 block size, 1GB dataset**: 251.507 GB/s
- **2 threads, 65536 block size, 1GB dataset**: 165.443 GB/s
- **4 threads, 262144 block size, 10GB dataset**: 165.433 GB/s

### CPU SYNC Performance

#### Top 3 Raw Write Rates
- **2 threads, 65536 block size, 1GB dataset**: 348.392 TB/s
- **1 thread, 262144 block size, 10GB dataset**: 347.622 TB/s
- **2 threads, 65536 block size, 10GB dataset**: 346.856 TB/s

#### Top 3 Observed Write Rates
- **4 threads, 1048576 block size, 1GB dataset**: 165.456 GB/s
- **2 threads, 1048576 block size, 1GB dataset**: 165.385 GB/s
- **4 threads, 65536 block size, 1GB dataset**: 165.261 GB/s

### CPU ASYNC Performance

#### Top 3 Raw Write Rates
- **8 threads, 65536 block size, 1GB dataset**: 349.939 TB/s
- **4 threads, 65536 block size, 1GB dataset**: 348.392 TB/s
- **2 threads, 65536 block size, 10GB dataset**: 348.392 TB/s

#### Top 3 Observed Write Rates
- **2 threads, 1048576 block size, 10GB dataset**: 165.309 GB/s
- **8 threads, 65536 block size, 1GB dataset**: 165.305 GB/s
- **4 threads, 262144 block size, 1GB dataset**: 165.267 GB/s

## SYNC vs ASYNC Comparison by Thread Count

| Threads | SYNC (GB/s) | ASYNC (GB/s) | Improvement |
|---------|-------------|--------------|-------------|
| 1       | 164.77      | 164.88       | 0.07%       |
| 2       | 164.89      | 164.88       | -0.01%      |
| 4       | 164.96      | 172.16       | 4.37%       |
| 8       | 171.88      | 164.95       | -4.03%      |

## Average Performance by Thread Count

### SYNC Mode

| Threads | Avg Raw Write Rate (GB/s) | Avg Observed Write Rate (GB/s) |
|---------|---------------------------|--------------------------------|
| 1       | 344.578                   | 164.769                        |
| 2       | 343.556                   | 164.892                        |
| 4       | 342.317                   | 164.955                        |
| 8       | 343.940                   | 171.876                        |

### ASYNC Mode

| Threads | Avg Raw Write Rate (GB/s) | Avg Observed Write Rate (GB/s) |
|---------|---------------------------|--------------------------------|
| 1       | 342.426                   | 164.884                        |
| 2       | 342.861                   | 164.880                        |
| 4       | 342.750                   | 172.160                        |
| 8       | 341.540                   | 164.949                        |

### Average Performance by Block Size

| Block Size | Avg Raw Write Rate (GB/s) | Avg Observed Write Rate (GB/s) |
|------------|---------------------------|--------------------------------|
| 65536      | 0                         | 0                              |
| 262144     | 0                         | 0                              |
| 1048576    | 0                         | 0                              |
