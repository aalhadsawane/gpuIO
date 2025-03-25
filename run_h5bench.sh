#!/bin/bash
# Script: run_h5bench.sh
# Purpose: Run h5bench experiments varying IO threads, block size, and dataset size.
#          Capture complete benchmark output (e.g., write time, metadata time, etc.)
#          for both GPU direct and CPU copy modes.
#          Results are stored in benchmark_results.csv.

# Check for required dependencies
command -v bc >/dev/null 2>&1 || { echo >&2 "Error: bc is required for calculations but not installed. Please install bc and try again."; exit 1; }

# Source the configuration file
config_file="$(dirname "$0")/benchmark_config.conf"
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file not found at $config_file"
    exit 1
fi

# Parse configuration file
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ $line =~ ^#.*$ ]] && continue
    [[ -z $line ]] && continue
    
    # Remove any whitespace from both key and value
    key=$(echo "${line%%=*}" | tr -d '[:space:]')
    value=$(echo "${line#*=}" | tr -d '[:space:]')
    
    # Skip if key is empty
    [[ -z $key ]] && continue
    
    # Convert comma-separated values to arrays
    if [[ $value == *","* ]]; then
        # Convert to array
        IFS=',' read -ra arr <<< "$value"
        # Create array variable
        declare -a "$key=(${arr[*]})"
    else
        # Regular variable
        declare "$key=$value"
    fi
done < "$config_file"

# Debug: Print loaded variables
echo "Loaded configuration:"
echo "HDF5_HOME: $HDF5_HOME"
echo "BUILD_DIR: $BUILD_DIR"
echo "RESULTS_DIR: $RESULTS_DIR"
echo "CSV_DIR: $CSV_DIR"
echo "IO_THREADS: ${IO_THREADS[*]}"
echo "BLOCK_SIZES: ${BLOCK_SIZES[*]}"
echo "DATASET_SIZES: ${DATASET_SIZES[*]}"
echo "MODES: ${MODES[*]}"
# Using only SYNC mode, removing IO_MODES entirely
# echo "IO_MODES: ${IO_MODES[*]}"

# Validate required variables
if [ -z "$CSV_DIR" ]; then
    echo "Error: CSV_DIR is not set in configuration file"
    exit 1
fi

# Ensure HDF5 libraries are in LD_LIBRARY_PATH during benchmarks
export HDF5_HOME="$HDF5_HOME"
export LD_LIBRARY_PATH=$HDF5_HOME/lib:$LD_LIBRARY_PATH

# Set MPI thread safety for I/O threads
export MPICH_MAX_THREAD_SAFETY="multiple"

# Move to the build directory
cd "$BUILD_DIR"

# Create Results directory with capital R (as specified)
mkdir -p "${RESULTS_DIR}"

# Create incremental directory in CSV folder
mkdir -p "${CSV_DIR}"

# Find the next available RunN directory
run_num=1
while [ -d "${CSV_DIR}/Run${run_num}" ]; do
    run_num=$((run_num + 1))
done

# Create the new RunN directory and set it as output directory
new_run_dir="$CSV_DIR/Run$run_num"
mkdir -p "${new_run_dir}"

# Create nodename.txt with the hostname
hostname > "${new_run_dir}/nodename.txt"
echo "Created nodename.txt with hostname in ${new_run_dir}"

# --- Output CSV File ---
output_csv="${new_run_dir}/raw_output.csv"
# Create the CSV file with header - removing io_mode from the header
echo "mode,io_threads,block_size,dataset_size,ranks,compute_time,write_size,raw_write_time,metadata_time,h5fcreate_time,h5fflush_time,h5fclose_time,completion_time,raw_write_rate,observed_write_rate" > "${output_csv}"

# Ensure the file exists and is writable
if [ ! -f "${output_csv}" ]; then
    echo "Error: Failed to create output CSV file at ${output_csv}"
    exit 1
fi

for mode in "${MODES[@]}"; do
    if [ "$mode" == "GPU" ]; then
        echo "Setting up for GPU direct transfers..."
        export GPUDIRECT_STORAGE=1
        export LEGATE_IO_USE_VFD_GDS=1
    else
        echo "Setting up for CPU copy (fallback mode)..."
        unset GPUDIRECT_STORAGE
        export LEGATE_IO_USE_VFD_GDS=0
    fi

    # Remove the IO_MODES loop completely and always run in SYNC mode
    echo "Testing with I/O mode: SYNC (fixed)"
    
    for io_threads in "${IO_THREADS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for dataset_size in "${DATASET_SIZES[@]}"; do
                # Extract numeric part of dataset size (remove "GB")
                dataset_size_num=${dataset_size%GB}
                
                # Convert to bytes (1GB = 1,073,741,824 bytes)
                dataset_bytes=$(echo "$dataset_size_num * 1073741824" | bc)
                
                # Calculate bytes per thread
                bytes_per_thread=$(echo "$dataset_bytes / $io_threads" | bc)
                
                # Calculate number of particles (elements)
                # Assuming sizeof(double) = 8 bytes
                num_particles=$(echo "$bytes_per_thread / 8" | bc)
                
                # Set dimensions ensuring NUM_PARTICLES = DIM_1 * DIM_2 * DIM_3
                # Using 1D array with remaining dimensions as 1
                DIM_1=$num_particles
                DIM_2=1
                DIM_3=1
                
                # Calculate expected data size per process and total
                # Each process will write num_particles * 8 bytes * 1 timestep
                per_process_bytes=$(echo "$num_particles * 8" | bc)
                per_process_mb=$(echo "scale=2; $per_process_bytes / 1048576" | bc)
                total_mb=$(echo "scale=2; $per_process_mb * $io_threads" | bc)
                
                # Debug: Print the calculated values before creating config
                echo "Calculated dimensions for dataset_size=${dataset_size}, io_threads=${io_threads}:"
                echo "  num_particles = $num_particles"
                echo "  DIM_1 = $DIM_1"
                echo "  DIM_2 = $DIM_2"
                echo "  DIM_3 = $DIM_3"
                echo "  block_size = $block_size"
                echo "  Expected data size per process: ~${per_process_bytes} bytes (~${per_process_mb} MB)"
                echo "  Expected total data size: ~${total_mb} MB (${io_threads} processes × 1 timestep)"

                # For the simple approach with mpirun + h5bench_write
                output_file="/home/gpuio/gpuIO/Results/output_${mode}_${io_threads}_${block_size}_${dataset_size}.h5"
                log_file="/home/gpuio/gpuIO/Results/log_${mode}_${io_threads}_${block_size}_${dataset_size}.txt"
                run_dir="/home/gpuio/gpuIO/Results/runs_${mode}_${io_threads}_${block_size}_${dataset_size}"
                
                # Create the run directory and ensure it exists
                mkdir -p "$run_dir"
                
                # Remove any existing output files to prevent issues
                rm -f "$output_file"
                
                # Create a text configuration file for h5bench_write - this is our primary approach now
                text_config="${run_dir}/h5bench.cfg"
                cat > "$text_config" << EOF
IO_OPERATION=WRITE
MEM_PATTERN=CONTIG
FILE_PATTERN=CONTIG
NUM_PARTICLES=${num_particles}
TIMESTEPS=1
NUM_DIMS=3
DIM_1=${DIM_1}
DIM_2=${DIM_2}
DIM_3=${DIM_3}
BLOCK_SIZE=${block_size}
COLLECTIVE_DATA=YES
COLLECTIVE_METADATA=YES
TIMESTEP_COMPUTE_TIME=4
EOF

                # Ensure file has proper permissions
                chmod 644 "$text_config"
                
                # Check if h5bench_write exists and run the benchmark
                if [ ! -f "./h5bench_write" ]; then
                    echo "ERROR: h5bench_write executable not found in current directory!"
                    continue
                fi
                
                # Output for debug info
                echo "Configuration file:"
                cat "$text_config"
                echo ""
                
                # Run the benchmark using mpirun + h5bench_write
                echo "Running benchmark: Mode=$mode, I/O Threads=$io_threads, Block Size=$block_size, Dataset Size=$dataset_size"
                echo "RUNNING: mpirun -n ${io_threads} ./h5bench_write ${text_config} ${output_file}"
                mpirun -n ${io_threads} ./h5bench_write "${text_config}" "${output_file}" > "${log_file}" 2>&1
                benchmark_status=$?
                
                # Check if the benchmark succeeded
                if [ $benchmark_status -ne 0 ]; then
                    echo "WARNING: Benchmark exited with non-zero status: $benchmark_status"
                    echo "Error details (first 20 lines):"
                    head -20 "$log_file"
                    echo "..."
                    echo "Last 10 lines of log:"
                    tail -10 "$log_file"
                else
                    echo "Benchmark completed successfully with exit code 0"
                fi
                
                # Create a directory for storing outputs
                results_dir="${run_dir}/output"
                mkdir -p "$results_dir"
                cp "$log_file" "${results_dir}/stdout"
                
                # Initialize metrics
                ranks="$io_threads"
                raw_write_rate="0 MB/s"
                observed_write_rate="0 MB/s"
                compute_time="0"
                write_size="0 MB"
                raw_write_time="0"
                metadata_time="0"
                h5fcreate_time="0"
                h5fflush_time="0"
                h5fclose_time="0"
                completion_time="0"
                
                # Extract metrics from log file
                if [ -f "$log_file" ]; then
                    echo "Extracting metrics from log file"
                    
                    # Extract metrics if available
                    if grep -q "write rate" "$log_file"; then
                        echo "Performance metrics found in log file"
                        
                        # Try to extract key metrics
                        compute_time=$(grep -i "Total emulated compute time:" "$log_file" | awk '{print $5}' || echo "0")
                        write_size=$(grep -i "Total write size:" "$log_file" | awk '{print $4 " " $5}' || echo "0 MB")
                        raw_write_time=$(grep -i "Raw write time:" "$log_file" | awk '{print $4}' || echo "0")
                        metadata_time=$(grep -i "Metadata time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fcreate_time=$(grep -i "H5Fcreate() time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fflush_time=$(grep -i "H5Fflush() time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fclose_time=$(grep -i "H5Fclose() time:" "$log_file" | awk '{print $3}' || echo "0")
                        completion_time=$(grep -i "Observed completion time:" "$log_file" | awk '{print $4}' || echo "0")
                        raw_write_rate=$(grep -i "Raw write rate" "$log_file" | awk '{print $4 " " $5}' || echo "0 MB/s")
                        observed_write_rate=$(grep -i "Observed write rate" "$log_file" | awk '{print $4 " " $5}' || echo "0 MB/s")
                        ranks=$(grep -i "Total number of ranks" "$log_file" | awk '{print $5}' || echo "$io_threads")
                        
                        echo "Extracted metrics:"
                        echo "  Compute time: $compute_time"
                        echo "  Write size: $write_size"
                        echo "  Raw write time: $raw_write_time"
                        echo "  Metadata time: $metadata_time"
                        echo "  H5Fcreate time: $h5fcreate_time"
                        echo "  H5Fflush time: $h5fflush_time"
                        echo "  H5Fclose time: $h5fclose_time"
                        echo "  Completion time: $completion_time"
                        echo "  Raw write rate: $raw_write_rate"
                        echo "  Observed write rate: $observed_write_rate"
                        echo "  Ranks: $ranks"
                    else
                        echo "No performance metrics found in log file"
                    fi
                else
                    echo "Log file not found: $log_file"
                fi
                
                # Append the results to the CSV file
                echo "$mode,$io_threads,$block_size,$dataset_size,$ranks,$compute_time,\"$write_size\",$raw_write_time,$metadata_time,$h5fcreate_time,$h5fflush_time,$h5fclose_time,$completion_time,\"$raw_write_rate\",\"$observed_write_rate\"" >> "$output_csv"

                # Print a summary
                echo "Results saved to CSV for Mode=$mode, I/O Threads=$io_threads, Block Size=$block_size, Dataset Size=$dataset_size"
                echo "Raw Write Rate: $raw_write_rate"
                echo "Observed Write Rate: $observed_write_rate"
                echo ""

            done
        done
    done
done

echo "Benchmarking complete. Results saved in $output_csv"

# Generate summary of the best performers
echo ""
echo "===== BENCHMARK SUMMARY ====="
echo ""

# For GPU mode
echo "Top 3 GPU Raw Write Rates:"
awk -F, 'NR>1 && $1=="GPU" {gsub(/"/, "", $14); split($14, rate, " "); print $2 " threads, " $3 " block size, " $4 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 GPU Observed Write Rates:"
awk -F, 'NR>1 && $1=="GPU" {gsub(/"/, "", $15); split($15, rate, " "); print $2 " threads, " $3 " block size, " $4 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# For CPU mode
echo ""
echo "Top 3 CPU Raw Write Rates:"
awk -F, 'NR>1 && $1=="CPU" {gsub(/"/, "", $14); split($14, rate, " "); print $2 " threads, " $3 " block size, " $4 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 CPU Observed Write Rates:"
awk -F, 'NR>1 && $1=="CPU" {gsub(/"/, "", $15); split($15, rate, " "); print $2 " threads, " $3 " block size, " $4 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# Summary by thread count (average across all configurations)
echo ""
echo "Average Performance by Thread Count:"
echo "Threads,Avg_Raw_Write_Rate,Avg_Observed_Write_Rate"
for threads in "${IO_THREADS[@]}"; do
    avg_raw=$(awk -F, -v t="$threads" 'NR>1 && $2==t {gsub(/"/, "", $14); split($14, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    avg_obs=$(awk -F, -v t="$threads" 'NR>1 && $2==t {gsub(/"/, "", $15); split($15, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    echo "$threads,$avg_raw,$avg_obs"
done

# Summary by block size (average across all configurations)
echo ""
echo "Average Performance by Block Size:"
echo "Block_Size,Avg_Raw_Write_Rate,Avg_Observed_Write_Rate"
for bs in "${BLOCK_SIZES[@]}"; do
    avg_raw=$(awk -F, -v b="$bs" 'NR>1 && $3==b {gsub(/"/, "", $14); split($14, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    avg_obs=$(awk -F, -v b="$bs" 'NR>1 && $3==b {gsub(/"/, "", $15); split($15, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    echo "$bs,$avg_raw,$avg_obs"
done

echo ""
echo "===== END OF SUMMARY ====="