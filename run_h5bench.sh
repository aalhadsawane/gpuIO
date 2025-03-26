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
echo "DUMP_DIR: $DUMP_DIR"
echo "CSV_DIR: $CSV_DIR"
echo "IO_THREADS: ${IO_THREADS[*]}"
echo "BLOCK_SIZES: ${BLOCK_SIZES[*]}"
echo "DATASET_SIZES: ${DATASET_SIZES[*]}"
echo "MODES: ${MODES[*]}"
echo "COMPUTE_TIME: ${COMPUTE_TIME:-4} (default: 4)"
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

# Clear existing dump directory
echo "Clearing previous results from ${DUMP_DIR}"
rm -rf "${DUMP_DIR}"/*

# Create dump directory
mkdir -p "${DUMP_DIR}"

# Create incremental directory in CSV folder
mkdir -p "${CSV_DIR}"

# Find the next available RunN directory by finding the maximum existing run number
run_num=1
for dir in "${CSV_DIR}"/Run*; do
    if [ -d "$dir" ]; then
        current_num=$(basename "$dir" | sed 's/Run//')
        if [ "$current_num" -gt "$run_num" ]; then
            run_num=$current_num
        fi
    fi
done
run_num=$((run_num + 1))

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
                
                # Set number of timesteps
                timesteps=2
                # set compute time
                COMPUTE_TIME=4s
                # Calculate number of particles (elements)
                # H5bench stores 8 values per particle (7 floats + 1 int), each 4 bytes = 32 bytes per particle
                # To match the requested dataset size with multiple timesteps:
                # num_particles = bytes_per_thread / (32 * timesteps)
                num_particles=$(echo "$bytes_per_thread / (32 * $timesteps)" | bc)
                
                # Set dimensions ensuring NUM_PARTICLES = DIM_1 * DIM_2 * DIM_3
                # Using 1D array with remaining dimensions as 1
                DIM_1=$num_particles
                DIM_2=1
                DIM_3=1
                
                # Calculate expected data size per process and total
                # Each process will write num_particles * 32 bytes * timesteps
                per_process_bytes=$(echo "$num_particles * 32 * $timesteps" | bc)
                per_process_mb=$(echo "scale=2; $per_process_bytes / 1048576" | bc)
                total_mb=$(echo "scale=2; $per_process_mb * $io_threads" | bc)
                
                # Debug: Print the calculated values before creating config
                echo "Calculated dimensions for dataset_size=${dataset_size}, io_threads=${io_threads}:"
                echo "  num_particles = $num_particles"
                echo "  DIM_1 = $DIM_1"
                echo "  DIM_2 = $DIM_2"
                echo "  DIM_3 = $DIM_3"
                echo "  block_size = $block_size"
                echo "  timesteps = $timesteps"
                echo "  Expected data size per process: ~${per_process_bytes} bytes (~${per_process_mb} MB)"
                echo "  Expected total data size: ~${total_mb} MB (${io_threads} processes × ${timesteps} timesteps)"

                # For the simple approach with mpirun + h5bench_write
                output_file="${DUMP_DIR}/output_${mode}_${io_threads}_${block_size}_${dataset_size}.h5"
                log_file="${DUMP_DIR}/log_${mode}_${io_threads}_${block_size}_${dataset_size}.txt"
                run_dir="${DUMP_DIR}/runs_${mode}_${io_threads}_${block_size}_${dataset_size}"
                
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
TIMESTEPS=${timesteps}
NUM_DIMS=3
DIM_1=${DIM_1}
DIM_2=${DIM_2}
DIM_3=${DIM_3}
BLOCK_SIZE=${block_size}
COLLECTIVE_DATA=YES
COLLECTIVE_METADATA=YES
EMULATED_COMPUTE_TIME_PER_TIMESTEP=${COMPUTE_TIME:-4s}
EOF

                # Also create a JSON configuration file (for potential use later)
                json_config="${run_dir}/h5bench.json"
                cat > "$json_config" << EOF
{
  "IO_OPERATION": "WRITE",
  "MEM_PATTERN": "CONTIG",
  "FILE_PATTERN": "CONTIG",
  "NUM_PARTICLES": "${num_particles}",
  "TIMESTEPS": "${timesteps}",
  "NUM_DIMS": "3",
  "DIM_1": "${DIM_1}",
  "DIM_2": "${DIM_2}",
  "DIM_3": "${DIM_3}",
  "BLOCK_SIZE": "${block_size}",
  "COLLECTIVE_DATA": "YES",
  "COLLECTIVE_METADATA": "YES",
  "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "${COMPUTE_TIME:-4s}"
}
EOF

                # Ensure files have proper permissions
                chmod 644 "$text_config"
                chmod 644 "$json_config"
                
                # Check if h5bench_write exists and run the benchmark
                if [ ! -f "./h5bench_write" ]; then
                    echo "ERROR: h5bench_write executable not found in current directory!"
                    continue
                fi
                
                # Output for debug info
                echo "Configuration file:"
                cat "$text_config"
                echo ""
                
                # Run the benchmark using mpirun + h5bench_write with the text config file
                echo "Running benchmark: Mode=$mode, I/O Threads=$io_threads, Block Size=$block_size, Dataset Size=$dataset_size"
                echo "Benchmark log will be saved to: ${log_file}"
                echo "Configuration file: ${text_config}"
                echo "Emulated compute time setting: ${COMPUTE_TIME:-4s} seconds"
                echo "RUNNING: mpirun --use-hwthread-cpus -n ${io_threads} ./h5bench_write ${text_config} ${output_file}"
                mpirun --use-hwthread-cpus -n ${io_threads} ./h5bench_write "${text_config}" "${output_file}" > "${log_file}" 2>&1
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
                    # Clean up the output H5 file after successful completion
                    if [ -f "$output_file" ]; then
                        # Check if the file is still being written to
                        if lsof "$output_file" > /dev/null 2>&1; then
                            echo "Warning: File is still in use, waiting for it to be fully closed..."
                            sleep 2
                        fi
                        echo "Cleaning up output file: $output_file"
                        rm -f "$output_file"
                    else
                        echo "Warning: Output file not found for cleanup: $output_file"
                    fi
                fi
                
                # Create a directory for storing outputs
                results_dir="${run_dir}/output"
                mkdir -p "$results_dir"
                cp "$log_file" "${results_dir}/stdout"
                
                # Initialize metrics
                ranks="$io_threads"
                raw_write_rate="0 MB/s"
                observed_write_rate="0 MB/s"
                compute_time="${COMPUTE_TIME:-4}"  # Use the configured compute time directly
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
                    
                    # Check if the emulated compute time was properly recognized
                    if grep -q "Emulated compute time per timestep" "$log_file"; then
                        echo "✓ Emulated compute time setting was found in benchmark config"
                        grep -i "Emulated compute time per timestep" "$log_file"
                    else
                        echo "⚠️ Emulated compute time setting was NOT found in benchmark output"
                        echo "Checking benchmark configuration file format..."
                        if grep -q "EMULATED_COMPUTE_TIME_PER_TIMESTEP" "$text_config"; then
                            echo "✓ EMULATED_COMPUTE_TIME_PER_TIMESTEP=${COMPUTE_TIME:-4} is present in config file"
                        else
                            echo "❌ EMULATED_COMPUTE_TIME_PER_TIMESTEP is missing from config file"
                        fi
                    fi
                    
                    # Check if the total emulated compute time is reported
                    if grep -q "Total emulated compute time" "$log_file"; then
                        echo "✓ Total emulated compute time was reported in results"
                        grep -i "Total emulated compute time" "$log_file"
                        # Extract the reported compute time for comparison
                        reported_compute_time=$(grep -i "Total emulated compute time:" "$log_file" | awk '{print $5}' || echo "0")
                        if [ "$reported_compute_time" != "0" ] && [ "$reported_compute_time" != "0.000" ]; then
                            echo "✓ Non-zero compute time detected: $reported_compute_time seconds"
                        else
                            echo "⚠️ Compute time is reported as zero despite configuration"
                        fi
                    else
                        echo "❌ No total emulated compute time reported in results"
                    fi
                    
                    # Extract metrics if available
                    if grep -q "write rate" "$log_file"; then
                        echo "Performance metrics found in log file"
                        
                        # Try to extract key metrics
                        # compute_time is now set directly from config, not extracted
                        write_size=$(grep -i "Total write size:" "$log_file" | awk '{print $4 " " $5}' || echo "0 MB")
                        raw_write_time=$(grep -i "Raw write time:" "$log_file" | awk '{print $4}' || echo "0")
                        metadata_time=$(grep -i "Metadata time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fcreate_time=$(grep -i "H5Fcreate() time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fflush_time=$(grep -i "H5Fflush() time:" "$log_file" | awk '{print $3}' || echo "0")
                        h5fclose_time=$(grep -i "H5Fclose() time:" "$log_file" | awk '{print $3}' || echo "0")
                        completion_time=$(grep -i "Observed completion time:" "$log_file" | awk '{print $4}' || echo "0")
                        # Capture the full write rate string including units
                        raw_write_rate=$(grep -i "Raw write rate:" "$log_file" | sed 's/.*Raw write rate: *//' || echo "0 MB/s")
                        observed_write_rate=$(grep -i "Observed write rate:" "$log_file" | sed 's/.*Observed write rate: *//' || echo "0 MB/s")
                        ranks=$(grep -i "Total number of ranks" "$log_file" | awk '{print $5}' || echo "$io_threads")
                        
                        echo "Extracted metrics:"
                        echo "  Compute time (from config): $compute_time"
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
awk -F, 'NR>1 && $1=="GPU" {print $2 " threads, " $3 " block size, " $4 " dataset: " $14}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 GPU Observed Write Rates:"
awk -F, 'NR>1 && $1=="GPU" {print $2 " threads, " $3 " block size, " $4 " dataset: " $15}' "$output_csv" | sort -t: -k2 -nr | head -3

# For CPU mode
echo ""
echo "Top 3 CPU Raw Write Rates:"
awk -F, 'NR>1 && $1=="CPU" {print $2 " threads, " $3 " block size, " $4 " dataset: " $14}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 CPU Observed Write Rates:"
awk -F, 'NR>1 && $1=="CPU" {print $2 " threads, " $3 " block size, " $4 " dataset: " $15}' "$output_csv" | sort -t: -k2 -nr | head -3

# Summary by thread count (average across all configurations)
echo ""
echo "Average Performance by Thread Count:"
echo "Threads,Raw_Write_Rates,Observed_Write_Rates"
for threads in "${IO_THREADS[@]}"; do
    # Calculate number of samples for this thread count
    count=$(awk -F, -v t="$threads" 'NR>1 && $2==t {count++} END {print (count>0 ? count : 0)}' "$output_csv")
    
    # Skip if no samples
    if [ "$count" -eq 0 ]; then
        echo "$threads,No samples,No samples"
        continue
    fi
    
    # List all raw rates for this thread count (preserving units)
    raw_rates=$(awk -F, -v t="$threads" 'NR>1 && $2==t {print $14}' "$output_csv" | tr '\n' ',' | sed 's/,$//')
    observed_rates=$(awk -F, -v t="$threads" 'NR>1 && $2==t {print $15}' "$output_csv" | tr '\n' ',' | sed 's/,$//')
    
    echo "$threads,$raw_rates,$observed_rates"
done

# Summary by block size (average across all configurations)
echo ""
echo "Average Performance by Block Size:"
echo "Block_Size,Raw_Write_Rates,Observed_Write_Rates"
for bs in "${BLOCK_SIZES[@]}"; do
    # Calculate number of samples for this block size
    count=$(awk -F, -v b="$bs" 'NR>1 && $3==b {count++} END {print (count>0 ? count : 0)}' "$output_csv")
    
    # Skip if no samples
    if [ "$count" -eq 0 ]; then
        echo "$bs,No samples,No samples"
        continue
    fi
    
    # List all raw rates for this block size (preserving units)
    raw_rates=$(awk -F, -v b="$bs" 'NR>1 && $3==b {print $14}' "$output_csv" | tr '\n' ',' | sed 's/,$//')
    observed_rates=$(awk -F, -v b="$bs" 'NR>1 && $3==b {print $15}' "$output_csv" | tr '\n' ',' | sed 's/,$//')
    
    echo "$bs,$raw_rates,$observed_rates"
done

echo ""
echo "===== END OF SUMMARY ====="