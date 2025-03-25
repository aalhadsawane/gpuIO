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
echo "IO_MODES: ${IO_MODES[*]}"

# Validate required variables
if [ -z "$CSV_DIR" ]; then
    echo "Error: CSV_DIR is not set in configuration file"
    exit 1
fi

# Ensure HDF5 libraries are in LD_LIBRARY_PATH during benchmarks
export HDF5_HOME="$HDF5_HOME"
export LD_LIBRARY_PATH=$HDF5_HOME/lib:$LD_LIBRARY_PATH

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
# Create the CSV file with header
echo "mode,io_mode,io_threads,block_size,dataset_size,ranks,compute_time,write_size,raw_write_time,metadata_time,h5fcreate_time,h5fflush_time,h5fclose_time,completion_time,raw_write_rate,observed_write_rate" > "${output_csv}"

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

    for io_mode in "${IO_MODES[@]}"; do
        echo "Testing with I/O mode: $io_mode"
        
        for io_threads in "${IO_THREADS[@]}"; do
            for block_size in "${BLOCK_SIZES[@]}"; do
                for dataset_size in "${DATASET_SIZES[@]}"; do

                    # Create a JSON config file for h5bench
                    config_file="/home/gpuio/gpuIO/Results/config_${mode}_${io_mode}_${io_threads}_${block_size}_${dataset_size}.json"
                    output_file="/home/gpuio/gpuIO/Results/output_${mode}_${io_mode}_${io_threads}_${block_size}_${dataset_size}.h5"
                    log_file="/home/gpuio/gpuIO/Results/log_${mode}_${io_mode}_${io_threads}_${block_size}_${dataset_size}.txt"
                    run_dir="/home/gpuio/gpuIO/Results/runs_${mode}_${io_mode}_${io_threads}_${block_size}_${dataset_size}"
                    
                    # Create the run directory and ensure it exists
                    mkdir -p "$run_dir"
                    
                    # Remove any existing output files to prevent issues
                    rm -f "$output_file"
                    
                    # Create the JSON configuration file
                    cat > "$config_file" << EOF
{
    "mpi": {
        "command": "mpirun",
        "ranks": "1",
        "configuration": "-n 1"
    },
    "vol": {
    },
    "file-system": {
    },
    "directory": "${run_dir}",
    "benchmarks": [
        {
            "benchmark": "write",
            "file": "${output_file}",
            "configuration": {
                "MEM_PATTERN": "CONTIG",
                "FILE_PATTERN": "CONTIG",
                "TIMESTEPS": "10",
                "DELAYED_CLOSE_TIMESTEPS": "2",
                "COLLECTIVE_DATA": "YES",
                "COLLECTIVE_METADATA": "YES",
                "EMULATED_COMPUTE_TIME_PER_TIMESTEP": "2 s", 
                "NUM_DIMS": "3",
                "DIM_1": "1028",
                "DIM_2": "1028",
                "DIM_3": "512",
                "MODE": "${io_mode}"
            }
        }
    ]
}
EOF

                    # Ensure file has proper permissions
                    chmod 644 "$config_file"

                    echo "Running benchmark: Mode=$mode, I/O Mode=$io_mode, I/O Threads=$io_threads, Block Size=$block_size, Dataset Size=$dataset_size"
                    
                    # Run the benchmark using the main h5bench command with the JSON config
                    ./h5bench "$config_file" | tee "$log_file"

                    # Extract metrics from the log file and run directory
                    run_subdir=$(find "$run_dir" -mindepth 1 -maxdepth 1 -type d | sort -r | head -1)
                    
                    if [ -n "$run_subdir" ]; then
                        stdout_file="$run_subdir/stdout"
                        
                        if [ -f "$stdout_file" ]; then
                            echo "Extracting metrics from stdout file: $stdout_file"
                            
                            # Extract ALL performance metrics from the stdout file in the same order as displayed
                            ranks=$(grep "Total number of ranks:" "$stdout_file" | awk '{print $5}')
                            compute_time=$(grep "Total emulated compute time:" "$stdout_file" | awk '{print $5}')
                            
                            # Extract write size with units
                            write_size_line=$(grep "Total write size:" "$stdout_file")
                            write_size=$(echo "$write_size_line" | awk '{print $4" "$5}')
                            
                            raw_write_time=$(grep "Raw write time:" "$stdout_file" | awk '{print $4}')
                            metadata_time=$(grep "Metadata time:" "$stdout_file" | awk '{print $3}')
                            h5fcreate_time=$(grep "H5Fcreate() time:" "$stdout_file" | awk '{print $3}')
                            h5fflush_time=$(grep "H5Fflush() time:" "$stdout_file" | awk '{print $3}')
                            h5fclose_time=$(grep "H5Fclose() time:" "$stdout_file" | awk '{print $3}')
                            completion_time=$(grep "Observed completion time:" "$stdout_file" | awk '{print $4}')
                            
                            # Extract write rates with units
                            raw_write_rate_line=$(grep "SYNC Raw write rate:" "$stdout_file")
                            raw_write_rate=$(echo "$raw_write_rate_line" | awk '{print $5" "$6}')
                            
                            observed_write_rate_line=$(grep "SYNC Observed write rate:" "$stdout_file")
                            observed_write_rate=$(echo "$observed_write_rate_line" | awk '{print $5" "$6}')
                            
                            # Log what we found
                            echo "Successfully extracted metrics from $stdout_file"
                            
                            # Print current progress for tracking long-running benchmarks
                            mode_count=${#MODES[@]}
                            io_mode_count=${#IO_MODES[@]}
                            io_threads_count=${#IO_THREADS[@]}
                            block_size_count=${#BLOCK_SIZES[@]}
                            dataset_size_count=${#DATASET_SIZES[@]}
                            
                            total_runs=$((mode_count * io_mode_count * io_threads_count * block_size_count * dataset_size_count))
                            current_mode_idx=$(printf '%s\n' "${MODES[@]}" | grep -n "^$mode$" | cut -d: -f1)
                            current_io_mode_idx=$(printf '%s\n' "${IO_MODES[@]}" | grep -n "^$io_mode$" | cut -d: -f1)
                            current_io_threads_idx=$(printf '%s\n' "${IO_THREADS[@]}" | grep -n "^$io_threads$" | cut -d: -f1)
                            current_block_size_idx=$(printf '%s\n' "${BLOCK_SIZES[@]}" | grep -n "^$block_size$" | cut -d: -f1)
                            current_dataset_size_idx=$(printf '%s\n' "${DATASET_SIZES[@]}" | grep -n "^$dataset_size$" | cut -d: -f1)
                            
                            completed_runs=$(( 
                                (current_mode_idx - 1) * io_mode_count * io_threads_count * block_size_count * dataset_size_count +
                                (current_io_mode_idx - 1) * io_threads_count * block_size_count * dataset_size_count +
                                (current_io_threads_idx - 1) * block_size_count * dataset_size_count +
                                (current_block_size_idx - 1) * dataset_size_count +
                                current_dataset_size_idx
                            ))
                            
                            progress_pct=$(( completed_runs * 100 / total_runs ))
                            echo "Benchmark progress: $completed_runs/$total_runs ($progress_pct%)"
                            
                        else
                            echo "WARNING: stdout file not found in run directory $run_subdir"
                            # Print the directory contents for debugging
                            echo "Contents of $run_subdir:"
                            ls -la "$run_subdir"
                            
                            # Check stderr for errors
                            stderr_file="$run_subdir/stderr"
                            if [ -f "$stderr_file" ]; then
                                echo "Checking stderr for errors:"
                                grep -i error "$stderr_file" | head -10
                            fi
                        fi
                    else
                        echo "WARNING: No run subdirectory found in $run_dir"
                        # Print contents of the run directory
                        echo "Contents of $run_dir:"
                        ls -la "$run_dir"
                    fi
                    
                    # If we couldn't extract metrics, the run might have failed
                    if [ -z "$completion_time" ] || [ -z "$raw_write_rate" ]; then
                        echo "WARNING: Could not extract metrics, benchmark might have failed"
                        
                        # Check if there's any output at all
                        if [ -f "$log_file" ]; then
                            echo "Contents of log file ($log_file):"
                            head -50 "$log_file" | grep -v "^$"
                        fi
                    fi

                    # Fallback to zero if any metric is missing
                    write_time=${completion_time:-0}
                    metadata_time=${metadata_time:-0}
                    raw_write_time=${raw_write_time:-0}
                    flush_time=${h5fflush_time:-0}
                    close_time=${h5fclose_time:-0}
                    sync_raw_write_rate=${raw_write_rate:-0}
                    sync_observed_write_rate=${observed_write_rate:-0}
                    ranks=${ranks:-0}
                    compute_time=${compute_time:-0}
                    write_size=${write_size:-"0 MB"}
                    h5fcreate_time=${h5fcreate_time:-0}

                    # Append the results to the CSV file with extended format
                    echo "$mode,$io_mode,$io_threads,$block_size,$dataset_size,$ranks,$compute_time,\"$write_size\",$raw_write_time,$metadata_time,$h5fcreate_time,$h5fflush_time,$h5fclose_time,$completion_time,\"$raw_write_rate\",\"$observed_write_rate\"" >> "$output_csv"

                    # Print a summary of the results
                    echo "Results for Mode=$mode, I/O Mode=$io_mode, I/O Threads=$io_threads, Block Size=$block_size, Dataset Size=$dataset_size:"
                    echo "  Ranks: $ranks"
                    echo "  Compute Time: $compute_time s"
                    echo "  Write Size: $write_size"
                    echo "  Raw Write Time: $raw_write_time s"
                    echo "  Metadata Time: $metadata_time s"
                    echo "  H5Fcreate() Time: $h5fcreate_time s"
                    echo "  H5Fflush() Time: $h5fflush_time s"
                    echo "  H5Fclose() Time: $h5fclose_time s"
                    echo "  Observed Completion Time: $completion_time s"
                    echo "  Raw Write Rate: $raw_write_rate"
                    echo "  Observed Write Rate: $observed_write_rate"
                    echo ""

                done
            done
        done
    done
done

echo "Benchmarking complete. Results saved in $output_csv"

# Generate summary of the best performers
echo ""
echo "===== BENCHMARK SUMMARY ====="
echo ""

# For GPU mode with SYNC
echo "Top 3 GPU SYNC Raw Write Rates:"
awk -F, 'NR>1 && $1=="GPU" && $2=="SYNC" {gsub(/"/, "", $15); split($15, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 GPU SYNC Observed Write Rates:"
awk -F, 'NR>1 && $1=="GPU" && $2=="SYNC" {gsub(/"/, "", $16); split($16, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# For GPU mode with ASYNC
echo ""
echo "Top 3 GPU ASYNC Raw Write Rates:"
awk -F, 'NR>1 && $1=="GPU" && $2=="ASYNC" {gsub(/"/, "", $15); split($15, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 GPU ASYNC Observed Write Rates:"
awk -F, 'NR>1 && $1=="GPU" && $2=="ASYNC" {gsub(/"/, "", $16); split($16, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# For CPU mode with SYNC
echo ""
echo "Top 3 CPU SYNC Raw Write Rates:"
awk -F, 'NR>1 && $1=="CPU" && $2=="SYNC" {gsub(/"/, "", $15); split($15, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 CPU SYNC Observed Write Rates:"
awk -F, 'NR>1 && $1=="CPU" && $2=="SYNC" {gsub(/"/, "", $16); split($16, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# For CPU mode with ASYNC
echo ""
echo "Top 3 CPU ASYNC Raw Write Rates:"
awk -F, 'NR>1 && $1=="CPU" && $2=="ASYNC" {gsub(/"/, "", $15); split($15, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

echo ""
echo "Top 3 CPU ASYNC Observed Write Rates:"
awk -F, 'NR>1 && $1=="CPU" && $2=="ASYNC" {gsub(/"/, "", $16); split($16, rate, " "); print $3 " threads, " $4 " block size, " $5 " dataset: " rate[1] " " rate[2]}' "$output_csv" | sort -t: -k2 -nr | head -3

# Compare SYNC vs ASYNC by thread count
echo ""
echo "SYNC vs ASYNC Comparison by Thread Count (Average Observed Write Rate):"
for threads in "${IO_THREADS[@]}"; do
    sync_avg=$(awk -F, -v t="$threads" 'NR>1 && $2=="SYNC" && $3==t {gsub(/"/, "", $16); split($16, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0)}' "$output_csv")
    async_avg=$(awk -F, -v t="$threads" 'NR>1 && $2=="ASYNC" && $3==t {gsub(/"/, "", $16); split($16, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0)}' "$output_csv")
    
    # Calculate improvement percentage if both values are non-zero
    if (( $(echo "$sync_avg > 0" | bc -l) )) && (( $(echo "$async_avg > 0" | bc -l) )); then
        improvement=$(echo "($async_avg - $sync_avg) / $sync_avg * 100" | bc -l)
        printf "%s threads: SYNC=%.2f GB/s, ASYNC=%.2f GB/s (%.2f%% improvement)\n" "$threads" "$sync_avg" "$async_avg" "$improvement"
    else
        printf "%s threads: SYNC=%.2f GB/s, ASYNC=%.2f GB/s\n" "$threads" "$sync_avg" "$async_avg"
    fi
done

# Summary by thread count (average across all configurations)
echo ""
echo "Average Performance by Thread Count for SYNC Mode:"
echo "Threads,Avg_Raw_Write_Rate,Avg_Observed_Write_Rate"
for threads in "${IO_THREADS[@]}"; do
    avg_raw=$(awk -F, -v t="$threads" 'NR>1 && $2=="SYNC" && $3==t {gsub(/"/, "", $15); split($15, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    avg_obs=$(awk -F, -v t="$threads" 'NR>1 && $2=="SYNC" && $3==t {gsub(/"/, "", $16); split($16, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    echo "$threads,$avg_raw,$avg_obs"
done

echo ""
echo "Average Performance by Thread Count for ASYNC Mode:"
echo "Threads,Avg_Raw_Write_Rate,Avg_Observed_Write_Rate"
for threads in "${IO_THREADS[@]}"; do
    avg_raw=$(awk -F, -v t="$threads" 'NR>1 && $2=="ASYNC" && $3==t {gsub(/"/, "", $15); split($15, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
    avg_obs=$(awk -F, -v t="$threads" 'NR>1 && $2=="ASYNC" && $3==t {gsub(/"/, "", $16); split($16, rate, " "); sum+=rate[1]; count++} END {print (count>0 ? sum/count : 0) " GB/s"}' "$output_csv")
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