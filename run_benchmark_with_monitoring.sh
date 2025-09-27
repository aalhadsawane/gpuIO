#!/bin/bash

# Benchmark Runner with GPU Monitoring
# Runs h5bench benchmarks while monitoring GPU VRAM usage

# Configuration
BENCHMARK_SCRIPT=""
MONITOR_DURATION=7200  # 2 hours default (for long-running benchmarks)
MONITOR_INTERVAL=10    # 10 second intervals (suitable for long monitoring)
LOG_DIR="benchmark_logs_$(date +%Y%m%d_%H%M%S)"

# Function to show help
show_help() {
    echo "Benchmark Runner with GPU Monitoring"
    echo ""
    echo "Usage: $0 [OPTIONS] <benchmark_script>"
    echo ""
    echo "Arguments:"
    echo "  benchmark_script         Path to benchmark script (run_h5bench_write.sh or run_h5bench_read.sh)"
    echo ""
    echo "Options:"
    echo "  -d, --duration SECONDS   Monitor duration (default: 7200 = 2 hours)"
    echo "  -i, --interval SECONDS   Monitoring interval (default: 10 seconds)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 run_h5bench_write.sh"
    echo "  $0 -d 3600 run_h5bench_read.sh  # Monitor for 1 hour"
    echo "  $0 -i 5 run_h5bench_write.sh   # Sample every 5 seconds"
    echo ""
}

# Function to setup monitoring
setup_monitoring() {
    echo "Setting up GPU monitoring..."
    mkdir -p "$LOG_DIR"
    
    # Start GPU monitoring in background (with nohup for detached execution)
    nohup ./monitor_gpu.sh -d "$MONITOR_DURATION" -i "$MONITOR_INTERVAL" > "$LOG_DIR/gpu_monitor_output.log" 2>&1 &
    MONITOR_PID=$!
    
    echo "GPU monitoring started (PID: $MONITOR_PID)"
    echo "Log directory: $LOG_DIR"
    echo ""
}

# Function to cleanup
cleanup() {
    echo ""
    echo "Cleaning up..."
    
    # Stop GPU monitoring
    if [ ! -z "$MONITOR_PID" ]; then
        echo "Stopping GPU monitoring (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
    
    echo "Cleanup completed."
}

# Function to run benchmark
run_benchmark() {
    local script="$1"
    
    if [ ! -f "$script" ]; then
        echo "ERROR: Benchmark script '$script' not found!"
        exit 1
    fi
    
    echo "Running benchmark: $script"
    echo "Start time: $(date)"
    echo ""
    
    # Run the benchmark and capture output
    # Add ./ prefix if script doesn't start with / (absolute path)
    if [[ "$script" != /* ]]; then
        script="./$script"
    fi
    
    # Ensure we're in the correct directory
    cd "$(dirname "$0")"
    echo "Current directory: $(pwd)"
    echo "Script path: $script"
    
    nohup "$script" 2>&1 | tee "$LOG_DIR/benchmark_output.log"
    local benchmark_exit_code=${PIPESTATUS[0]}
    
    echo ""
    echo "Benchmark completed at: $(date)"
    echo "Exit code: $benchmark_exit_code"
    
    return $benchmark_exit_code
}

# Function to generate summary
generate_summary() {
    echo ""
    echo "=== BENCHMARK SUMMARY ==="
    echo "Log directory: $LOG_DIR"
    echo "Benchmark script: $BENCHMARK_SCRIPT"
    echo "Start time: $(date)"
    echo ""
    
    # Show GPU usage summary
    if [ -f "$LOG_DIR/gpu_demon.log" ]; then
        echo "=== GPU USAGE SUMMARY ==="
        
        # Use the analyze function from monitor_gpu.sh
        ./monitor_gpu.sh -a "$LOG_DIR/gpu_demon.log"
        
        echo ""
        echo "Detailed GPU monitoring log: $LOG_DIR/gpu_demon.log"
        echo "Log format: CSV with columns: timestamp,gpu_idx,name,mem_used_mb,mem_total_mb,gpu_util_pct,mem_util_pct,temp_c,power_w,gr_clock_mhz,mem_clock_mhz,sm_clock_mhz,cpu_usage_pct,load_avg,processes"
    fi
    
    echo "Benchmark output log: $LOG_DIR/benchmark_output.log"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            MONITOR_DURATION="$2"
            shift 2
            ;;
        -i|--interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$BENCHMARK_SCRIPT" ]; then
                BENCHMARK_SCRIPT="$1"
            else
                echo "ERROR: Multiple benchmark scripts specified"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if benchmark script is provided
if [ -z "$BENCHMARK_SCRIPT" ]; then
    echo "ERROR: No benchmark script specified"
    show_help
    exit 1
fi

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Main execution
echo "=== BENCHMARK RUNNER WITH GPU MONITORING ==="
echo "Benchmark script: $BENCHMARK_SCRIPT"
echo "Monitor duration: $MONITOR_DURATION seconds"
echo "Monitor interval: $MONITOR_INTERVAL seconds"
echo ""

# Setup monitoring
setup_monitoring

# Give monitoring a moment to start
sleep 2

# Run benchmark
run_benchmark "$BENCHMARK_SCRIPT"
BENCHMARK_EXIT_CODE=$?

# Generate summary
generate_summary

exit $BENCHMARK_EXIT_CODE
