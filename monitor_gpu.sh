#!/bin/bash

# GPU Monitoring Script for h5bench
# Tracks GPU VRAM usage during benchmarks to confirm CUDA memory allocation

# Configuration
LOG_FILE="gpu_demon.log"
INTERVAL=10  # seconds between samples (10 minutes = 60 samples)
DURATION=7200  # default 2 hours, can be overridden

# Function to get GPU info with comprehensive data
get_gpu_info() {
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.gr,clocks.mem,clocks.sm --format=csv,noheader,nounits
}

# Function to get GPU processes
get_gpu_processes() {
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
}

# Function to get CPU utilization
get_cpu_info() {
    # Get overall CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    # Get load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    echo "${cpu_usage},${load_avg}"
}

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to monitor GPU
monitor_gpu() {
    local duration=${1:-$DURATION}
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    
    log_with_timestamp "Starting comprehensive GPU monitoring for ${duration} seconds"
    log_with_timestamp "Log file: $LOG_FILE"
    log_with_timestamp "Monitoring interval: ${INTERVAL} seconds"
    log_with_timestamp "Expected samples: $((duration / INTERVAL))"
    echo ""
    
    # Write CSV header for easy parsing
    echo "timestamp,gpu_idx,name,mem_used_mb,mem_total_mb,gpu_util_pct,mem_util_pct,temp_c,power_w,gr_clock_mhz,mem_clock_mhz,sm_clock_mhz,cpu_usage_pct,load_avg,processes" >> "$LOG_FILE"
    
    local sample_count=0
    while [ $(date +%s) -lt $end_time ]; do
        local current_time=$(date '+%Y-%m-%d %H:%M:%S')
        local gpu_info=$(get_gpu_info)
        local cpu_info=$(get_cpu_info)
        local gpu_processes=$(get_gpu_processes)
        
        if [ -n "$gpu_info" ]; then
            # Parse GPU info
            echo "$gpu_info" | while IFS=',' read -r timestamp gpu_index name mem_used mem_total gpu_util mem_util temp power gr_clock mem_clock sm_clock; do
                # Clean up the values (remove spaces)
                gpu_index=$(echo "$gpu_index" | xargs)
                name=$(echo "$name" | xargs)
                mem_used=$(echo "$mem_used" | xargs)
                mem_total=$(echo "$mem_total" | xargs)
                gpu_util=$(echo "$gpu_util" | xargs)
                mem_util=$(echo "$mem_util" | xargs)
                temp=$(echo "$temp" | xargs)
                power=$(echo "$power" | xargs)
                gr_clock=$(echo "$gr_clock" | xargs)
                mem_clock=$(echo "$mem_clock" | xargs)
                sm_clock=$(echo "$sm_clock" | xargs)
                
                # Parse CPU info
                local cpu_usage=$(echo "$cpu_info" | cut -d',' -f1)
                local load_avg=$(echo "$cpu_info" | cut -d',' -f2)
                
                # Format processes info
                local processes_info=""
                if [ -n "$gpu_processes" ]; then
                    processes_info=$(echo "$gpu_processes" | tr '\n' ';' | sed 's/;$//')
                else
                    processes_info="none"
                fi
                
                # Write CSV line
                echo "$current_time,$gpu_index,$name,$mem_used,$mem_total,$gpu_util,$mem_util,$temp,$power,$gr_clock,$mem_clock,$sm_clock,$cpu_usage,$load_avg,\"$processes_info\"" >> "$LOG_FILE"
                
                # Also show a summary every 10 samples (every ~100 seconds with 10s interval)
                if [ $((sample_count % 10)) -eq 0 ]; then
                    printf "[%s] GPU%d: %sMB/%sMB, GPU:%s%%, Mem:%s%%, CPU:%s%%, Load:%s\n" \
                           "$current_time" "$gpu_index" "$mem_used" "$mem_total" "$gpu_util" "$mem_util" "$cpu_usage" "$load_avg"
                fi
            done
        else
            log_with_timestamp "ERROR: No GPU information available"
        fi
        
        sample_count=$((sample_count + 1))
        sleep "$INTERVAL"
    done
    
    log_with_timestamp "GPU monitoring completed. Total samples: $sample_count"
}

# Function to show current GPU status
show_gpu_status() {
    echo "=== Current GPU Status ==="
    nvidia-smi
    echo ""
}

# Function to analyze the log file
analyze_log() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo "ERROR: Log file '$log_file' not found!"
        return 1
    fi
    
    echo "=== GPU Monitoring Analysis ==="
    echo "Log file: $log_file"
    echo ""
    
    # Skip header line and analyze data
    local data_lines=$(tail -n +2 "$log_file" | grep -v "^\[" | grep -v "^$")
    
    if [ -z "$data_lines" ]; then
        echo "No data found in log file."
        return 1
    fi
    
    # Peak GPU memory usage
    local peak_mem=$(echo "$data_lines" | cut -d',' -f4 | sort -nr | head -1)
    echo "Peak GPU Memory Usage: ${peak_mem}MB"
    
    # Average GPU memory usage
    local avg_mem=$(echo "$data_lines" | cut -d',' -f4 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count}')
    echo "Average GPU Memory Usage: ${avg_mem}MB"
    
    # Peak GPU utilization
    local peak_gpu_util=$(echo "$data_lines" | cut -d',' -f6 | sort -nr | head -1)
    echo "Peak GPU Utilization: ${peak_gpu_util}%"
    
    # Average GPU utilization
    local avg_gpu_util=$(echo "$data_lines" | cut -d',' -f6 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count}')
    echo "Average GPU Utilization: ${avg_gpu_util}%"
    
    # Peak CPU usage
    local peak_cpu=$(echo "$data_lines" | cut -d',' -f13 | sort -nr | head -1)
    echo "Peak CPU Usage: ${peak_cpu}%"
    
    # Average CPU usage
    local avg_cpu=$(echo "$data_lines" | cut -d',' -f13 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count}')
    echo "Average CPU Usage: ${avg_cpu}%"
    
    # Total samples
    local total_samples=$(echo "$data_lines" | wc -l)
    echo "Total Samples: $total_samples"
    
    echo ""
    echo "=== Recent Activity (last 5 samples) ==="
    echo "$data_lines" | tail -5 | while IFS=',' read -r timestamp gpu_idx name mem_used mem_total gpu_util mem_util temp power gr_clock mem_clock sm_clock cpu_usage load_avg processes; do
        printf "%-20s GPU%d: %sMB, GPU:%s%%, CPU:%s%%\n" "$timestamp" "$gpu_idx" "$mem_used" "$gpu_util" "$cpu_usage"
    done
}

# Function to show help
show_help() {
    echo "GPU Monitor for h5bench (Enhanced for long-running benchmarks)"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --duration SECONDS    Duration to monitor (default: 7200 = 2 hours)"
    echo "  -i, --interval SECONDS    Sampling interval (default: 10 seconds)"
    echo "  -s, --status             Show current GPU status and exit"
    echo "  -a, --analyze [LOG_FILE] Analyze existing log file (default: gpu_demon.log)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Monitor for 2 hours (10s intervals)"
    echo "  $0 -d 3600              # Monitor for 1 hour"
    echo "  $0 -i 5                 # Sample every 5 seconds"
    echo "  $0 -s                   # Show current status"
    echo "  $0 -a                   # Analyze gpu_demon.log"
    echo "  $0 -a custom.log        # Analyze custom log file"
    echo ""
    echo "Output:"
    echo "  - Creates gpu_demon.log with CSV format"
    echo "  - Includes GPU memory, utilization, clocks, temperature, power"
    echo "  - Includes CPU usage and load average"
    echo "  - Includes running processes on GPU"
    echo "  - Shows summary every 10 samples (~100 seconds)"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            if [[ $INTERVAL -lt 1 ]]; then
                echo "ERROR: Interval must be a positive number (got: $INTERVAL)"
                exit 1
            fi
            shift 2
            ;;
        -s|--status)
            show_gpu_status
            exit 0
            ;;
        -a|--analyze)
            if [ -n "$2" ] && [[ ! "$2" =~ ^- ]]; then
                analyze_log "$2"
            else
                analyze_log "$LOG_FILE"
            fi
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Show initial status
show_gpu_status

# Start monitoring
monitor_gpu "$DURATION"
