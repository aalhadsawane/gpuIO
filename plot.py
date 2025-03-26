import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
import argparse
import sys
import glob
import os

# Import generate_decision_tree from make_decision_tree.py
from make_decision_tree import generate_decision_tree

def find_latest_run():
    # Get the CSV directory path
    csv_dir = Path(__file__).parent / 'CSV'
    
    # Find all RunN directories
    run_dirs = glob.glob(str(csv_dir / 'Run*'))
    
    if not run_dirs:
        return None
    
    # Extract run numbers and find the latest
    run_numbers = [int(d.split('Run')[-1]) for d in run_dirs]
    latest_run = max(run_numbers)
    
    return latest_run

def get_data_path():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate performance plots and decision tree from benchmark data')
    parser.add_argument('--data-path', type=str, help='Path to the raw_output.csv file')
    parser.add_argument('--output-dir', type=str, help='Directory to save the plots')
    args = parser.parse_args()

    # Find the latest run number
    latest_run = find_latest_run()
    if latest_run is None:
        print("\nError: No RunN directories found in CSV folder")
        sys.exit(1)

    # If data path is provided as argument, use it
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Use the latest run's raw_output.csv
        data_path = Path(__file__).parent / 'CSV' / f'Run{latest_run}' / 'raw_output.csv'
        print(f"\nUsing data from latest run: Run{latest_run}")

    # If output directory is provided as argument, use it
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use a 'graphs' subdirectory inside the RunN directory
        output_dir = Path(__file__).parent / 'CSV' / f'Run{latest_run}' / 'graphs'
        print(f"Using output directory: Run{latest_run}/graphs")

    return data_path, output_dir

# Get the data path and output directory
data_path, output_dir = get_data_path()

# Check if input file exists
if not data_path.exists():
    print(f"\nError: File not found at {data_path}")
    print("Please provide a valid path to raw_output.csv")
    sys.exit(1)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nReading data from: {data_path}")
print(f"Saving plots to: {output_dir}")

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Read the data
df = pd.read_csv(data_path, comment='#')  # Explicitly handle lines starting with # as comments

# Convert write rates to numeric values (remove units)
df['raw_write_rate'] = df['raw_write_rate'].str.extract('(\d+\.?\d*)').astype(float)
df['observed_write_rate'] = df['observed_write_rate'].str.extract('(\d+\.?\d*)').astype(float)

# Add block size in KB for better readability in plots
df['block_size_kb'] = df['block_size'].astype(int) / 1024
df['io_threads'] = df['io_threads'].astype(int)

# Rename GPU to GDS in the dataset for display purposes
df.loc[df['mode'] == 'GPU', 'mode'] = 'GDS'

# Calculate percentiles for consistent scaling across all plots
p5 = np.percentile(df['observed_write_rate'], 5)
p95 = np.percentile(df['observed_write_rate'], 95)
print(f"Using percentile-based scaling: P5={p5:.2f}, P95={p95:.2f}")

# Calculate dynamic y-axis limit based on data
y_max = min(p95 * 1.1, max(df['observed_write_rate']) * 1.1)  # Use 110% of P95 or max value, whichever is smaller
y_min = max(0, p5 * 0.9)  # Use 90% of P5 or 0, whichever is larger

# Get unique values for axes
block_sizes = sorted(df['block_size_kb'].unique())
thread_counts = sorted(df['io_threads'].unique())

# ======== PLOT 1: Performance Heatmap with Raw Data Points ========
# Create separate plots for each dataset size
for dataset_size in df['dataset_size'].unique():
    plt.figure(figsize=(14, 10))
    
    # Create subplots for GDS and CPU modes
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle(f'Performance Heatmap: {dataset_size} Dataset', fontsize=20)
    
    # Define subplot locations
    positions = {
        'GDS': 0,
        'CPU': 1
    }
    
    # Define the color mapping for regular values (using percentiles)
    main_norm = mcolors.Normalize(vmin=y_min, vmax=y_max)  # Use dynamic range
    main_cmap = plt.cm.viridis
    
    # Create a custom colormap for outliers (values outside the P5-P95 range)
    colors = [(0.5, 0, 0.5), (0, 0, 0.5)]  # Purple to dark blue for < P5
    cmap_low = LinearSegmentedColormap.from_list('custom_low', colors, N=100)
    
    colors = [(1, 0, 0), (1, 0.5, 0)]  # Red to orange for > P95
    cmap_high = LinearSegmentedColormap.from_list('custom_high', colors, N=100)
    
    # Process each mode
    for mode, col in positions.items():
        ax = axs[col]
        
        # Filter data for current mode and dataset_size
        mask = (df['mode'] == mode) & (df['dataset_size'] == dataset_size)
        subset = df[mask]
        
        # Create unique identifier for each thread/block size combination
        pivot_data = subset.pivot_table(
            values='observed_write_rate',
            index='io_threads',
            columns='block_size_kb',
            aggfunc='mean'
        )
        
        # Create a masked array for values within the main range and outliers
        values_array = pivot_data.values
        mask_within_range = (values_array >= p5) & (values_array <= p95)
        mask_below_range = values_array < p5
        mask_above_range = values_array > p95
        
        # Plot the main heatmap with the dynamic range
        heatmap = sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap=main_cmap, 
                  ax=ax, cbar=True, annot_kws={"size": 10}, 
                  vmin=y_min, vmax=y_max, 
                  cbar_kws={'label': 'Write Speed (GB/s)'})
        
        # Now overlay the outliers with different colors
        for i in range(pivot_data.shape[0]):
            for j in range(pivot_data.shape[1]):
                val = pivot_data.iloc[i, j]
                if not np.isnan(val):
                    if val < p5:
                        # Purple for low values
                        color = 'purple'
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.7))
                        ax.text(j + 0.5, i + 0.5, f'{val:.1f}', 
                                ha='center', va='center', color='white', fontweight='bold')
                    elif val > p95:
                        # Red for high values
                        color = 'red'
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.7))
                        ax.text(j + 0.5, i + 0.5, f'{val:.1f}', 
                                ha='center', va='center', color='white', fontweight='bold')
        
        # Add color bar explanation
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(f'Write Speed (GB/s) - P5 to P95 Range ({p5:.1f} to {p95:.1f})', fontsize=12)
        
        # Add explanation for outlier colors
        ax.text(0.5, -0.1, f"Purple: < {p5:.1f} GB/s, Red: > {p95:.1f} GB/s", 
                transform=ax.transAxes, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Set axis limits based on data
        ax.set_xlim(0, len(block_sizes))
        ax.set_ylim(0, len(thread_counts))
        
        # Set axis labels
        ax.set_xticks(np.arange(len(block_sizes)) + 0.5)
        ax.set_yticks(np.arange(len(thread_counts)) + 0.5)
        ax.set_xticklabels([f'{x:.0f}' for x in block_sizes])
        ax.set_yticklabels([f'{x}' for x in thread_counts])
        
        ax.set_title(f'{mode}', fontsize=16)
        ax.set_xlabel('Block Size (KB)', fontsize=14)
        ax.set_ylabel('I/O Threads', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'heatmap_with_raw_points_{dataset_size}.png', dpi=300)
    plt.close()

# ======== PLOT 2: Comparative Performance Map ========
# Create a plot showing which mode (GDS/CPU) performs better for each config

# Find global min/max for consistent scale across all performance maps for differences
all_diffs = []
for dataset_size in df['dataset_size'].unique():
    subset = df[df['dataset_size'] == dataset_size]
    
    # Create pivot tables for GDS and CPU
    gds_data = subset[subset['mode'] == 'GDS'].pivot_table(
        values='observed_write_rate',
        index='io_threads',
        columns='block_size_kb',
        aggfunc='mean'
    )
    
    cpu_data = subset[subset['mode'] == 'CPU'].pivot_table(
        values='observed_write_rate',
        index='io_threads',
        columns='block_size_kb',
        aggfunc='mean'
    )
    
    # Calculate performance difference (GDS - CPU)
    diff_data = gds_data - cpu_data
    all_diffs.append(diff_data)

# Calculate percentile-based scale for differences
flat_diffs = np.concatenate([d.values.flatten() for d in all_diffs])
flat_diffs = flat_diffs[~np.isnan(flat_diffs)]  # Remove NaN values
diff_p5 = np.percentile(flat_diffs, 5)  # 5th percentile
diff_p95 = np.percentile(flat_diffs, 95)  # 95th percentile
diff_vmax = max(abs(diff_p5), abs(diff_p95))  # Symmetric scale based on percentiles
print(f"Performance difference percentiles: P5={diff_p5:.2f}, P95={diff_p95:.2f}, vmax={diff_vmax:.2f}")

for dataset_size in df['dataset_size'].unique():
    # Create a figure 
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'GDS vs CPU Performance Comparison: {dataset_size} Dataset', fontsize=20)
    
    # Filter data for current dataset_size
    subset = df[df['dataset_size'] == dataset_size]
    
    # Create pivot tables for GDS and CPU
    gds_data = subset[subset['mode'] == 'GDS'].pivot_table(
        values='observed_write_rate',
        index='io_threads',
        columns='block_size_kb',
        aggfunc='mean'
    )
    
    cpu_data = subset[subset['mode'] == 'CPU'].pivot_table(
        values='observed_write_rate',
        index='io_threads',
        columns='block_size_kb',
        aggfunc='mean'
    )
    
    # Calculate performance difference (GDS - CPU)
    diff_data = gds_data - cpu_data
    
    # Create a custom diverging colormap with neutral center
    colors = ['blue', 'lightblue', 'white', 'lightgreen', 'green']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    # Use the percentile-based scale for symmetric color scaling
    vmax = diff_vmax
    
    # Plot the performance difference
    im = ax.imshow(diff_data, cmap=cmap, interpolation='nearest', 
                 vmin=-vmax, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'GDS advantage (GB/s) [scale: ±{vmax:.2f} GB/s based on 5-95 percentiles]')
    
    # Add annotations with actual values, with special highlighting for outliers
    for i in range(len(diff_data.index)):
        for j in range(len(diff_data.columns)):
            value = diff_data.iloc[i, j]
            if not np.isnan(value):
                # Determine text color and style based on value
                if abs(value) > vmax:
                    # Outlier - make it bold and add a marker
                    text_color = 'black'
                    fontweight = 'bold'
                    value_text = f'{value:.1f}*'  # Add asterisk to highlight
                else:
                    text_color = 'black'
                    fontweight = 'normal'
                    value_text = f'{value:.1f}'
                    
                ax.text(j, i, value_text, 
                       ha='center', va='center', 
                       color=text_color, fontweight=fontweight)
    
    # Set tick labels
    ax.set_xticks(range(len(diff_data.columns)))
    ax.set_yticks(range(len(diff_data.index)))
    ax.set_xticklabels([f'{x:.0f}' for x in diff_data.columns])
    ax.set_yticklabels(diff_data.index)
    
    ax.set_title(f'Performance Difference (GDS - CPU)', fontsize=16)
    ax.set_xlabel('Block Size (KB)', fontsize=14)
    ax.set_ylabel('I/O Threads', fontsize=14)
    
    # Add a detailed explanation of the color scale
    plt.figtext(0.5, 0.01, 
                "Blue indicates CPU outperforms GDS. Green indicates GDS outperforms CPU.\n"
                "White indicates similar performance. Values show the exact difference in GB/s. * marks values outside the percentile range.", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_dir / f'gds_vs_cpu_comparison_{dataset_size}.png', dpi=300)
    plt.close()

# ======== PLOT 3: Points Plot with All Data Points ========
# Fixed-size points with write speed on y-axis, block size on x-axis, and different markers for thread counts

# For the point plots, use strict percentile-based y-axis range
y_padding = (p95 - p5) * 0.1  # 10% padding for visualization
y_min = p5 - y_padding
y_max = p95 + y_padding

for dataset_size in df['dataset_size'].unique():
    plt.figure(figsize=(12, 10))
    
    # Filter data
    subset = df[df['dataset_size'] == dataset_size]
    
    # Identify outliers
    outliers = subset[(subset['observed_write_rate'] < p5) | (subset['observed_write_rate'] > p95)]
    regular_data = subset[(subset['observed_write_rate'] >= p5) & (subset['observed_write_rate'] <= p95)]
    
    # Markers for different thread counts
    markers = {1: 'o', 2: 's', 4: '^', 8: 'd'}
    
    # Plot regular GDS data points - offset to the left
    gds_data = regular_data[regular_data['mode'] == 'GDS']
    for thread_count, marker in markers.items():
        thread_data = gds_data[gds_data['io_threads'] == thread_count]
        if not thread_data.empty:
            plt.scatter(
                thread_data['block_size_kb'] - 50,  # fixed left offset
                thread_data['observed_write_rate'],
                s=100,
                color='blue',
                alpha=0.7,
                marker=marker,
                edgecolor='darkblue',
                linewidth=1,
                label=f'GDS {thread_count} Threads'
            )
    
    # Plot regular CPU data points - offset to the right
    cpu_data = regular_data[regular_data['mode'] == 'CPU']
    for thread_count, marker in markers.items():
        thread_data = cpu_data[cpu_data['io_threads'] == thread_count]
        if not thread_data.empty:
            plt.scatter(
                thread_data['block_size_kb'] + 50,  # fixed right offset
                thread_data['observed_write_rate'],
                s=100,
                color='red',
                alpha=0.7,
                marker=marker,
                edgecolor='darkred',
                linewidth=1,
                label=f'CPU {thread_count} Threads'
            )
    
    # Add text labels for regular data points
    for _, row in gds_data.iterrows():
        plt.annotate(
            f"{row['io_threads']}t", 
            (row['block_size_kb'] - 50, row['observed_write_rate']),  # fixed offset
            xytext=(-15, 0),
            textcoords='offset points',
            fontsize=8,
            ha='right',
            color='blue'
        )
    
    for _, row in cpu_data.iterrows():
        plt.annotate(
            f"{row['io_threads']}t", 
            (row['block_size_kb'] + 50, row['observed_write_rate']),  # fixed offset
            xytext=(15, 0),
            textcoords='offset points',
            fontsize=8,
            ha='left',
            color='red'
        )
    
    # Handle outliers with special markers and arrows
    for _, row in outliers.iterrows():
        # Determine position and color based on mode
        if row['mode'] == 'GDS':
            x_pos = row['block_size_kb'] - 50  # fixed offset
            color = 'blue'
            arrow_offset = -30  # Offset for text to the left
            ha = 'right'
        else:  # CPU
            x_pos = row['block_size_kb'] + 50  # fixed offset
            color = 'red'
            arrow_offset = 30  # Offset for text to the right
            ha = 'left'
        
        # Determine arrow direction and text position based on outlier type
        if row['observed_write_rate'] > p95:
            # High outlier - place at top with arrow pointing up
            y_arrow_start = p95
            y_arrow_end = p95 + y_padding * 0.7
            arrow_style = '-|>'
            y_text = p95 + y_padding * 0.8
            text = f"{row['io_threads']}t: {row['observed_write_rate']:.1f} GB/s ↑"
        else:
            # Low outlier - place at bottom with arrow pointing down
            y_arrow_start = p5
            y_arrow_end = p5 - y_padding * 0.7
            arrow_style = '-|>'
            y_text = p5 - y_padding * 0.8
            text = f"{row['io_threads']}t: {row['observed_write_rate']:.1f} GB/s ↓"
        
        # Add arrow
        arrow = FancyArrowPatch(
            (x_pos, y_arrow_start),
            (x_pos, y_arrow_end),
            arrowstyle=arrow_style,
            color=color,
            alpha=0.7,
            mutation_scale=12,
            linewidth=1.5
        )
        plt.gca().add_patch(arrow)
        
        # Add text label
        plt.annotate(
            text,
            (x_pos, y_text),
            xytext=(arrow_offset, 0),
            textcoords='offset points',
            fontsize=8,
            ha=ha,
            va='center',
            color=color,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
    
    # Add gray connecting lines between related points for regular data
    for thread in regular_data['io_threads'].unique():
        for block in regular_data['block_size_kb'].unique():
            # Find matching GDS and CPU data points
            gds_point = gds_data[(gds_data['io_threads'] == thread) & (gds_data['block_size_kb'] == block)]
            cpu_point = cpu_data[(cpu_data['io_threads'] == thread) & (cpu_data['block_size_kb'] == block)]
            
            if not gds_point.empty and not cpu_point.empty:
                plt.plot(
                    [gds_point['block_size_kb'].values[0] - 50, cpu_point['block_size_kb'].values[0] + 50], 
                    [gds_point['observed_write_rate'].values[0], cpu_point['observed_write_rate'].values[0]], 
                    'gray', alpha=0.3, linestyle='--', linewidth=0.5
                )
    
    plt.title(f'Performance - {dataset_size} Dataset', fontsize=16)
    plt.xlabel('Block Size', fontsize=14)
    plt.ylabel('Write Speed (GB/s)', fontsize=14)
    
    # Set custom x-tick labels
    plt.xticks(
        [64, 256, 1024],  # Actual positions
        ['64KB', '256KB', '1MB'],  # Custom labels
        fontsize=12
    )
    
    # Set y-axis limits to exact P5-P95 range with padding
    plt.ylim(y_min, y_max)
    
    # Add reference lines at P5 and P95 with labels
    plt.axhline(y=p5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(plt.xlim()[0], p5, f'P5: {p5:.1f}', 
            ha='left', va='bottom', fontsize=8, alpha=0.7)
    
    plt.axhline(y=p95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(plt.xlim()[0], p95, f'P95: {p95:.1f}', 
            ha='left', va='top', fontsize=8, alpha=0.7)
    
    # Add annotation explaining outliers
    plt.figtext(0.5, 0.02, 
        "Values outside P5-P95 range are shown with arrows and labeled with actual values.", 
        ha='center', fontsize=10, 
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with one entry per thread count and mode
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Configuration", loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_dir / f'point_plot_{dataset_size}.png', dpi=300)
    plt.close()

# ======== PLOT 4: Bar Charts for Direct Comparison ========
# Individual bar charts for each configuration

for dataset_size in df['dataset_size'].unique():
    for threads in df['io_threads'].unique():
        for block_size in df['block_size_kb'].unique():
            # Filter data
            mask = ((df['dataset_size'] == dataset_size) & 
                   (df['io_threads'] == threads) & 
                   (df['block_size_kb'] == block_size))
            
            if df[mask].shape[0] > 0:
                # Create figure
                plt.figure(figsize=(8, 6))
                
                # Group data by mode and calculate mean performance
                grouped = df[mask].groupby(['mode'])['observed_write_rate'].mean().reset_index()
                
                # Colors for the modes
                colors = {'GDS': 'blue', 'CPU': 'red'}
                bar_colors = [colors[mode] for mode in grouped['mode']]
                
                # Create bar chart
                bars = plt.bar(grouped['mode'], grouped['observed_write_rate'], color=bar_colors)
                
                # Add value labels on top of each bar
                for idx, (bar, height) in enumerate(zip(bars, grouped['observed_write_rate'])):
                    # Determine if this value is an outlier
                    is_outlier = height < p5 or height > p95
                    
                    # Calculate position for label - either on top of bar (within range) or at the P5/P95 bound with arrow (outlier)
                    if is_outlier:
                        if height > p95:
                            # Add an arrow pointing up from P95 to indicate high outlier
                            arrow = FancyArrowPatch(
                                (bar.get_x() + bar.get_width()/2, p95), 
                                (bar.get_x() + bar.get_width()/2, p95 * 1.05), 
                                arrowstyle='-|>', 
                                color='gray',
                                mutation_scale=10,
                                linewidth=1
                            )
                            plt.gca().add_patch(arrow)
                            # Label above the arrow
                            text_y = p95 * 1.08
                        else:
                            # Add an arrow pointing down from P5 to indicate low outlier
                            arrow = FancyArrowPatch(
                                (bar.get_x() + bar.get_width()/2, p5), 
                                (bar.get_x() + bar.get_width()/2, p5 * 0.95), 
                                arrowstyle='-|>', 
                                color='gray',
                                mutation_scale=10,
                                linewidth=1
                            )
                            plt.gca().add_patch(arrow)
                            # Label below the arrow
                            text_y = p5 * 0.92
                        
                        # Add actual value with asterisk to indicate outlier
                        plt.text(bar.get_x() + bar.get_width()/2, text_y,
                                f'{height:.1f}*',
                                ha='center', va='center', fontsize=10, 
                                fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.7))
                    else:
                        # Normal label on top of bar
                        plt.text(bar.get_x() + bar.get_width()/2, height + (p95-p5)*0.01,
                                f'{height:.1f}',
                                ha='center', va='bottom', fontsize=10)
                
                plt.title(f"Performance Comparison: {threads} threads, {block_size:.0f}KB, {dataset_size}", fontsize=14)
                plt.xlabel('Mode', fontsize=12)
                plt.ylabel('Observed Write Rate (GB/s)', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                
                # Standardize y-axis scale to P5-P95
                y_padding = (p95 - p5) * 0.1  # 10% padding
                plt.ylim(p5 - y_padding, p95 + y_padding)
                
                # Add reference lines at P5 and P95
                plt.axhline(y=p5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                plt.axhline(y=p95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                plt.text(plt.xlim()[0], p5, f'P5: {p5:.1f}', ha='left', va='bottom', fontsize=8, alpha=0.7)
                plt.text(plt.xlim()[0], p95, f'P95: {p95:.1f}', ha='left', va='top', fontsize=8, alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'bar_chart_{dataset_size}_{threads}threads_{block_size:.0f}KB.png', dpi=300)
                plt.close()

# ======== PLOT 5: Combined Bar Chart for All Configurations ========
# One chart showing all configurations together

for dataset_size in df['dataset_size'].unique():
    # Create a figure
    plt.figure(figsize=(20, 12))
    
    # Prepare data
    plot_data = []
    labels = []
    
    # Collect data for each thread and block size combination
    for threads in df['io_threads'].unique():
        for block_size in df['block_size_kb'].unique():
            mask = ((df['dataset_size'] == dataset_size) & 
                   (df['io_threads'] == threads) & 
                   (df['block_size_kb'] == block_size))
            
            if df[mask].shape[0] > 0:
                # For each configuration, get the mode performance
                for mode in ['GDS', 'CPU']:
                    subset = df[mask & (df['mode'] == mode)]
                    if not subset.empty:
                        plot_data.append(subset['observed_write_rate'].mean())
                        labels.append(f"{threads}t_{block_size:.0f}KB_{mode}")
    
    # Calculate positions for grouped bars
    num_configs = len(plot_data) // 2  # Each config has 2 bars (GDS/CPU)
    group_positions = np.arange(num_configs)
    bar_width = 0.35
    
    # Plot the bars with positions
    plt.figure(figsize=(max(15, num_configs * 1.5), 10))
    
    # Create custom positions for each bar
    positions = []
    for i in range(num_configs):
        for j in range(2):  # 2 bars per group (GDS/CPU)
            positions.append(i + (j - 0.5) * bar_width)
    
    # Create color map
    colors = ['blue', 'red'] * num_configs
    
    # Plot bars
    bars = plt.bar(positions, plot_data, width=bar_width, color=colors)
    
    # Add value labels and arrows for outliers
    for bar, height in zip(bars, plot_data):
        # Determine if this value is an outlier
        is_outlier = height < p5 or height > p95
        
        if is_outlier:
            if height > p95:
                # Add an arrow pointing up from P95 to indicate high outlier
                arrow = FancyArrowPatch(
                    (bar.get_x() + bar.get_width()/2, p95), 
                    (bar.get_x() + bar.get_width()/2, p95 * 1.05), 
                    arrowstyle='-|>', 
                    color='gray',
                    mutation_scale=10,
                    linewidth=1
                )
                plt.gca().add_patch(arrow)
                # Label above the arrow
                text_y = p95 * 1.08
            else:
                # Add an arrow pointing down from P5 to indicate low outlier
                arrow = FancyArrowPatch(
                    (bar.get_x() + bar.get_width()/2, p5), 
                    (bar.get_x() + bar.get_width()/2, p5 * 0.95), 
                    arrowstyle='-|>', 
                    color='gray',
                    mutation_scale=10,
                    linewidth=1
                )
                plt.gca().add_patch(arrow)
                # Label below the arrow
                text_y = p5 * 0.92
            
            # Add actual value with asterisk to indicate outlier
            plt.text(bar.get_x() + bar.get_width()/2, text_y,
                    f'{height:.1f}*',
                    ha='center', va='center', fontsize=8, 
                    fontweight='bold', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.7))
        elif height > 10:  # Only label bars with significant height and within range
            plt.text(bar.get_x() + bar.get_width()/2, height + (p95-p5)*0.01,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set x-axis labels
    unique_configs = [f"{t}t_{b:.0f}KB" for t in df['io_threads'].unique() 
                     for b in df['block_size_kb'].unique()][:num_configs]
    plt.xticks(group_positions, unique_configs, rotation=45, ha='right')
    
    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='GDS'),
        Patch(facecolor='red', label='CPU')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'Performance Comparison Across All Configurations - {dataset_size} Dataset', fontsize=16)
    plt.xlabel('Configuration (Threads and Block Size)', fontsize=14)
    plt.ylabel('Observed Write Rate (GB/s)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Standardize y-axis scale to P5-P95
    y_padding = (p95 - p5) * 0.1  # 10% padding
    plt.ylim(p5 - y_padding, p95 + y_padding)
    
    # Add reference lines at P5 and P95
    plt.axhline(y=p5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=p95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(plt.xlim()[0], p5, f'P5: {p5:.1f}', ha='left', va='bottom', fontsize=8, alpha=0.7)
    plt.text(plt.xlim()[0], p95, f'P95: {p95:.1f}', ha='left', va='top', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'combined_bar_chart_{dataset_size}.png', dpi=300)
    plt.close()

# ======== PLOT 6: Strip Plots with Raw Data and Medians ========
# Shows all individual data points with outliers at the top with arrows

for dataset_size in df['dataset_size'].unique():
    # Check for outliers
    df_subset = df[df['dataset_size'] == dataset_size]
    Q1 = df_subset['observed_write_rate'].quantile(0.25)
    Q3 = df_subset['observed_write_rate'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_cutoff = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df_subset[df_subset['observed_write_rate'] > outlier_cutoff]
    regular_data = df_subset[df_subset['observed_write_rate'] <= outlier_cutoff]
    
    # Create plot by block size
    plt.figure(figsize=(20, 10))
    
    # Make the main plot with all data, but set ylim to focus on main distribution
    ax = plt.subplot(111)
    
    # Plot all data points
    sns.stripplot(
        data=df_subset,
        x='block_size_kb',
        y='observed_write_rate',
        hue='mode',
        size=5,
        palette={'GDS': 'blue', 'CPU': 'red'},
        dodge=True,
        jitter=True,
        alpha=0.7,
        ax=ax
    )
    
    # Add median lines for better comparison
    sns.boxplot(
        data=df_subset,
        x='block_size_kb',
        y='observed_write_rate',
        hue='mode',
        palette={'GDS': 'blue', 'CPU': 'red'},
        dodge=True,
        width=0.6,
        fliersize=0,
        showcaps=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'none'},
        whiskerprops={'color': 'none'},
        medianprops={'color': 'black', 'linewidth': 2},
        ax=ax
    )
    
    # Cap the y-axis to focus on main distribution but allow space for outliers
    top_limit = min(outlier_cutoff * 1.1, y_max)
    plt.ylim(y_min, top_limit)  # Use y_min (P5-based) instead of arbitrary 160
    
    # For each outlier, add an arrow pointing to it from the top of the plot
    for _, row in outliers.iterrows():
        # Determine dodge offset based on mode (to match stripplot dodge)
        dodge_offset = -0.2 if row['mode'] == 'GDS' else 0.2
        
        # Find the position on the x-axis
        x_pos = list(df_subset['block_size_kb'].unique()).index(row['block_size_kb']) + dodge_offset
        
        # Add arrow pointing to where the outlier would be
        arrow = FancyArrowPatch(
            (x_pos, top_limit * 0.9),  # Start from 90% of the top
            (x_pos, top_limit * 0.95),  # End at 95% of the top
            connectionstyle="arc3,rad=0",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color='gray'
        )
        ax.add_patch(arrow)
        
        # Add text annotation for the outlier value
        color = 'blue' if row['mode'] == 'GDS' else 'red'
        plt.text(x_pos, top_limit * 0.85, 
                f"{row['mode']}: {row['observed_write_rate']:.1f} GB/s", 
                color=color, ha='center', va='center', 
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    plt.title(f'Performance by Block Size - {dataset_size} Dataset', fontsize=18)
    plt.xlabel('Block Size (KB)', fontsize=14)
    plt.ylabel('Observed Write Rate (GB/s)', fontsize=14)
    plt.legend(title='Mode')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'stripplot_by_blocksize_{dataset_size}.png', dpi=300)
    plt.close()
    
    # Similar plot but with thread count on x-axis
    plt.figure(figsize=(20, 10))
    
    # Make the main plot with all data
    ax = plt.subplot(111)
    sns.stripplot(
        data=df_subset,
        x='io_threads',
        y='observed_write_rate',
        hue='mode',
        size=5,
        palette={'GDS': 'blue', 'CPU': 'red'},
        dodge=True,
        jitter=True,
        alpha=0.7,
        ax=ax
    )
    
    # Add median lines for better comparison
    sns.boxplot(
        data=df_subset,
        x='io_threads',
        y='observed_write_rate',
        hue='mode',
        palette={'GDS': 'blue', 'CPU': 'red'},
        dodge=True,
        width=0.6,
        fliersize=0,
        showcaps=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'none'},
        whiskerprops={'color': 'none'},
        medianprops={'color': 'black', 'linewidth': 2},
        ax=ax
    )
    
    # Cap the y-axis to focus on main distribution but allow space for outliers
    plt.ylim(y_min, top_limit)  # Use y_min (P5-based) instead of arbitrary 160
    
    # For each outlier, add an arrow pointing to it from the top of the plot
    for _, row in outliers.iterrows():
        # Determine dodge offset based on mode (to match stripplot dodge)
        dodge_offset = -0.2 if row['mode'] == 'GDS' else 0.2
        
        # Find the position on the x-axis
        x_pos = list(df_subset['io_threads'].unique()).index(row['io_threads']) + dodge_offset
        
        # Add arrow pointing to where the outlier would be
        arrow = FancyArrowPatch(
            (x_pos, top_limit * 0.9),  # Start from 90% of the top
            (x_pos, top_limit * 0.95),  # End at 95% of the top
            connectionstyle="arc3,rad=0",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color='gray'
        )
        ax.add_patch(arrow)
        
        # Add text annotation for the outlier value
        color = 'blue' if row['mode'] == 'GDS' else 'red'
        plt.text(x_pos, top_limit * 0.85, 
                f"{row['mode']}: {row['observed_write_rate']:.1f} GB/s", 
                color=color, ha='center', va='center', 
                fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    plt.title(f'Performance by Thread Count - {dataset_size} Dataset', fontsize=18)
    plt.xlabel('I/O Threads', fontsize=14)
    plt.ylabel('Observed Write Rate (GB/s)', fontsize=14)
    plt.legend(title='Mode')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'stripplot_by_threads_{dataset_size}.png', dpi=300)
    plt.close()

# After all other plots are generated, always generate the decision tree
try:
    # Generate decision tree - use the run directory (parent of output_dir) for decision tree output
    run_dir = output_dir.parent
    generate_decision_tree(df, run_dir)
    print("\nDecision tree analysis complete!")
except Exception as e:
    print(f"\nError generating decision tree: {e}")
    print("Decision tree generation failed, but other plots were created successfully.")

print(f"\nPlots have been saved to: {output_dir}")
