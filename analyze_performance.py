#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from sklearn.tree import plot_tree

def clean_write_rate(rate_str):
    # Remove whitespace and convert to lowercase for consistent comparison
    rate_str = rate_str.strip().lower()
    
    # Extract the numeric value
    value = float(rate_str.split()[0])
    
    # Convert to GB/s based on the unit
    if 'tb/s' in rate_str:
        return value * 1024  # Convert TB/s to GB/s
    elif 'gb/s' in rate_str:
        return value
    elif 'mb/s' in rate_str:
        return value / 1024  # Convert MB/s to GB/s
    else:
        raise ValueError(f"Unknown unit in rate string: {rate_str}")

def load_and_prepare_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points from {csv_path}")
    
    # Convert dataset_size to numeric (GB)
    df['dataset_size_gb'] = df['dataset_size'].str.replace('GB', '').astype(float)
    
    # Clean write rates
    df['observed_write_rate'] = df['observed_write_rate'].apply(clean_write_rate)
    df['raw_write_rate'] = df['raw_write_rate'].apply(clean_write_rate)
    
    # Check if we have both GPU and CPU data
    if not all(mode in df['mode'].unique() for mode in ['GPU', 'CPU']):
        print("Warning: Dataset does not contain both CPU and GPU data")
        return None, None, None, None
    
    # Group data by configuration and directly compare GPU vs CPU
    configs = df.groupby(['io_mode', 'io_threads', 'block_size', 'dataset_size'])
    
    # Store comparison results
    comparisons = []
    
    # Define a small threshold for equality (1% difference)
    equality_threshold = 0.01
    
    # For each configuration, compare GPU vs CPU
    for config, group in configs:
        gpu_group = group[group['mode'] == 'GPU']
        cpu_group = group[group['mode'] == 'CPU']
        
        if len(gpu_group) > 0 and len(cpu_group) > 0:
            gpu_rate = gpu_group['observed_write_rate'].mean()
            cpu_rate = cpu_group['observed_write_rate'].mean()
            
            # Calculate performance difference
            if cpu_rate > 0:  # Avoid division by zero
                perf_diff = (gpu_rate - cpu_rate) / cpu_rate
            else:
                perf_diff = 1.0 if gpu_rate > 0 else 0.0
            
            # Determine which is better (2=GPU, 1=EQUAL, 0=CPU)
            if perf_diff > equality_threshold:
                comparison = 2  # GPU is better
            elif perf_diff < -equality_threshold:
                comparison = 0  # CPU is better
            else:
                comparison = 1  # They are equal
            
            # Add to comparisons dataframe
            io_mode, io_threads, block_size, dataset_size = config
            comparisons.append({
                'io_mode': io_mode,
                'io_threads': io_threads, 
                'block_size': block_size,
                'dataset_size': dataset_size,
                'dataset_size_gb': float(dataset_size.replace('GB', '')),
                'io_mode_encoded': 1 if io_mode == 'ASYNC' else 0,
                'comparison': comparison,
                'gpu_rate': gpu_rate,
                'cpu_rate': cpu_rate,
                'perf_diff': perf_diff
            })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparisons)
    
    # Print data statistics
    print(f"\nComparison results:")
    print(f"GPU better: {len(comparison_df[comparison_df['comparison'] == 2])} cases")
    print(f"CPU better: {len(comparison_df[comparison_df['comparison'] == 0])} cases")
    print(f"Equal performance: {len(comparison_df[comparison_df['comparison'] == 1])} cases")
    
    # Prepare features for analysis
    X = comparison_df[['io_threads', 'block_size', 'dataset_size_gb', 'io_mode_encoded']]
    y = comparison_df['comparison']
    
    # Create a label encoder for io_mode
    label_encoder = LabelEncoder()
    label_encoder.fit(['SYNC', 'ASYNC'])
    
    return comparison_df, X, y, label_encoder

def create_decision_tree(X, y, label_encoder, csv_path):
    # Create a custom decision tree where each level uses a specific feature
    # We'll build the tree manually to follow a strict hierarchy
    
    # Determine the order of features based on their information gain
    temp_tree = DecisionTreeClassifier(max_depth=1)
    feature_importance = []
    
    # Test each feature individually to find its importance
    feature_names_internal = ['io_threads', 'block_size', 'dataset_size_gb', 'io_mode_encoded']
    display_names = ['I/O Threads', 'Block Size (bytes)', 'Dataset Size (GB)', 'I/O Mode']
    
    for i, feature in enumerate(feature_names_internal):
        temp_tree.fit(X[[feature]], y)
        feature_importance.append((i, temp_tree.tree_.impurity[0] - temp_tree.tree_.impurity.sum() * temp_tree.tree_.n_node_samples[1:].sum() / temp_tree.tree_.n_node_samples[0]))
    
    # Sort features by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    feature_order = [f[0] for f in feature_importance]
    
    print("\nFeature Importance Order:")
    for i, idx in enumerate(feature_order):
        print(f"Level {i+1}: {display_names[idx]} (Information Gain: {feature_importance[i][1]:.4f})")
    
    # Create a decision tree with the desired feature order
    # We use the max_features and max_depth parameters to control the order
    
    # Define the feature order (from most to least important)
    X_ordered = X.iloc[:, feature_order]
    ordered_feature_names = [feature_names_internal[i] for i in feature_order]
    
    # Create the tree with ordered features
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_ordered, y)
    
    # Create a figure for the tree
    plt.figure(figsize=(24, 12))
    
    # Order the display names according to the feature order
    ordered_display_names = [display_names[i] for i in feature_order]
    
    # Define class names and colors
    class_names = ['CPU Better', 'EQUAL', 'GPU Better']
    
    # Add text showing the CSV path being analyzed
    plt.text(0.01, 0.99, f"Based on: {csv_path}", 
             ha='left', va='top', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot the tree with custom styling
    plot_tree(
        tree, 
        feature_names=ordered_display_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2,
        proportion=True,
        impurity=True
    )
    
    # Add a legend
    import matplotlib.patches as mpatches
    cpu_patch = mpatches.Patch(color='#FAD7A0', label='CPU Better')
    equal_patch = mpatches.Patch(color='#D5F5E3', label='EQUAL')
    gpu_patch = mpatches.Patch(color='#A9CCE3', label='GPU Better')
    plt.legend(handles=[cpu_patch, equal_patch, gpu_patch], loc='lower right')
    
    # Determine output directory based on csv_path
    import os
    if '--data-path' in sys.argv:
        # If custom path, save in parent directory
        output_dir = os.path.dirname(csv_path)
    else:
        # If using latest RunN, save in that folder
        output_dir = os.path.dirname(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure in the determined directory
    output_file = os.path.join(output_dir, 'decision_tree.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Extract decision paths from the tree
    n_nodes = tree.tree_.node_count
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
    # Print the decision rules - focus on leaf nodes only
    print("\nDecision Rules (Leaf Node Decisions):")
    for i in range(n_nodes):
        if tree.tree_.children_left[i] == -1:  # leaf node
            # Determine the predicted class at this leaf
            class_value = np.argmax(tree.tree_.value[i][0])
            class_name = class_names[class_value]
            n_samples = tree.tree_.n_node_samples[i]
            
            # Get percentage of samples in this class
            samples_in_class = tree.tree_.value[i][0][class_value]
            purity = (samples_in_class / n_samples) * 100
            
            # Trace the path from the root to this leaf
            node_id = i
            rules = []
            while node_id != 0:
                # Find the parent node
                parent_found = False
                for j in range(n_nodes):
                    if tree.tree_.children_left[j] == node_id or tree.tree_.children_right[j] == node_id:
                        is_left = tree.tree_.children_left[j] == node_id
                        feature_idx = feature[j]
                        if feature_idx >= 0:  # Not a leaf node
                            feature_name = ordered_display_names[feature_idx]
                            threshold_value = threshold[j]
                            
                            if feature_name == "I/O Mode":
                                # For I/O Mode, show SYNC or ASYNC directly
                                mode_values = label_encoder.inverse_transform([0, 1])
                                if is_left:
                                    rules.append(f"{feature_name} = {mode_values[0]}")
                                else:
                                    rules.append(f"{feature_name} = {mode_values[1]}")
                            else:
                                if is_left:
                                    rules.append(f"{feature_name} <= {threshold_value:.2f}")
                                else:
                                    rules.append(f"{feature_name} > {threshold_value:.2f}")
                                    
                        node_id = j
                        parent_found = True
                        break
                if not parent_found:
                    break
            
            # Print the rules in reverse order (from root to leaf)
            rules.reverse()
            print(f"\nIF {' AND '.join(rules)} THEN {class_name} (samples: {n_samples}, purity: {purity:.1f}%)")
    
    # Print feature importance again
    print("\nFeature Importance in Decision Tree:")
    feature_importance_values = tree.feature_importances_
    ordered_importance = [(ordered_display_names[i], feature_importance_values[i]) for i in range(len(ordered_display_names))]
    for name, importance in sorted(ordered_importance, key=lambda x: x[1], reverse=True):
        print(f"{name}: {importance:.4f}")
    
    print(f"\nDecision tree saved to: {output_file}")
    
    return tree, feature_order

def analyze_thresholds(df, tree, feature_order):
    print("\n=== Performance Analysis ===")
    
    # Define display names for features
    display_names = ['I/O Threads', 'Block Size (bytes)', 'Dataset Size (GB)', 'I/O Mode']
    ordered_display_names = [display_names[i] for i in feature_order]
    
    # Get decision tree thresholds
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
    features = tree.tree_.feature[tree.tree_.feature != -2]
    
    print("\nDecision Tree Key Thresholds:")
    for i, (threshold, feature) in enumerate(zip(thresholds, features)):
        if feature >= 0:  # Not a leaf
            feature_name = ordered_display_names[feature]
            print(f"{feature_name}: {threshold:.2f}")
    
    # Map feature order to original column names
    feature_names_internal = ['io_threads', 'block_size', 'dataset_size_gb', 'io_mode_encoded']
    ordered_feature_names = [feature_names_internal[i] for i in feature_order]
    
    # Analyze by each feature
    for i, feature_idx in enumerate(feature_order):
        original_feature_name = feature_names_internal[feature_idx]
        
        # Handle io_mode_encoded specially
        if original_feature_name == 'io_mode_encoded':
            feature_name = 'io_mode'
        else:
            feature_name = original_feature_name
        
        print(f"\n{display_names[feature_idx].upper()} Analysis (Level {i+1}):")
        
        if feature_name == 'io_mode':
            # For io_mode we need to handle the categorical values
            values = df['io_mode'].unique()
        else:
            values = sorted(df[feature_name].unique())
        
        for value in values:
            subset = df[df[feature_name] == value]
            gpu_better = len(subset[subset['comparison'] == 2])
            equal = len(subset[subset['comparison'] == 1])
            cpu_better = len(subset[subset['comparison'] == 0])
            total = len(subset)
            
            if total > 0:
                gpu_pct = (gpu_better / total) * 100
                cpu_pct = (cpu_better / total) * 100
                equal_pct = (equal / total) * 100
                
                if feature_name == 'block_size':
                    value_str = f"{value/1024/1024:.2f} MB"
                else:
                    value_str = str(value)
                
                print(f"\n{display_names[feature_idx]}: {value_str}")
                print(f"GPU Better: {gpu_better} cases ({gpu_pct:.1f}%)")
                print(f"CPU Better: {cpu_better} cases ({cpu_pct:.1f}%)")
                print(f"Equal: {equal} cases ({equal_pct:.1f}%)")
                
                if 'gpu_rate' in df.columns and 'cpu_rate' in df.columns:
                    gpu_rate = subset['gpu_rate'].mean()
                    cpu_rate = subset['cpu_rate'].mean()
                    print(f"Avg GPU Rate: {gpu_rate:.2f} GB/s, Avg CPU Rate: {cpu_rate:.2f} GB/s")

def get_data_path():
    parser = argparse.ArgumentParser(description='Analyze H5bench performance data')
    parser.add_argument('--data-path', type=str, help='Path to raw_output.csv file')
    args = parser.parse_args()
    
    if args.data_path:
        return args.data_path
    
    # If no path provided, find the latest RunN directory
    import glob
    import os
    
    run_dirs = glob.glob('CSV/Run*')
    if not run_dirs:
        print("No RunN directories found!")
        sys.exit(1)
    
    latest_run = max(run_dirs, key=os.path.getctime)
    csv_path = os.path.join(latest_run, 'raw_output.csv')
    
    if not os.path.exists(csv_path):
        print(f"raw_output.csv not found in {latest_run}")
        sys.exit(1)
    
    return csv_path

def main():
    # Get the data path
    csv_path = get_data_path()
    print(f"Analyzing data from {csv_path}")
    
    # Load and analyze data
    df, X, y, label_encoder = load_and_prepare_data(csv_path)
    
    if df is None:
        print("Error: Failed to prepare data. Make sure the CSV file contains both GPU and CPU data.")
        sys.exit(1)
    
    # Create and analyze decision tree
    tree, feature_order = create_decision_tree(X, y, label_encoder, csv_path)
    
    # Analyze thresholds
    analyze_thresholds(df, tree, feature_order)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 