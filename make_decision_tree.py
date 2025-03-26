#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import sys
from graphviz import Digraph

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index (0=io_threads, 1=block_size, etc.)
        self.threshold = threshold  # Split threshold
        self.left = left         # Left subtree (<= threshold)
        self.right = right       # Right subtree (> threshold)
        self.value = value       # Predicted class (0=CPU, 1=EQUAL, 2=GPU)

def entropy(y):
    """Calculate Shannon entropy of target values"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(X, y, feature, threshold):
    """Calculate information gain for a given feature and threshold"""
    parent_entropy = entropy(y)
    
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    n_left, n_right = sum(left_mask), sum(right_mask)
    if n_left == 0 or n_right == 0:
        return 0
    
    child_entropy = (n_left/len(y)) * entropy(y[left_mask]) + \
                    (n_right/len(y)) * entropy(y[right_mask])
    
    return parent_entropy - child_entropy

def find_best_split(X, y):
    """Find the best feature and threshold to split on"""
    best_gain = -1
    best_feature, best_threshold = None, None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = information_gain(X, y, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=5):
    """Recursively build decision tree with proper stopping criteria"""
    # Stopping conditions
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return DecisionNode(value=np.argmax(np.bincount(y)))
    
    # Find best split
    feature, threshold, gain = find_best_split(X, y)
    
    # Lower the information gain threshold to allow more splits
    if gain < 0.001:  # Much smaller minimum gain threshold
        return DecisionNode(value=np.argmax(np.bincount(y)))
    
    # Split data
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    # Don't split if either branch would be empty
    if sum(left_mask) == 0 or sum(right_mask) == 0:
        return DecisionNode(value=np.argmax(np.bincount(y)))
    
    # Recursively build subtrees
    left = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    
    return DecisionNode(feature=feature, threshold=threshold, left=left, right=right)

def visualize_tree(tree, feature_names, class_names, filename):
    """Create Graphviz visualization of decision tree"""
    from graphviz import Digraph
    
    dot = Digraph()
    nodes = [(tree, 0)]
    node_id = 1
    
    while nodes:
        node, parent_id = nodes.pop()
        
        if node.value is not None:
            # Use only two colors: red for CPU, blue for GPU
            color = '#FFA07A' if node.value == 0 else '#87CEEB'
            label = f"{class_names[node.value]}"
            dot.node(str(parent_id), label, style='filled', color=color, fontcolor='black')
        else:
            label = f"{feature_names[node.feature]}\n<= {node.threshold:.2f}"
            dot.node(str(parent_id), label, shape='box')
            
            # Add left and right children
            dot.edge(str(parent_id), str(node_id))
            nodes.append((node.left, node_id))
            node_id += 1
            
            dot.edge(str(parent_id), str(node_id))
            nodes.append((node.right, node_id))
            node_id += 1
    
    try:
        dot.render(filename, view=False, cleanup=True)
        print(f"Decision tree visualization saved to {filename}.pdf")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("To create visualizations, please install Graphviz:")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  CentOS/RHEL: sudo yum install graphviz")
        print("  macOS: brew install graphviz")
        print("  Windows: Download from https://graphviz.org/download/")
    
    return dot

def extract_rules(node, feature_names, class_names, path=None):
    """Extract decision rules from tree"""
    if path is None:
        path = []
    
    if node.value is not None:
        return [{"path": path.copy(), "decision": node.value, "class": class_names[node.value]}]
    
    rules = []
    # Left branch
    path.append(f"{feature_names[node.feature]} <= {node.threshold:.2f}")
    rules.extend(extract_rules(node.left, feature_names, class_names, path))
    path.pop()
    
    # Right branch
    path.append(f"{feature_names[node.feature]} > {node.threshold:.2f}")
    rules.extend(extract_rules(node.right, feature_names, class_names, path))
    path.pop()
    
    return rules

def clean_write_rate(rate_str):
    """Clean write rate string to extract numeric value in GB/s"""
    try:
        # Remove whitespace and convert to lowercase for consistent comparison
        rate_str = str(rate_str).strip().lower()
        
        # If already a number, return it
        try:
            return float(rate_str)
        except ValueError:
            pass
        
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
    except Exception as e:
        print(f"Error cleaning rate: {rate_str}, error: {e}")
        return 0.0

def load_data(csv_path):
    """Load and preprocess benchmark data"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, comment='#')  # Explicitly handle lines starting with # as comments
    
    # Convert dataset_size to numeric (GB)
    df['dataset_size_gb'] = df['dataset_size'].str.replace('GB', '').astype(float)
    
    # Clean write rates
    df['observed_write_rate'] = df['observed_write_rate'].apply(clean_write_rate)
    
    # Check if we have both GPU and CPU data
    modes_in_data = df['mode'].unique()
    gpu_mode = 'GPU' if 'GPU' in modes_in_data else 'GDS'
    cpu_mode = 'CPU'
    
    if not all(mode in modes_in_data for mode in [gpu_mode, cpu_mode]):
        print(f"Warning: Dataset does not contain both {cpu_mode} and {gpu_mode} data")
        return None, None, None
    
    # Compare GPU vs CPU for each configuration
    comparisons = []
    for (io_threads, block_size, dataset_size), group in df.groupby(['io_threads', 'block_size', 'dataset_size']):
        gpu_group = group[group['mode'] == gpu_mode]
        cpu_group = group[group['mode'] == cpu_mode]
        
        if len(gpu_group) == 0 or len(cpu_group) == 0:
            continue
            
        gpu_rate = gpu_group['observed_write_rate'].mean()
        cpu_rate = cpu_group['observed_write_rate'].mean()
        
        # Simple binary comparison - is GPU better than CPU?
        # 1 = GPU better, 0 = CPU better
        comparison = 1 if gpu_rate > cpu_rate else 0
        
        # Calculate performance difference (just for reporting)
        if cpu_rate > 0:
            perf_diff = (gpu_rate - cpu_rate) / cpu_rate
        else:
            perf_diff = 1.0 if gpu_rate > 0 else 0.0
            
        comparisons.append({
            'io_threads': io_threads,
            'block_size': block_size,
            'dataset_size_gb': float(dataset_size.replace('GB', '')),
            'target': comparison,
            'gpu_rate': gpu_rate,
            'cpu_rate': cpu_rate,
            'perf_diff': perf_diff
        })
    
    comparison_df = pd.DataFrame(comparisons)
    
    print(f"Processed {len(comparison_df)} configurations:")
    print(f"GPU better: {len(comparison_df[comparison_df['target'] == 1])} cases")
    print(f"CPU better: {len(comparison_df[comparison_df['target'] == 0])} cases")
    
    X = comparison_df[['io_threads', 'block_size', 'dataset_size_gb']].values
    y = comparison_df['target'].values
    
    return X, y, comparison_df

def get_data_path():
    parser = argparse.ArgumentParser(description='Build decision tree from benchmark data')
    parser.add_argument('--data-path', type=str, help='Path to raw_output.csv file')
    parser.add_argument('--output-dir', type=str, help='Directory to save the decision tree')
    args = parser.parse_args()
    
    if args.data_path:
        data_path = args.data_path
        output_dir = args.output_dir or os.path.dirname(args.data_path)
    else:
        # If no path provided, find the latest RunN directory
        import glob
        run_dirs = glob.glob('CSV/Run*')
        if not run_dirs:
            print("No RunN directories found!")
            sys.exit(1)
        
        latest_run = max(run_dirs, key=os.path.getctime)
        data_path = os.path.join(latest_run, 'raw_output.csv')
        output_dir = latest_run
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        sys.exit(1)
    
    return data_path, output_dir

def generate_decision_tree(df_or_path, output_dir=None):
    """
    Generate a decision tree from benchmark data.
    
    Parameters:
    -----------
    df_or_path : pandas.DataFrame or str
        Either a DataFrame containing the benchmark data or a path to a CSV file.
    output_dir : str, optional
        Directory to save the decision tree files. If None and df_or_path is a string,
        it will use the parent directory of the CSV file.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the comparison results.
    """
    print("\n===== Generating Decision Tree =====")
    
    # Handle input: either a DataFrame or a path to a CSV
    if isinstance(df_or_path, str):
        csv_path = df_or_path
        # If output_dir not specified, use the parent directory of the CSV
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        # Load and prepare data
        X, y, comparison_df = load_data(csv_path)
        if X is None:
            print("Error: Failed to prepare data")
            return None
    else:
        # Input is already a DataFrame
        df = df_or_path
        
        # If no output directory provided, try to infer it
        if output_dir is None:
            # Try to find a RunN directory in the current path
            import glob
            run_dirs = glob.glob('CSV/Run*')
            if run_dirs:
                output_dir = max(run_dirs, key=os.path.getctime)
            else:
                output_dir = '.'
        
        # Ensure output_dir is a Path object if DataFrame was provided
        output_dir = os.path.abspath(output_dir)
        
        # Process data from the provided DataFrame
        modes_in_data = df['mode'].unique()
        gpu_mode = 'GPU' if 'GPU' in modes_in_data else 'GDS'
        cpu_mode = 'CPU'
        
        # Prepare data for decision tree
        comparisons = []
        
        # Group by configuration
        for (io_threads, block_size, dataset_size), group in df.groupby(['io_threads', 'block_size', 'dataset_size']):
            gpu_group = group[group['mode'] == gpu_mode]
            cpu_group = group[group['mode'] == cpu_mode]
            
            if len(gpu_group) == 0 or len(cpu_group) == 0:
                continue
                
            gpu_rate = gpu_group['observed_write_rate'].mean()
            cpu_rate = cpu_group['observed_write_rate'].mean()
            
            # Simple binary comparison - is GPU better than CPU?
            # 1 = GPU better, 0 = CPU better
            comparison = 1 if gpu_rate > cpu_rate else 0
            
            # Calculate performance difference (for reporting)
            if cpu_rate > 0:
                perf_diff = (gpu_rate - cpu_rate) / cpu_rate
            else:
                perf_diff = 1.0 if gpu_rate > 0 else 0.0
                
            # Convert string dataset size to numeric
            dataset_size_gb = float(dataset_size.replace('GB', ''))
                
            comparisons.append({
                'io_threads': io_threads,
                'block_size': block_size,
                'dataset_size_gb': dataset_size_gb,
                'target': comparison,
                'gpu_rate': gpu_rate,
                'cpu_rate': cpu_rate,
                'perf_diff': perf_diff
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparisons)
        
        X = comparison_df[['io_threads', 'block_size', 'dataset_size_gb']].values
        y = comparison_df['target'].values
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Display summary
    print(f"Processed {len(comparison_df)} configurations:")
    print(f"GPU better: {len(comparison_df[comparison_df['target'] == 1])} cases")
    print(f"CPU better: {len(comparison_df[comparison_df['target'] == 0])} cases")
    
    # Define feature and class names
    feature_names = ['I/O Threads', 'Block Size (bytes)', 'Dataset Size (GB)']
    class_names = ['CPU Better', 'GPU Better']
    
    # Build decision tree
    print("Building decision tree...")
    tree = build_tree(X, y, max_depth=4)
    
    # Visualize and save decision tree
    print("Generating visualization...")
    output_path = os.path.join(output_dir, 'decision_tree')
    visualize_tree(tree, feature_names, class_names, output_path)
    
    # Extract and save decision rules
    rules = extract_rules(tree, feature_names, class_names)
    
    with open(os.path.join(output_dir, 'decision_rules.txt'), 'w') as f:
        f.write(f"Decision Tree Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Summary:\n")
        f.write(f"Total configurations analyzed: {len(comparison_df)}\n")
        f.write(f"GPU better: {len(comparison_df[comparison_df['target'] == 1])} cases\n")
        f.write(f"CPU better: {len(comparison_df[comparison_df['target'] == 0])} cases\n\n")
        
        f.write("Decision Rules:\n")
        f.write("----------------\n")
        for i, rule in enumerate(rules, 1):
            path_str = " AND ".join(rule["path"])
            f.write(f"Rule {i}: IF {path_str} THEN {rule['class']}\n")
    
    print(f"Decision rules saved to {os.path.join(output_dir, 'decision_rules.txt')}")
    return comparison_df

def main():
    # Get data path and output directory
    data_path, output_dir = get_data_path()
    
    # Generate the decision tree
    generate_decision_tree(data_path, output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 