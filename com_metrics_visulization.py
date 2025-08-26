import json
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors

class MultiMetricsVisualizer:
    """
    Load metric data from multiple JSONL files in a specified directory and provide visualization methods.
    Supports plotting data from different files in different colors on the same figure.
    """
    def __init__(self, directory_path, metric_name):
        """
        Args:
            directory_path (str): Path to the directory containing JSONL files
            metric_name (str): Name of the metric to visualize (e.g., 'val/test_score/nq_val_em')
        """
        self.directory_path = directory_path
        self.metric_name = metric_name
        self.dataframes = {}  # Stores the dataframe of each file, keyed by file name
        self.file_paths = []  # Stores paths to found JSONL files
        self._find_jsonl_files()
        self._load_all_data()
    
    def _find_jsonl_files(self):
        """Find all JSONL files in the directory."""
        try:
            dir_path = Path(self.directory_path)
            if not dir_path.is_dir():
                print(f"Error: '{self.directory_path}' is not a directory")
                return
            
            # Recursively search for all .jsonl files
            self.file_paths = list(dir_path.rglob("*.jsonl"))
            
            if not self.file_paths:
                print(f"No JSONL files found in directory '{self.directory_path}'")
            else:
                print(f"Found {len(self.file_paths)} JSONL files:")
                for i, path in enumerate(self.file_paths, 1):
                    print(f"{i}. {path}")
        except Exception as e:
            print(f"Error finding files: {e}")
    
    def _load_data_from_file(self, file_path):
        """Load data from a single JSONL file."""
        raw_data_list = []
        line_number = 0
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        flat_record = {'global_step': record.get('global_step')}
                        
                        # Process the metrics dictionary
                        if 'metrics' in record and isinstance(record['metrics'], dict):
                            flat_record.update(record['metrics'])
                        
                        raw_data_list.append(flat_record)
                    except json.JSONDecodeError as json_err:
                        print(f"  Warning: Skipping invalid JSON on line {line_number} of {file_path.name}: {json_err}")
                    except Exception as line_err:
                        print(f"  Warning: Error processing line {line_number} of {file_path.name}: {line_err}")
            
            if raw_data_list:
                df = pd.DataFrame(raw_data_list)
                df['file_name'] = file_path.stem  # Add file name as a column
                return df
            else:
                print(f"  Warning: {file_path.name} has no valid data")
                return None
                
        except FileNotFoundError:
            print(f"  Error: File not found '{file_path}'")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
        return None
    
    def _load_all_data(self):
        """Load data from all found JSONL files."""
        if not self.file_paths:
            print("No files to load")
            return
            
        for file_path in self.file_paths:
            print(f"\nLoading file: {file_path.name}")
            df = self._load_data_from_file(file_path)
            if df is not None and not df.empty:
                self.dataframes[file_path.stem] = df
                print(f"  Successfully loaded {len(df)} records")
        
        print(f"\nLoaded data from {len(self.dataframes)} files in total")
    
    def has_data(self):
        """Check if any data has been successfully loaded."""
        return bool(self.dataframes)
    
    def plot_metric_comparison(self, output_dir=None):
        """
        Plot the specified metric from all files on the same figure using different colors.
        
        Args:
            output_dir (str, optional): Directory to save the output image. If None, does not save.
        """
        if not self.has_data():
            print("Error: No data to plot")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Generate a unique sequence of colors
        colors = self._generate_colors(len(self.dataframes))
        
        for (file_name, df), color in zip(self.dataframes.items(), colors):
            # Ensure data is sorted by global_step
            df = df.sort_values('global_step')
            
            # Check if the metric exists
            if self.metric_name not in df.columns:
                print(f"  警告: 文件 '{file_name}' 中不存在指标 '{self.metric_name}'")
                continue
                
            # Plot the metric curve
            plt.plot(
                df['global_step'], 
                df[self.metric_name], 
                label=file_name, 
                color=color,
                linewidth=2.5
            )
        
        # Set chart properties
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel(self.metric_name, fontsize=12)
        
        title = f"Comparison of '{self.metric_name}' Metric Across Multiple Files"
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save or show the image
        if output_dir:
            output_path = Path(output_dir) / f"{self.metric_name.replace('/', '_')}_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nChart saved to: {output_path}")
        
        plt.show()
        plt.close()
    
    def _generate_colors(self, num_colors):
        """Generate a set of unique colors."""
        # Use matplotlib's tab10 color palette as the base
        base_colors = plt.cm.tab10(np.linspace(0, 1, min(num_colors, 10)))
        
        # If more colors are needed, extend with hsv palette
        if num_colors > 10:
            additional_colors = plt.cm.hsv(np.linspace(0, 1, num_colors - 10))
            return np.vstack([base_colors, additional_colors])
        
        return base_colors


def main():
    """Parse arguments and run visualization"""
    parser = argparse.ArgumentParser(description="Metric Comparison Visualization Across Multiple Files")
    parser.add_argument('--directory', default = "./outputs/_final_results_nqp600/todraw/",
                        help="Path to the directory containing JSONL files")
    parser.add_argument('--metric',  default = "val/test_score/nq_val_em",
                        help="Name of the metric to compare (e.g., 'val/test_score/nq_val_em')")
    parser.add_argument('--output', default="./outputs/_final_results_nqp600/nq600_vis_com_metrics",
                        elp="Directory to save output image (optional)")
    
    args = parser.parse_args()
    
    # Create a visualizer instance
    visualizer = MultiMetricsVisualizer(args.directory, args.metric)
    
    # Check if data loaded successfully
    if not visualizer.has_data():
        print("错误: 没有加载到有效数据")
        return
    
    # Plot metric comparison chart
    visualizer.plot_metric_comparison(args.output)


if __name__ == "__main__":
    main()
