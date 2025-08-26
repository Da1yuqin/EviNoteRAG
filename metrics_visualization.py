import json
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd


class MetricsVisualizer:
    """
    Loads metrics data from a JSONL file associated with a project/experiment
    and provides methods for visualizing various metrics.
    """
    def __init__(self, file_path):
        """
        Args:
            project_name (str): Name of the project.
            experiment_name (str): Name of the experiment.
            base_dir (str, optional): Base directory containing metric files.
                                      Defaults to "/ossfs/workspace/FactAgent/DeepResearcher/val_metrics".
        """
        self.file_path = file_path
        self.data = None
        self.dataframe = None
        self._load_data()
    
    def _load_data(self):
        """
        Private method to load data from the JSONL file specified by self.file_path.
        Populates self.data and self.dataframe.
        """

        raw_data_list = []
        print(f"Attempting to load data from: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            # Flatten the structure slightly for easier DataFrame creation
                            # Combine global_step with the nested metrics dictionary
                            flat_record = {'global_step': record.get('global_step')}
                            if 'metrics' in record and isinstance(record['metrics'], dict):
                                flat_record.update(record['metrics']) # Add all keys from metrics dict
                            raw_data_list.append(flat_record)
                        except json.JSONDecodeError as json_err:
                            print(f"  Warning: Skipping invalid JSON on line {line_number}. Error: {json_err}")
                        except Exception as line_err:
                             print(f"  Warning: Error processing line {line_number}. Error: {line_err}")

            self.data = raw_data_list # Store the raw list of dicts
            # Convert to pandas DataFrame if data was loaded
            if self.data:
                self.dataframe = pd.DataFrame(self.data)
                print(f"Successfully loaded {len(self.data)} records.")
            else:
                 print("No data loaded, DataFrame not created.")


        except FileNotFoundError:
            print(f"  ERROR: File not found '{self.file_path}'")
            self.data = [] # Ensure data is an empty list on error
            self.dataframe = pd.DataFrame() # Ensure dataframe is empty
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred while loading data: {e}")
            self.data = []
            self.dataframe = pd.DataFrame()            

    def is_data_loaded(self):
        """Checks if data was successfully loaded."""
        return self.dataframe is not None and not self.dataframe.empty

    def list_available_metrics(self):
        """Returns a list of metric keys available in the loaded data (excluding global_step)."""
        if self.is_data_loaded():
            return [col for col in self.dataframe.columns if col != 'global_step']
        return []
    
    def primary_metrics_visualization(self):   
        if not self.is_data_loaded():
            print("Error: No data loaded, cannot plot.")
            return
        
        """Visualizing sequence length evolving over training steps"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe['global_seqlen/min'], color='blue', label='seqlen/min')
        plt.plot(self.dataframe['global_seqlen/max'], color='red', label='seqlen/max')
        plt.plot(self.dataframe['global_seqlen/mean'], color='green', label='seqlen/mean')
        plt.xlabel("Training Steps")
        plt.ylabel("Sequence Length")
        plt.title("Sequence Length Evolving Over Training Steps")
        plt.legend()
        plt.savefig('./outputs/metrics_val/seqlen_over_steps.png')
        plt.close()

        """Visualizing response length evolving over training steps"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe['response_length/min'], color='blue', label='resplen/min')
        plt.plot(self.dataframe['response_length/max'], color='red', label='resplen/max')
        plt.plot(self.dataframe['response_length/mean'], color='green', label='resplen/mean')
        plt.xlabel("Training Steps")
        plt.ylabel("Response Length")
        plt.title("Response Length Evolving Over Training Steps")
        plt.legend()
        plt.savefig('./outputs/metrics_val/resplen_over_steps.png')
        plt.close()

        """Visualizing loss evolving over training steps"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe['actor/entropy_loss'], color='blue', label='actor/entropy_loss')
        plt.plot(self.dataframe['actor/pg_loss'], color='red', label='actor/pg_loss')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Loss Evolving Over Training Steps")
        plt.legend()
        plt.savefig('./outputs/metrics_val/loss_over_steps.png')
        plt.close()

        """Visualizing EM score evolving over training steps"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dataframe['val/test_score/2wikimultihopqa_val'], label='2wikimultihopqa')
        plt.plot(self.dataframe['val/test_score/bamboogle_val'], label='bamboogle')
        plt.plot(self.dataframe['val/test_score/hotpotqa_val'], label='hotpotqa')
        plt.plot(self.dataframe['val/test_score/musique_val'], label='musique')
        plt.plot(self.dataframe['val/test_score/nq_val'], label='nq')
        plt.plot(self.dataframe['val/test_score/popqa_val'], label='popqa')
        plt.plot(self.dataframe['val/test_score/triviaqa_val'], label='triviaqa')
        plt.xlabel("Training Steps")
        plt.ylabel("EM score")
        plt.title("EM score Evolving Over Training Steps")
        plt.legend()
        plt.savefig('./outputs/metrics_val/EM_over_steps.png')
        plt.close()

 
def main():
    """ Parses arguments """
    parser = argparse.ArgumentParser(description="Metrics Visualization")
    parser.add_argument('--file_path', default = "./outputs/EviNoteRAG/local/EviNoteRAG.jsonl", type=str)

    os.makedirs('./outputs/metrics_val', exist_ok=True)
    args = parser.parse_args()

    # Create an instance of the visualizer
    visualizer = MetricsVisualizer(args.file_path)

    # Check if data loaded successfully
    if not visualizer.is_data_loaded():
        print("Error: No data loaded.")
        return
    
    visualizer.primary_metrics_visualization()

if __name__ == "__main__":
    main()    