import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import dateutil.parser
from ast import literal_eval

# def get_sorted_experiments(base_path, from_timestamp=None):
#     experiments = []
#     base = Path(base_path)
    
#     for project_folder in base.iterdir():
#         if project_folder.is_dir():
#             # Get all date folders and sort them by creation time
#             date_folders = []
#             for date_folder in project_folder.iterdir():
#                 if date_folder.is_dir():
#                     # Get creation time of the date folder as Unix timestamp
#                     creation_time = int(date_folder.stat().st_mtime)
#                     date_folders.append((date_folder, creation_time))
            
#             # Sort date folders by creation time in descending order (newest first)
#             date_folders.sort(key=lambda x: x[1], reverse=True)
            
#             # Process date folders
#             for date_folder, creation_time in date_folders:
#                 # If we have a from_timestamp and this folder is older, skip it and all older folders
#                 if from_timestamp is not None and creation_time < from_timestamp:
#                     break
                    
#                 for exp_folder in date_folder.iterdir():
#                     if exp_folder.is_dir():
#                         node_file = exp_folder / "node.json"
#                         state_file = exp_folder / "quam_state" / "state.json"
#                         wiring_file = exp_folder / "quam_state" / "wiring.json"
#                         if state_file.exists() and wiring_file.exists() and node_file.exists():
#                             metadata = load_json(node_file)
        
#                             if "created_at" not in metadata:
#                                 raise ValueError(f"Missing 'created_at' in {node_file}")
                            
#                             # Parse timestamp and ensure it's a Unix timestamp
#                             timestamp = parse_timestamp(metadata["created_at"])
#                             id = metadata["id"]
                            
#                             experiments.append(dict(state_file=state_file, wiring_file=wiring_file, node_file=node_file, timestamp=timestamp, id=id))                       
    
#     experiments.sort(key=lambda x: x["id"])
#     return experiments

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_timestamp(timestamp_str):
    """Convert ISO format timestamp string to Unix timestamp (seconds since epoch)"""
    dt = dateutil.parser.isoparse(timestamp_str)
    return int(dt.timestamp())

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_dicts(dict1: dict, dict2: dict, timestamp: int, backend: str):
    changes = []
    flat1 = flatten_dict(dict1)
    flat2 = flatten_dict(dict2)
    all_keys = set(flat1.keys()).union(set(flat2.keys()))
    
    for key in all_keys:
        if flat1.get(key) != flat2.get(key):
            changes.append(f"{backend} - {key} : {flat1.get(key)} -> {flat2.get(key)} at {timestamp}")
    
    return changes

class ChangeAnalyzer:
    def __init__(self, log_file: str, backend: str):
        self.log_file = log_file
        self.backend = backend
        
    def read_changes(self, target_key):
        timestamps = []
        values = []
        with open(self.log_file, 'r') as f:
            for line in f:
                backend, other_parts = line.strip().split(" - ")
                if backend == self.backend:
                    parts = other_parts.strip().split(" at ")
                    if len(parts) == 2 and target_key in parts[0]:
                        key_value, timestamp = parts
                        old_value, new_value = key_value.split(" : ")[1].split(" -> ")
                        if new_value != "None":
                            try:
                                timestamps.append(int(timestamp))
                                if new_value.startswith('[') and new_value.endswith(']'):
                                    value = literal_eval(new_value)
                                    values.append(value)
                                else:
                                    values.append(float(new_value))
                            except (ValueError, SyntaxError) as e:
                                raise ValueError(f"Could not parse value as list: {new_value}") from e
        return timestamps, values
    
    def plot_changes(self, target_key, time_frame=None):
        timestamps, values = self.read_changes(target_key)
        if not timestamps:
            print("No changes found for key:", target_key)
            return
        
        human_times = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        if time_frame:
            start_time, end_time = time_frame
            human_times, values = zip(*[(t, v) for t, v in zip(human_times, values) if start_time <= t <= end_time])
        
        plt.figure(figsize=(10, 5))
        
        # Special handling for confusion matrix plots
        if target_key.endswith('.confusion_matrix'):
            # Extract 00 and 11 elements from each 2x2 matrix
            fidelity_00 = [matrix[0][0] for matrix in values]
            fidelity_11 = [matrix[1][1] for matrix in values]
            
            plt.plot(human_times, fidelity_00, marker='o', linestyle='-', label='00 readout fidelity')
            plt.plot(human_times, fidelity_11, marker='o', linestyle='-', label='11 readout fidelity')
            plt.ylabel("Readout Fidelity")
            plt.legend()
        else:
            plt.plot(human_times, values, marker='o', linestyle='-')
            plt.ylabel(target_key)
            
        plt.xlabel("Time")
        plt.title(f"{self.backend} - Changes in {target_key} over time")
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    
    log_file = os.path.join(os.path.dirname(__file__), "state_log.yml")
    # log_state_changes(log_file=log_file)
    
    analyzer = ChangeAnalyzer(log_file, backend="qc_qwtune")
    analyzer.plot_changes("qubits.qA1.resonator.confusion_matrix") # , time_frame=(datetime.datetime(2025, 3, 5), datetime.datetime(2025, 3, 7)))
