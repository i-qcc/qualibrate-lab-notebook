import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import dateutil.parser

def get_sorted_experiments(base_path):
    experiments = []
    base = Path(base_path)
    
    for date_folder in base.iterdir():
        if date_folder.is_dir():
            for exp_folder in date_folder.iterdir():
                if exp_folder.is_dir():
                    node_file = exp_folder / "node.json"
                    state_file = exp_folder / "quam_state" / "state.json"
                    wiring_file = exp_folder / "quam_state" / "wiring.json"
                    if state_file.exists() and wiring_file.exists() and node_file.exists():
                        
                        metadata = load_json(node_file)
        
                        if "created_at" not in metadata:
                            raise ValueError(f"Missing 'created_at' in {node_file}")
                        
                        timestamp = parse_timestamp(metadata["created_at"])
                        id = metadata["id"]
                        
                        experiments.append((state_file, wiring_file, timestamp, id))
    
    experiments.sort(key=lambda x: x[-1])  # Sort by date
    return experiments

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_timestamp(timestamp_str):
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

def log_changes(log_path, changes):
    with open(log_path, 'w') as f:
        f.write("\n".join(changes) + "\n")

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
                            timestamps.append(int(timestamp))
                            values.append(float(new_value))
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
        plt.plot(human_times, values, marker='o', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel(target_key)
        plt.title(f"{self.backend} - Changes in {target_key} over time")
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()

def log_state_changes(log_file: str = "changes_log.yml"):
    base_path = r"/home/omrieoqm/.qualibrate/user_storage/QC1"  # Adjust path
    
    experiments = get_sorted_experiments(base_path)
    all_changes = []
    
    prev_state, prev_wiring = None, None
    
    for state_path, wiring_path, timestamp, _ in experiments:
        curr_state = load_json(state_path)
        curr_wiring = load_json(wiring_path)
        
        backend = curr_wiring["network"].get("quantum_computer_backend", "unspecified_backend")
        
        if prev_state is not None:
            state_changes = compare_dicts(prev_state, curr_state, timestamp, backend)
            wiring_changes = compare_dicts(prev_wiring, curr_wiring, timestamp, backend)
            
            all_changes.extend(state_changes + wiring_changes)

        prev_state, prev_wiring = curr_state, curr_wiring
    
    
    log_changes(log_file, all_changes)
    print(f"Changes logged to {log_file}")

if __name__ == "__main__":
    
    log_file = os.path.join(os.path.dirname(__file__), "changes_log.yml")
    # log_state_changes(log_file=log_file)
    
    analyzer = ChangeAnalyzer(log_file, backend="qc_qwtune")
    analyzer.plot_changes("qubits.qA5.T2ramsey") # , time_frame=(datetime.datetime(2025, 3, 5), datetime.datetime(2025, 3, 7)))
