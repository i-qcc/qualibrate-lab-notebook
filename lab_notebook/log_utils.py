import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import dateutil.parser
from ast import literal_eval

def get_sorted_experiments(base_path):
    experiments = []
    base = Path(base_path)
    
    for project_folder in base.iterdir():
        if project_folder.is_dir():
            for date_folder in project_folder.iterdir():
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
                                
                                experiments.append(dict(state_file=state_file, wiring_file=wiring_file, node_file=node_file, timestamp=timestamp, id=id))                       
    
    experiments.sort(key=lambda x: x["id"])
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


def log_state_changes(log_file: str = "state_log.yml", path: str = "/home/omrieoqm/.qualibrate/user_storage", experiments: list[dict] = None):
    
    if experiments is None:
        experiments = get_sorted_experiments(path)
        
    all_changes = []
    
    prev_state, prev_wiring = None, None
    
    for exp in experiments:
        state_path = exp["state_file"]
        wiring_path = exp["wiring_file"]
        timestamp = exp["timestamp"]
        
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
    
    log_file = os.path.join(os.path.dirname(__file__), "state_log.yml")
    # log_state_changes(log_file=log_file)
    
    analyzer = ChangeAnalyzer(log_file, backend="qc_qwtune")
    analyzer.plot_changes("qubits.qA1.resonator.confusion_matrix") # , time_frame=(datetime.datetime(2025, 3, 5), datetime.datetime(2025, 3, 7)))
