import os
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, send_file, abort, request
from dateutil import parser
from collections import defaultdict
import glob
import ast
import argparse
from .log_utils import ChangeAnalyzer, log_state_changes
import numpy as np
from functools import lru_cache
import time

app = Flask(__name__)

# Default configuration
DEFAULT_LAB_DATA_PATH = "/home/omrieoqm/.qualibrate/user_storage"
LAB_DATA_PATH = DEFAULT_LAB_DATA_PATH
STATE_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_log.yml")

# Cache for experiment data
_experiment_cache = {
    'data': None,
    'timestamp': 0,
    'is_valid': True  # Flag to indicate if cache is valid
}

def set_lab_data_path(path):
    """Set the lab data path and ensure it exists"""
    global LAB_DATA_PATH
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    LAB_DATA_PATH = path

def get_latest_logged_timestamp():
    """Get the timestamp of the latest entry in the state log"""
    if not os.path.exists(STATE_LOG_FILE):
        return 0
    
    latest_timestamp = 0
    with open(STATE_LOG_FILE, 'r') as f:
        for line in f:
            try:
                timestamp = int(line.split(" at ")[-1].strip())
                latest_timestamp = max(latest_timestamp, timestamp)
            except (ValueError, IndexError):
                continue
    
    return latest_timestamp

def update_state_log_incremental():
    """Update the state log with only new experiments since last update"""
    latest_timestamp = get_latest_logged_timestamp()
    
    # Get all folders in the lab data path
    all_folders = sorted([f for f in os.listdir(LAB_DATA_PATH) 
                         if os.path.isdir(os.path.join(LAB_DATA_PATH, f))])
    
    for folder in all_folders:
        folder_path = os.path.join(LAB_DATA_PATH, folder)
        
        # Get all date folders
        date_folders = sorted([f for f in os.listdir(folder_path) 
                             if os.path.isdir(os.path.join(folder_path, f)) 
                             and f.startswith('2025')])
        
        for date_folder in date_folders:
            date_path = os.path.join(folder_path, date_folder)
            
            # Get all experiment folders for this date
            exp_folders = sorted([f for f in os.listdir(date_path) 
                                if os.path.isdir(os.path.join(date_path, f))])
            
            for exp_folder in exp_folders:
                exp_path = os.path.join(date_path, exp_folder)
                node_file = os.path.join(exp_path, 'node.json')
                
                if not os.path.exists(node_file):
                    continue
                    
                try:
                    with open(node_file, 'r') as f:
                        node_data = json.load(f)
                        if 'created_at' not in node_data:
                            continue
                        
                        exp_timestamp = int(parser.parse(node_data['created_at']).timestamp())
                        
                        # Skip if this experiment is already in the log
                        if exp_timestamp <= latest_timestamp:
                            continue
                        
                        # This is a new experiment, update the log
                        log_state_changes(STATE_LOG_FILE)
                        return  # Exit after finding and logging the first new experiment
                        
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

def get_state_keys(backend_name, filter_text=None):
    """Get all unique state keys from the log file, filtered by prefix and optional text"""
    if not os.path.exists(STATE_LOG_FILE):
        return []
    
    keys = set()
    with open(STATE_LOG_FILE, 'r') as f:
        for line in f:
            if line.startswith(f"{backend_name} -"):
                parts = line.split(" - ")
                if len(parts) == 2:
                    key = parts[1].split(" : ")[0]
                    # Only include keys that start with qubits or qubit_pairs
                    if key.startswith(('qubits.', 'qubit_pairs.')):
                        # Apply text filter if provided
                        if filter_text is None or filter_text.lower() in key.lower():
                            keys.add(key)
    
    return sorted(list(keys))

def get_state_data(backend_name, key, from_date=None, to_date=None):
    """Get state data for a specific key and time range"""
    if not os.path.exists(STATE_LOG_FILE):
        return {"error": "No state data available"}
    
    analyzer = ChangeAnalyzer(STATE_LOG_FILE, backend_name)
    timestamps, values = analyzer.read_changes(key)
    
    if not timestamps:
        return {"error": "No data available for the specified parameters"}
    
    # Convert timestamps to datetime objects
    timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Filter by date range if specified
    if from_date:
        from_date = parser.parse(from_date)
        timestamps, values = zip(*[(t, v) for t, v in zip(timestamps, values) if t >= from_date])
    
    if to_date:
        to_date = parser.parse(to_date)
        timestamps, values = zip(*[(t, v) for t, v in zip(timestamps, values) if t <= to_date])
    
    return {
        "timestamps": [t.isoformat() for t in timestamps],
        "values": values
    }

def dict_to_hashable(d):
    """Convert a dictionary to a hashable format"""
    if isinstance(d, dict):
        return tuple(sorted((k, dict_to_hashable(v)) for k, v in d.items()))
    elif isinstance(d, list):
        return tuple(dict_to_hashable(item) for item in d)
    elif isinstance(d, float) and (d == float('inf') or d == float('-inf')):
        return str(d)  # Convert inf/-inf to string representation
    return d

def hashable_to_dict(h):
    """Convert a hashable format back to a dictionary"""
    if isinstance(h, tuple):
        if len(h) == 2 and isinstance(h[0], str):  # Key-value pair
            return {h[0]: hashable_to_dict(h[1])}
        return [hashable_to_dict(item) for item in h]
    elif isinstance(h, str):
        if h == 'inf':
            return float('inf')
        elif h == '-inf':
            return float('-inf')
        try:
            return float(h)
        except ValueError:
            return h
    return h

@lru_cache(maxsize=32)
def get_state_changes_cached(current_state_hash, previous_state_hash):
    """Cached version of state changes comparison"""
    changes = []
    
    def compare_states(current, previous, path=""):
        if isinstance(current, dict) and isinstance(previous, dict):
            for key in set(current.keys()) | set(previous.keys()):
                current_val = current.get(key)
                previous_val = previous.get(key)
                new_path = f"{path}.{key}" if path else key
                compare_states(current_val, previous_val, new_path)
        elif current != previous:
            changes.append(f"{path} : {previous} -> {current}")
    
    # Convert hashable format back to dictionaries for comparison
    current_state = hashable_to_dict(current_state_hash)
    previous_state = hashable_to_dict(previous_state_hash)
    
    compare_states(current_state, previous_state)
    return changes

def get_state_changes(current_state, previous_state):
    """Compare two state dictionaries and return a list of changes in the specified format"""
    # Convert dictionaries to hashable format
    current_state_hash = dict_to_hashable(current_state)
    previous_state_hash = dict_to_hashable(previous_state)
    
    return get_state_changes_cached(current_state_hash, previous_state_hash)

def clear_experiment_cache():
    """Clear the experiment data cache"""
    _experiment_cache['data'] = None
    _experiment_cache['timestamp'] = 0
    _experiment_cache['is_valid'] = False

def get_experiment_data(force_refresh=False):
    """Get all experiment data organized by date and quantum computer backend"""
    # Return cached data if it's valid and not forcing refresh
    if not force_refresh and _experiment_cache['data'] is not None and _experiment_cache['is_valid']:
        return _experiment_cache['data']
    
    experiments = []
    previous_state = None
    previous_backend = None
    
    # Get all folders in the lab data path
    all_folders = sorted([f for f in os.listdir(LAB_DATA_PATH) 
                         if os.path.isdir(os.path.join(LAB_DATA_PATH, f))])
    
    for folder in all_folders:
        folder_path = os.path.join(LAB_DATA_PATH, folder)
        
        # Get all date folders
        date_folders = sorted([f for f in os.listdir(folder_path) 
                             if os.path.isdir(os.path.join(folder_path, f)) 
                             and f.startswith('2025')])
        
        for date_folder in date_folders:
            date_path = os.path.join(folder_path, date_folder)
            
            # Get all experiment folders for this date
            exp_folders = sorted([f for f in os.listdir(date_path) 
                                if os.path.isdir(os.path.join(date_path, f))])
            
            for exp_folder in exp_folders:
                exp_path = os.path.join(date_path, exp_folder)
                
                # Get all figure files
                plot_files = []
                for f in os.listdir(exp_path):
                    if f.startswith('figure') and f.endswith('.png'):
                        plot_files.append(os.path.join(exp_path, f))
                
                # Only skip if there are no plot files
                if not plot_files:
                    continue
                
                # Sort plot files to put figure_fidelity.png or figure_fidelities.png first if they exist
                plot_files.sort(key=lambda x: (
                    not (x.endswith('figure_fidelity.png') or x.endswith('figure_fidelities.png')),
                    x
                ))
                
                # Get experiment metadata
                node_file = os.path.join(exp_path, 'node.json')
                wiring_file = os.path.join(exp_path, 'quam_state', 'wiring.json')
                state_file = os.path.join(exp_path, 'quam_state', 'state.json')
                
                # Initialize experiment data
                experiment = {
                    'folder': exp_folder,
                    'path': exp_path,
                    'plot_files': plot_files,
                    'lab_folder': folder,
                    'state_changes': []
                }
                
                # Load metadata if available
                if os.path.exists(node_file):
                    try:
                        with open(node_file, 'r') as f:
                            node_data = json.load(f)
                            if 'created_at' in node_data:
                                try:
                                    experiment['date'] = parser.parse(node_data['created_at']).isoformat()
                                except (ValueError, TypeError):
                                    experiment['date'] = parser.parse(date_folder).isoformat()
                            else:
                                experiment['date'] = parser.parse(date_folder).isoformat()
                    except (json.JSONDecodeError, IOError):
                        experiment['date'] = parser.parse(date_folder).isoformat()
                else:
                    experiment['date'] = parser.parse(date_folder).isoformat()
                
                # Get quantum computer backend
                if os.path.exists(wiring_file):
                    try:
                        with open(wiring_file, 'r') as f:
                            wiring_data = json.load(f)
                            experiment['quantum_computer_backend'] = wiring_data.get('network', {}).get('quantum_computer_backend', 'unspecified_backend_name')
                    except (json.JSONDecodeError, KeyError, IOError):
                        experiment['quantum_computer_backend'] = 'unspecified_backend_name'
                else:
                    experiment['quantum_computer_backend'] = 'unspecified_backend_name'
                
                # Track state changes
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            current_state = json.load(f)
                            if previous_state is not None and previous_backend == experiment['quantum_computer_backend']:
                                changes = get_state_changes(current_state, previous_state)
                                if changes:
                                    experiment['state_changes'] = changes
                            previous_state = current_state
                            previous_backend = experiment['quantum_computer_backend']
                    except (json.JSONDecodeError, IOError):
                        pass
                
                experiments.append(experiment)
    
    # Sort experiments by date
    experiments.sort(key=lambda x: x['date'], reverse=True)
    
    # Update cache
    _experiment_cache['data'] = experiments
    _experiment_cache['timestamp'] = time.time()
    _experiment_cache['is_valid'] = True
    
    return experiments

@app.route('/')
def index():
    experiments = get_experiment_data()
    backend_names = sorted(set(exp['quantum_computer_backend'] for exp in experiments))
    return render_template('backend_selection.html', backend_names=backend_names)

@app.route('/backend/<backend_name>')
def backend_view(backend_name):
    experiments = get_experiment_data()
    backend_experiments = [exp for exp in experiments if exp['quantum_computer_backend'] == backend_name]
    return render_template('index.html', 
                         experiments=backend_experiments,
                         backend_name=backend_name,
                         LAB_DATA_PATH=LAB_DATA_PATH)

@app.route('/state_visualization/<backend_name>')
def state_visualization(backend_name):
    # Get available state keys (only qubits and qubit_pairs)
    state_keys = get_state_keys(backend_name)
    
    return render_template('state_visualization.html',
                         backend_name=backend_name,
                         state_keys=state_keys)

@app.route('/api/state_data/<backend_name>')
def get_state_data_api(backend_name):
    key = request.args.get('key')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    
    if not key:
        return jsonify({"error": "No key specified"})
    
    return jsonify(get_state_data(backend_name, key, from_date, to_date))

@app.route('/api/filter_state_keys/<backend_name>')
def filter_state_keys(backend_name):
    filter_text = request.args.get('filter', '')
    keys = get_state_keys(backend_name, filter_text)
    return jsonify(keys)

@app.route('/api/refresh_log')
def refresh_log():
    """Endpoint to manually refresh the state log"""
    try:
        log_state_changes(STATE_LOG_FILE)
        # Force refresh of experiment data
        get_experiment_data(force_refresh=True)
        return jsonify({"status": "success", "message": "State log updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/plot/<path:plot_path>')
def serve_plot(plot_path):
    # Ensure the path is within the LAB_DATA_PATH
    full_path = os.path.join(LAB_DATA_PATH, plot_path)
    if not os.path.exists(full_path) or not full_path.startswith(LAB_DATA_PATH):
        abort(404)
    return send_file(full_path)

@app.route('/experiments')
def get_experiments():
    experiments = get_experiment_data()
    return jsonify(experiments)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Lab Notebook application')
    parser.add_argument('--lab-path', type=str, default=DEFAULT_LAB_DATA_PATH,
                      help=f'Path to lab data directory (default: {DEFAULT_LAB_DATA_PATH})')
    args = parser.parse_args()
    
    try:
        set_lab_data_path(args.lab_path)
        print(f"Using lab data path: {LAB_DATA_PATH}")
        # Update state log when starting the application
        log_state_changes(STATE_LOG_FILE)
        app.run(debug=True)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1) 