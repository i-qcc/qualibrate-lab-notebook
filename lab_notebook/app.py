import os
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, send_file, abort, request
from dateutil import parser
from collections import defaultdict
import glob
import ast
import argparse
from lab_notebook.log_utils import ChangeAnalyzer, get_sorted_experiments, log_state_changes
import numpy as np
from functools import lru_cache
import time
from pathlib import Path
import hashlib

app = Flask(__name__)

# Default configuration
DEFAULT_LAB_DATA_PATH = "/home/omrieoqm/.qualibrate/user_storage"
LAB_DATA_PATH = DEFAULT_LAB_DATA_PATH
STATE_LOGS_DIR = os.path.expanduser("~/.lab_notebook/state_logs")
CACHE_DIR = os.path.expanduser("~/.lab_notebook/experiment_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache for experiment data
_experiment_cache = {
    'data': None,
    'timestamp': 0,
    'is_valid': True  # Flag to indicate if cache is valid
}

def get_path_id(lab_path):
    """Get a 4-digit integer ID for a given lab path"""
    # Generate MD5 hash
    path_hash = hashlib.md5(lab_path.encode()).hexdigest()
    # Convert first 4 characters of hash to integer and take modulo 10000
    return int(path_hash[:4], 16) % 10000

def get_state_log_path(lab_path):
    """Get the state log path for a given lab path"""
    # Create state logs directory if it doesn't exist
    os.makedirs(STATE_LOGS_DIR, exist_ok=True)
    
    # Get the path ID
    path_id = get_path_id(lab_path)
    
    # Create the state log file path
    return os.path.join(STATE_LOGS_DIR, f"{path_id}_state_log.yml")

def get_cache_file_path(lab_path):
    """Get the cache file path for a given lab path"""
    # Get the path ID
    path_id = get_path_id(lab_path)
    
    # Create the cache file path
    return os.path.join(CACHE_DIR, f"{path_id}_experiment_cache.json")

# Initialize STATE_LOG_FILE and CACHE_FILE with the default path
STATE_LOG_FILE = get_state_log_path(LAB_DATA_PATH)
CACHE_FILE = get_cache_file_path(LAB_DATA_PATH)

def save_experiment_cache():
    """Save the experiment cache to a file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(_experiment_cache, f, indent=2)

def load_experiment_cache():
    """Load the experiment cache from file if it exists"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print("Warning: Cache file corrupted, starting fresh")
            return None
    return None

def set_lab_data_path(path):
    """Set the lab data path and ensure it exists"""
    global LAB_DATA_PATH, STATE_LOG_FILE, CACHE_FILE
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    LAB_DATA_PATH = path
    STATE_LOG_FILE = get_state_log_path(path)
    CACHE_FILE = get_cache_file_path(path)

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
    
    experiments = get_sorted_experiments(LAB_DATA_PATH, from_timestamp=latest_timestamp)
    
    new_experiments = [exp for exp in experiments if exp['timestamp'] > latest_timestamp]
    
    log_state_changes(STATE_LOG_FILE, experiments=new_experiments)

def get_state_data(backend_name, key, from_date=None, to_date=None):
    """Get state data for a specific key and time range"""
    if not os.path.exists(STATE_LOG_FILE):
        return {
            "timestamps": [],
            "values": [],
            "error": "No state data available"
        }
    
    try:
        # Read all changes from the state log
        changes = []
        with open(STATE_LOG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse the line format: "key : old_value -> new_value"
                parts = line.split(' : ')
                if len(parts) != 2:
                    continue
                    
                change_key, values = parts
                if change_key != key:
                    continue
                    
                # Parse the values
                value_parts = values.split(' -> ')
                if len(value_parts) != 2:
                    continue
                    
                old_value, new_value = value_parts
                
                # Try to convert the new value to float
                try:
                    new_value = float(new_value)
                except ValueError:
                    continue
                
                # Get the timestamp from the experiment data
                experiments = get_experiment_data()
                for exp in experiments:
                    if any(key in change for change in exp.get('state_changes', [])):
                        timestamp = exp.get('timestamp', 0)
                        changes.append((timestamp, new_value))
                        break
        
        if not changes:
            return {
                "timestamps": [],
                "values": [],
                "error": "No data available for the specified parameters"
            }
        
        # Sort changes by timestamp
        changes.sort(key=lambda x: x[0])
        timestamps, values = zip(*changes)
        
        # Convert timestamps to datetime objects
        timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Filter by date range if specified
        if from_date:
            from_date = parser.parse(from_date)
            filtered_data = [(t, v) for t, v in zip(timestamps, values) if t >= from_date]
            if filtered_data:
                timestamps, values = zip(*filtered_data)
            else:
                return {
                    "timestamps": [],
                    "values": [],
                    "error": "No data available in the specified date range"
                }
        
        if to_date:
            to_date = parser.parse(to_date)
            filtered_data = [(t, v) for t, v in zip(timestamps, values) if t <= to_date]
            if filtered_data:
                timestamps, values = zip(*filtered_data)
            else:
                return {
                    "timestamps": [],
                    "values": [],
                    "error": "No data available in the specified date range"
                }
        
        return {
            "timestamps": [t.isoformat() for t in timestamps],
            "values": list(values),
            "error": None
        }
        
    except Exception as e:
        print(f"Error in get_state_data: {str(e)}")
        return {
            "timestamps": [],
            "values": [],
            "error": f"Error processing data: {str(e)}"
        }

def format_value(value):
    """Format a value for display in state changes"""
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(format_value(v) for v in value)}]"
    return str(value)

def compare_states(current, previous, path=""):
    """Compare two states and record changes"""
    changes = []
    
    if isinstance(current, dict) and isinstance(previous, dict):
        for key in set(current.keys()) | set(previous.keys()):
            new_path = f"{path}.{key}" if path else key
            current_val = current.get(key)
            previous_val = previous.get(key)
            
            if current_val != previous_val:
                if isinstance(current_val, (dict, list)) and isinstance(previous_val, (dict, list)):
                    compare_states(current_val, previous_val, new_path)
                else:
                    changes.append(f"{new_path} : {format_value(previous_val)} -> {format_value(current_val)}")
    
    elif isinstance(current, list) and isinstance(previous, list):
        for i, (curr_item, prev_item) in enumerate(zip(current, previous)):
            new_path = f"{path}[{i}]"
            if curr_item != prev_item:
                if isinstance(curr_item, (dict, list)) and isinstance(prev_item, (dict, list)):
                    compare_states(curr_item, prev_item, new_path)
                else:
                    changes.append(f"{new_path} : {format_value(prev_item)} -> {format_value(curr_item)}")
    else:
        if current != previous:
            changes.append(f"{path} : {format_value(previous)} -> {format_value(current)}")
    
    return changes

def get_state_changes(node_data):
    """Extract state changes from node data"""
    changes = []
    if 'patches' in node_data:
        for patch in node_data['patches']:
            path = format_path(patch['path'])
            old_value = format_value(patch.get('old', 'None'))
            new_value = format_value(patch['value'])
            changes.append(f"{path} : {old_value} -> {new_value}")
    return changes

def format_path(path):
    """Format a path for display in state changes"""
    return '.'.join(str(p) for p in path)

def get_experiment_data(full_refresh=False):
    """Get all experiment data organized by date and quantum computer backend"""
    global _experiment_cache
    
    # If forcing refresh, clear the cache
    if full_refresh:
        _experiment_cache = {
            'data': None,
            'timestamp': 0,
            'is_valid': True
        }
    else:
        # Try to load from file cache first
        file_cache = load_experiment_cache()
        if file_cache is not None:
            _experiment_cache = file_cache
    
    # If cache is valid and not empty, use it
    if _experiment_cache['data'] is not None and _experiment_cache['is_valid']:
        experiments = _experiment_cache['data'].copy()
        # Get the latest experiment timestamp from cache
        latest_timestamp = max(exp['timestamp'] for exp in experiments)
    else:
        experiments = []
        latest_timestamp = None  # Will trigger full scan
    
    # First, collect all experiments with their metadata
    skipped_count = 0
    
    # Get all folders in the lab data path
    all_folders = sorted([f for f in os.listdir(LAB_DATA_PATH) 
                         if os.path.isdir(os.path.join(LAB_DATA_PATH, f))])
    
    for folder in all_folders:
        folder_path = os.path.join(LAB_DATA_PATH, folder)
        
        # Get all date folders and sort them by creation time
        date_folders = []
        for date_folder_name in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, date_folder_name)):
                # Get creation time of the date folder in Unix timestamp
                creation_time = int(os.path.getctime(os.path.join(folder_path, date_folder_name)))
                date_folders.append((date_folder_name, creation_time))
        
        # Sort date folders by creation time in descending order (newest first)
        date_folders.sort(key=lambda x: x[1], reverse=True)
        
        for date_folder_name, date_folder_creation_time in date_folders:
            # Skip older date folders if we have cached data
            if latest_timestamp is not None:
                if date_folder_creation_time < latest_timestamp:
                    break
            
            date_path = os.path.join(folder_path, date_folder_name)
            
            # Get all experiment folders for this date
            exp_folders = sorted([f for f in os.listdir(date_path) 
                                if os.path.isdir(os.path.join(date_path, f))])
            
            for exp_folder in exp_folders:
                exp_path = os.path.join(date_path, exp_folder)
                
                # Skip if we already have this experiment in cache
                if any(exp['path'] == exp_path for exp in experiments):
                    continue
                
                # Get all PNG files
                plot_files = []
                for f in os.listdir(exp_path):
                    if f.endswith('.png'):
                        plot_files.append(os.path.join(exp_path, f))
                
                # Only skip if there are no PNG files
                if not plot_files:
                    skipped_count += 1
                    continue
                
                # Sort plot files to put fidelity plots first if they exist
                plot_files.sort(key=lambda x: (
                    not ('fidelity' in os.path.basename(x).lower()),
                    x
                ))
                
                # Get experiment metadata
                node_file = os.path.join(exp_path, 'node.json')
                wiring_file = os.path.join(exp_path, 'quam_state', 'wiring.json')
                state_file = os.path.join(exp_path, 'quam_state', 'state.json')
                
                # Initialize experiment data with default backend
                experiment = {
                    'folder': exp_folder,
                    'path': exp_path,
                    'plot_files': plot_files,
                    'lab_folder': folder,
                    'state_changes': [],
                    'quantum_computer_backend': 'unspecified_backend_name',  # Default value
                    'state_file': state_file  # Store state file path for later use
                }
                
                # Load metadata and extract timestamp
                if not os.path.exists(node_file):
                    skipped_count += 1
                    continue
                
                try:
                    with open(node_file, 'r') as f:
                        node_data = json.load(f)
                        if 'created_at' not in node_data:
                            skipped_count += 1
                            continue
                            
                        # Parse the timestamp in the format "2025-03-18T12:42:29+02:00"
                        created_at = parser.parse(node_data['created_at'])
                        experiment['date'] = created_at.isoformat()
                        experiment['timestamp'] = int(created_at.timestamp())
                        
                        # Get state changes from patches
                        if 'patches' in node_data:
                            experiment['state_changes'] = get_state_changes(node_data)
                except (json.JSONDecodeError, ValueError, TypeError):
                    skipped_count += 1
                    continue
                
                # Get quantum computer backend
                if os.path.exists(wiring_file):
                    try:
                        with open(wiring_file, 'r') as f:
                            wiring_data = json.load(f)
                            experiment['quantum_computer_backend'] = wiring_data.get('network', {}).get('quantum_computer_backend', 'unspecified_backend_name')
                    except (json.JSONDecodeError, KeyError, IOError):
                        pass
                
                experiments.append(experiment)
    
    if skipped_count > 0:
        print(f"Warning: {skipped_count} experiments were skipped due to missing or invalid data")
    
    # Sort experiments by timestamp in ascending order (oldest first)
    experiments.sort(key=lambda x: x['timestamp'])
    
    # Sort experiments by timestamp in descending order (newest first) for display
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Update cache
    _experiment_cache['data'] = experiments
    _experiment_cache['timestamp'] = time.time()
    _experiment_cache['is_valid'] = True
    
    # Save cache to file
    save_experiment_cache()
    
    return experiments

@app.route('/')
def index():
    # Do incremental update for subsequent requests
    update_state_log_incremental()
    
    # Check if this is the first request since app start
    if not hasattr(app, '_initial_scan_done'):
        # Use cached data on first request unless full refresh was requested
        experiments = get_experiment_data(full_refresh=getattr(app, '_force_full_refresh', False))
        app._initial_scan_done = True
    else:
        # Do incremental update for subsequent requests
        experiments = get_experiment_data(full_refresh=False)
    
    backend_names = sorted(set(exp['quantum_computer_backend'] for exp in experiments))
    return render_template('backend_selection.html', backend_names=backend_names)

@app.route('/backend/<backend_name>')
def backend_view(backend_name):
    page = request.args.get('page', 1, type=int)
    search_term = request.args.get('search', '').lower()
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    per_page = 200  # Number of experiments per page
    
    experiments = get_experiment_data()
    backend_experiments = [exp for exp in experiments if exp['quantum_computer_backend'] == backend_name]
    
    # Apply filters
    if search_term:
        backend_experiments = [exp for exp in backend_experiments if search_term in exp['folder'].lower()]
    
    if date_from:
        date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
        backend_experiments = [exp for exp in backend_experiments if datetime.fromisoformat(exp['date']).date() >= date_from]
    
    if date_to:
        date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
        backend_experiments = [exp for exp in backend_experiments if datetime.fromisoformat(exp['date']).date() <= date_to]
    
    # Calculate pagination
    total_experiments = len(backend_experiments)
    total_pages = (total_experiments + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get experiments for current page
    paginated_experiments = backend_experiments[start_idx:end_idx]
    
    return render_template('index.html', 
                         experiments=paginated_experiments,
                         backend_name=backend_name,
                         LAB_DATA_PATH=LAB_DATA_PATH,
                         current_page=page,
                         total_pages=total_pages,
                         total_experiments=total_experiments,
                         search_term=search_term,
                         date_from=date_from,
                         date_to=date_to)

@app.route('/state_visualization')
def state_visualization():
    # Get all available backend names
    experiments = get_experiment_data()
    backend_names = sorted(set(exp['quantum_computer_backend'] for exp in experiments))
    
    return render_template('state_visualization.html',
                         backend_names=backend_names)

@app.route('/api/state_data/<backend_name>')
def get_state_data_api(backend_name):
    key = request.args.get('key')
    
    if not key:
        return jsonify({"error": "No key specified"})
    
    try:
        # Get all experiments for this backend
        experiments = _experiment_cache['data']
        backend_experiments = [exp for exp in experiments if exp['quantum_computer_backend'] == backend_name]
        
        # Collect all state changes for this key
        changes = []
        for exp in backend_experiments:
            for change in exp.get('state_changes', []):
                if change.startswith(key + ' : '):
                    # Parse the change: "key : old_value -> new_value"
                    parts = change.split(' : ')[1].split(' -> ')
                    if len(parts) == 2:
                        try:
                            new_value = parts[1]
                            # For confusion matrix, keep the string representation
                            if key.endswith('.confusion_matrix'):
                                changes.append((exp['timestamp'], new_value))
                            else:
                                # For other values, convert to float
                                new_value = float(new_value)
                                changes.append((exp['timestamp'], new_value))
                        except ValueError:
                            continue
        
        if not changes:
            return jsonify({
                "timestamps": [],
                "values": [],
                "error": "No data available for the specified parameters"
            })
        
        # Sort changes by timestamp
        changes.sort(key=lambda x: x[0])
        timestamps, values = zip(*changes)
        
        # Convert timestamps to ISO format
        timestamps = [datetime.fromtimestamp(ts).isoformat() for ts in timestamps]
        
        return jsonify({
            "timestamps": timestamps,
            "values": list(values),
            "error": None
        })
        
    except Exception as e:
        print(f"Error in get_state_data_api: {str(e)}")
        return jsonify({
            "timestamps": [],
            "values": [],
            "error": f"Error processing data: {str(e)}"
        })

@app.route('/api/filter_state_keys/<backend_name>')
def filter_state_keys(backend_name):
    filter_text = request.args.get('filter', '').lower()
    
    # Get all experiments for this backend
    experiments = _experiment_cache['data']
    backend_experiments = [exp for exp in experiments if exp['quantum_computer_backend'] == backend_name]
    
    # Collect all unique state keys
    keys = set()
    for exp in backend_experiments:
        for change in exp.get('state_changes', []):
            if ' : ' in change:
                key = change.split(' : ')[0]
                if key.startswith(('qubits.', 'qubit_pairs.')):
                    if not filter_text or filter_text in key.lower():
                        keys.add(key)
    
    return jsonify(sorted(list(keys)))

@app.route('/plot/<path:plot_path>')
def serve_plot(plot_path):
    # Ensure the path is within the LAB_DATA_PATH
    full_path = os.path.join(LAB_DATA_PATH, plot_path)
    if not os.path.exists(full_path) or not full_path.startswith(LAB_DATA_PATH):
        abort(404)
    return send_file(full_path)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Run the Lab Notebook application')
    arg_parser.add_argument('--lab-path', type=str, default=DEFAULT_LAB_DATA_PATH,
                      help=f'Path to lab data directory (default: {DEFAULT_LAB_DATA_PATH})')
    arg_parser.add_argument('--full-refresh', action='store_true',
                      help='Force a full refresh of the experiment cache')
    args = arg_parser.parse_args()
    
    try:
        set_lab_data_path(args.lab_path)
        print(f"Using lab data path: {LAB_DATA_PATH}")
        # Store the full refresh flag in the app object
        app._force_full_refresh = args.full_refresh
        if args.full_refresh:
            print("Performing full refresh of experiment cache")
        # Update state log when starting the application
        update_state_log_incremental()
        app.run(debug=True) 
    except ValueError as e:
        print(f"Error: {e}")
        exit(1) 