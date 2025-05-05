import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from dateutil import parser
import argparse
import time
import hashlib
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from lab_notebook.config import (
    DEFAULT_LAB_DATA_PATH,
    LAB_DATA_PATH,
    STATE_LOG_FILE,
    CACHE_FILE,
    get_state_log_path,
    get_cache_file_path
)
from lab_notebook.log_utils import compare_dicts, load_json
import signal
import sys

app = FastAPI(title="Lab Notebook API")

# Mount static files
app.mount("/static", StaticFiles(directory="lab_notebook/static"), name="static")

# Templates
templates = Jinja2Templates(directory="lab_notebook/templates")

# Cache for experiment data
_EXPERIMENT_CACHE = {
    'data': None,
    'timestamp': 0,
    'is_valid': True  # Flag to indicate if cache is valid
}

# Pydantic models for request/response validation
class StateDataResponse(BaseModel):
    timestamps: List[str]
    values: List[float]
    error: Optional[str] = None

class StateKeysResponse(BaseModel):
    keys: List[str]

def get_path_id(lab_path):
    """Get a 4-digit integer ID for a given lab path"""
    # Generate MD5 hash
    path_hash = hashlib.md5(lab_path.encode()).hexdigest()
    # Convert first 4 characters of hash to integer and take modulo 10000
    return int(path_hash[:4], 16) % 10000

def save_experiment_cache():
    """Save the experiment cache to a file"""
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(_EXPERIMENT_CACHE, f, indent=2)
    
    print(f"Experiment cache updated with {len(_EXPERIMENT_CACHE['data'])} experiments to {CACHE_FILE}.")

def load_experiment_cache():
    """Load the experiment cache from file if it exists"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                
                _EXPERIMENT_CACHE = json.load(f)
                return _EXPERIMENT_CACHE
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
        try:
            first_line = f.readline().strip()
            latest_timestamp = int(first_line.split(" at ")[-1].strip())
        except:
            print("State log file is corrupted first line is not in the expected format")
            os.remove(STATE_LOG_FILE)
            print("State log file deleted. Setting latest timestamp to 0.")
            latest_timestamp = 0
    
    return latest_timestamp


def log_state_changes(from_timestamp: int, log_file_path: str):

    
    
    all_changes = []

    is_state_changes_logged = False
    prev_state = None
    
    num_experiments = len(_EXPERIMENT_CACHE['data'])

    for i, exp in enumerate(_EXPERIMENT_CACHE['data']):
        if exp["timestamp"] < from_timestamp:
            break # timestamp is sorted in _EXPERIMENT_CACHE['data']

        state_path = exp["state_file"]
        wiring_path = exp["wiring_file"]
        timestamp = exp["timestamp"]  # Already a Unix timestamp

        curr_state = load_json(state_path)
        curr_wiring = load_json(wiring_path)

        backend = curr_wiring["network"].get("quantum_computer_backend", "unspecified_backend")

        if prev_state is not None:
            state_changes = compare_dicts(prev_state, curr_state, timestamp, backend)
            # wiring_changes = compare_dicts(prev_wiring, curr_wiring, timestamp, backend)

            all_changes.extend(state_changes)
            # all_changes.extend(state_changes + wiring_changes)

            if len(_EXPERIMENT_CACHE['data'][i-1]["state_changes"]) == 0: # TODO : in case the state changes are not already logged from the patches keys in node.json - this is a hack to bypass a current bug in qualibrate when running graphs 
                _EXPERIMENT_CACHE['data'][i-1]["state_changes"] = [
                    change[len(backend + ' - '):].rsplit(' at ', 1)[0] for change in state_changes
                ]  # no logging if backend name and timestamp
                is_state_changes_logged = True
        # prev_state, prev_wiring = curr_state, curr_wiring
        prev_state = curr_state

    with open(log_file_path, 'a') as f:  # Open the file in append mode
        f.write("\n".join(all_changes) + "\n")
    print(f"Changes appended to {log_file_path}")
    
    if is_state_changes_logged:
        save_experiment_cache()

def update_state_log_incremental():
    """Update the state log with only new experiments since last update"""
    
    latest_timestamp = get_latest_logged_timestamp()
    
    log_state_changes(latest_timestamp, log_file_path=STATE_LOG_FILE)

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
                update_experiment_data()
                for exp in _EXPERIMENT_CACHE['data']:
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
    # Convert path to string if it's not already
    path_str = str(path)
    
    # Remove leading slash if present
    if path_str.startswith('/'):
        path_str = path_str[1:]
    
    # Remove 'quam/' prefix if present
    if path_str.startswith('quam/'):
        path_str = path_str[5:]
    
    # Split by '/' and join with dots
    return '.'.join(path_str.split('/'))

def update_experiment_data(full_refresh=False):
    """Get all experiment data organized by date and quantum computer backend"""
    
    global _EXPERIMENT_CACHE
    
    # If forcing refresh, clear the cache
    if full_refresh:
        _EXPERIMENT_CACHE = {
            'data': None,
            'latest_timestamp': 0,
            'is_valid': True
        }
    else: # TODO: only need to load from file cache in the first run of app
        # Try to load from file cache first
        file_cache = load_experiment_cache()
        if file_cache is not None:
            _EXPERIMENT_CACHE = file_cache
    
    # If cache is valid and not empty, use it
    if _EXPERIMENT_CACHE['data'] is not None and _EXPERIMENT_CACHE['is_valid']:
        experiments = _EXPERIMENT_CACHE['data'].copy()
        # Get the latest experiment timestamp from cache
        latest_timestamp = _EXPERIMENT_CACHE['latest_timestamp']
    else:
        experiments = []
        latest_timestamp = None  # Will trigger full scan
    
    # First, collect all experiments with their metadata
    skipped_count = 0
        
    # Get all date folders and sort them by creation time
    date_folders = []
    for date_folder_name in os.listdir(LAB_DATA_PATH): # TODO: make this scanning more robust
        if os.path.isdir(os.path.join(LAB_DATA_PATH, date_folder_name)):
            # Get creation time of the date folder in Unix timestamp
            creation_time = int(os.path.getctime(os.path.join(LAB_DATA_PATH, date_folder_name)))
            date_folders.append((date_folder_name, creation_time))
    
    # Sort date folders by creation time in descending order (newest first)
    date_folders.sort(key=lambda x: x[1], reverse=True) # TODO : consider using a different way of sorting
    
    for date_folder_name, date_folder_creation_time in date_folders:
        # Skip older date folders if we have cached data
        if latest_timestamp is not None:
            if date_folder_creation_time < latest_timestamp:
                break
        
        date_path = os.path.join(LAB_DATA_PATH, date_folder_name)
        
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
                'state_changes': [],
                'quantum_computer_backend': 'unspecified_backend_name',  # Default value
                'state_file': state_file,  # Store state file path for later use
                'wiring_file': wiring_file
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
    _EXPERIMENT_CACHE['data'] = experiments
    _EXPERIMENT_CACHE['latest_timestamp'] = experiments[0]['timestamp']
    _EXPERIMENT_CACHE['is_valid'] = True
    
    # Save cache to file
    save_experiment_cache()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Check if this is the first request since app start
    if not hasattr(app, '_initial_scan_done'):
        # Use cached data on first request unless full refresh was requested
        update_experiment_data(full_refresh=getattr(app, '_force_full_refresh', False))
        app._initial_scan_done = True
    else:
        # Do incremental update for subsequent requests
        update_experiment_data(full_refresh=False)
    
    # Do incremental update for subsequent requests
    update_state_log_incremental()
    
    backend_names = sorted(set(exp['quantum_computer_backend'] for exp in _EXPERIMENT_CACHE['data']))
    return templates.TemplateResponse(
        "backend_selection.html",
        {"request": request, "backend_names": backend_names}
    )

@app.get("/backend/{backend_name}", response_class=HTMLResponse)
async def backend_view(
    request: Request,
    backend_name: str,
    page: int = Query(1, ge=1),
    search: str = Query(""),
    date_from: str = Query(""),
    date_to: str = Query("")
):
    per_page = 200  # Number of experiments per page
    
    update_experiment_data()
    backend_experiments = [exp for exp in _EXPERIMENT_CACHE['data'] if exp['quantum_computer_backend'] == backend_name]
    
    # Apply filters
    if search:
        backend_experiments = [exp for exp in backend_experiments if search.lower() in exp['folder'].lower()]
    
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
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "experiments": paginated_experiments,
            "backend_name": backend_name,
            "LAB_DATA_PATH": LAB_DATA_PATH,
            "current_page": page,
            "total_pages": total_pages,
            "total_experiments": total_experiments,
            "search_term": search,
            "date_from": date_from,
            "date_to": date_to
        }
    )

@app.get("/state_visualization", response_class=HTMLResponse)
async def state_visualization(request: Request):
    update_experiment_data()
    backend_names = sorted(set(exp['quantum_computer_backend'] for exp in _EXPERIMENT_CACHE['data']))
    
    return templates.TemplateResponse(
        "state_visualization.html",
        {"request": request, "backend_names": backend_names}
    )

@app.get("/api/state_data/{backend_name}", response_model=StateDataResponse)
async def get_state_data_api(
    backend_name: str,
    key: str = Query(..., description="The state key to get data for")
):
    try:
        # Get all experiments for this backend
        backend_experiments = [exp for exp in _EXPERIMENT_CACHE['data'] if exp['quantum_computer_backend'] == backend_name]
        
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
                            # For confusion matrix, extract 00 and 11 values
                            if key.endswith('.confusion_matrix'):
                                # Parse the matrix string
                                new_value = json.loads(new_value)            
                                # Extract 00 and 11 values
                                value_00 = float(new_value[0][0])
                                value_11 = float(new_value[1][1])
                                # Add both values with the same timestamp
                                changes.append((exp['timestamp'], value_00))
                                changes.append((exp['timestamp'], value_11))
                            else:
                                # For other values, convert to float
                                new_value = float(new_value)
                                changes.append((exp['timestamp'], new_value))
                        except ValueError:
                            continue
        
        if not changes:
            return StateDataResponse(
                timestamps=[],
                values=[],
                error="No data available for the specified parameters"
            )
        
        # Sort changes by timestamp
        changes.sort(key=lambda x: x[0])
        timestamps, values = zip(*changes)
        
        # Convert timestamps to ISO format
        timestamps = [datetime.fromtimestamp(ts).isoformat() for ts in timestamps]
        
        return StateDataResponse(
            timestamps=timestamps,
            values=list(values),
            error=None
        )
        
    except Exception as e:
        print(f"Error in get_state_data_api: {str(e)}")
        return StateDataResponse(
            timestamps=[],
            values=[],
            error=f"Error processing data: {str(e)}"
        )

@app.get("/api/filter_state_keys/{backend_name}", response_model=StateKeysResponse)
async def filter_state_keys(
    backend_name: str,
    filter: str = Query("", description="Filter text for state keys")
):
    filter_text = filter.lower()
    
    # Get all experiments for this backend
    backend_experiments = [exp for exp in _EXPERIMENT_CACHE['data'] if exp['quantum_computer_backend'] == backend_name]
    
    # Collect all unique state keys
    keys = set()
    for exp in backend_experiments:
        for change in exp.get('state_changes', []):
            if ' : ' in change:
                key = change.split(' : ')[0]
                if key.startswith(('qubits.', 'qubit_pairs.')):
                    if not filter_text or filter_text in key.lower():
                        keys.add(key)
    
    return StateKeysResponse(keys=sorted(list(keys)))

@app.get("/plot/{plot_path:path}")
async def serve_plot(plot_path: str):
    # Replace backslashes with forward slashes and remove any leading/trailing slashes
    plot_path = plot_path.replace('\\', '/').strip('/')
    
    # Ensure the path is within the LAB_DATA_PATH
    full_path = os.path.join(LAB_DATA_PATH, plot_path)
    if not os.path.exists(full_path) or not full_path.startswith(LAB_DATA_PATH):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(full_path)

@app.post("/shutdown")
async def shutdown():
    """Gracefully shut down the server"""
    print("Shutting down server...")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Server shutting down..."}

if __name__ == '__main__':
    import uvicorn
    
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
        
        print("\nStarting Lab Notebook server...")
        print("Access the application at: http://localhost:8000")
        print("Press Ctrl+C to stop the server\n")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1) 