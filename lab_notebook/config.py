import os
import hashlib

# Default configuration
DEFAULT_LAB_DATA_PATH =  os.path.expanduser("~/.qualibrate/user_storage/init_project") # os.path.expanduser("~/.from_cloud_storage/user_storage/QC1")  #
LAB_DATA_PATH = DEFAULT_LAB_DATA_PATH
STATE_LOGS_DIR = os.path.expanduser("~/.qualibrate-lab-notebook/state_logs")
CACHE_DIR = os.path.expanduser("~/.qualibrate-lab-notebook/experiment_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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