import argparse
from lab_notebook.app import app, DEFAULT_LAB_DATA_PATH, set_lab_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Lab Notebook application')
    parser.add_argument('--lab-path', type=str, default=DEFAULT_LAB_DATA_PATH,
                      help=f'Path to lab data directory (default: {DEFAULT_LAB_DATA_PATH})')
    parser.add_argument('--full-refresh', action='store_true',
                      help='Force a full refresh of the experiment cache')
    args = parser.parse_args()
    
    try:
        set_lab_data_path(args.lab_path)
        print(f"Using lab data path: {DEFAULT_LAB_DATA_PATH}")
        # Store the full refresh flag in the app object
        app._force_full_refresh = args.full_refresh
        if args.full_refresh:
            print("Performing full refresh of experiment cache")
        # # Update state log when starting the application
        # from lab_notebook.app import update_state_log_incremental
        # update_state_log_incremental()
        app.run(debug=True)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1) 