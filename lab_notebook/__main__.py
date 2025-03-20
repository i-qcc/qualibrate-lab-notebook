import argparse
from lab_notebook.app import app, DEFAULT_LAB_DATA_PATH, set_lab_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Lab Notebook application')
    parser.add_argument('--lab-path', type=str, default=DEFAULT_LAB_DATA_PATH,
                      help=f'Path to lab data directory (default: {DEFAULT_LAB_DATA_PATH})')
    args = parser.parse_args()
    
    try:
        set_lab_data_path(args.lab_path)
        print(f"Using lab data path: {args.lab_path}")
        app.run(debug=True)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1) 