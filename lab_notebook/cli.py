import argparse
import sys
from lab_notebook.app import app, set_lab_data_path

def main():
    parser = argparse.ArgumentParser(description='Lab Notebook - A web-based application for viewing and organizing laboratory experiment plots')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the Lab Notebook server')
    start_parser.add_argument('--lab-path', type=str, help='Path to lab data directory')
    start_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
    start_parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to (default: 8000)')
    start_parser.add_argument('--full-refresh', action='store_true', help='Force a full refresh of the experiment cache')

    args = parser.parse_args()

    if args.command == 'start':
        try:
            if args.lab_path:
                set_lab_data_path(args.lab_path)
            
            # Store the full refresh flag in the app object
            app._force_full_refresh = args.full_refresh
            if args.full_refresh:
                print("Performing full refresh of experiment cache")
            
            import uvicorn
            print(f"Starting Lab Notebook server on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 