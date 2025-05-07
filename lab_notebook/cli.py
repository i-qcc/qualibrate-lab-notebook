import argparse
import sys
import socket
from lab_notebook.app import app, set_lab_data_path

def is_port_in_use(port):
    """Check if a port is already in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', port))
        sock.close()
        return False
    except (socket.error, OSError):
        sock.close()
        return True

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

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
            from uvicorn.config import Config
            from uvicorn.main import Server
            
            port = args.port
            if is_port_in_use(port):
                print(f"Port {port} is already in use. Looking for an available port...")
                port = find_available_port(port)
                print(f"Found available port: {port}")
            
            config = Config(app, host=args.host, port=port, reload=False, log_level="error")
            server = Server(config=config)
            print("\nStarting Lab Notebook server...")
            print(f"Access the application at: http://localhost:{port}")
            print("Press Ctrl+C to stop the server\n")
            server.run()
            
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 