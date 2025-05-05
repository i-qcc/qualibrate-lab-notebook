# Qualibrate Lab Notebook

A web-based application for visualizing and analyzing experiment data from Qualibrate.

## About

Qualibrate Lab Notebook is an extension to Qualibrate that provides a web-based interface for:
- Visualizing experiment results and plots
- Tracking state changes between experiments
- Analyzing experiment data across different quantum computer backends
- Organizing experiments chronologically

## Installation

```bash
pip install qualibrate-lab-notebook
```

## Usage

Start the Qualibrate Lab Notebook server:

```bash
qualibrate-lab-notebook start --lab-path /path/to/your/qualibrate/data
```

### Command Line Options

- `--lab-path`: Path to your Qualibrate data directory (default: ~/.qualibrate/user_storage)
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--full-refresh`: Force a full refresh of the experiment cache

### Examples

```bash
# Start with default settings (uses ~/.qualibrate/user_storage)
qualibrate-lab-notebook start

# Start with custom lab path
qualibrate-lab-notebook start --lab-path /path/to/qualibrate_data/project/data (for example : "~/.qualibrate/user_storage/QC1")

# Start with custom host and port
qualibrate-lab-notebook start --host 127.0.0.1 --port 8080

# Start with full refresh
qualibrate-lab-notebook start --full-refresh
```

## Features

- Visualize Qualibrate experiment plots and results
- Track state changes between experiments
- Support for multiple quantum computer backends
- Automatic experiment data caching
- Web-based interface for easy access
- Integration with Qualibrate's data storage format

## Requirements

- Python 3.11 or 3.12
- FastAPI
- Uvicorn
- Matplotlib
- NumPy
- H5Py

## License

BSD 3-Clause License

## Author

Quantum Machines (https://www.quantum-machines.co) 