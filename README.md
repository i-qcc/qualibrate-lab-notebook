# Lab Notebook POC

A web-based laboratory notebook application for tracking and visualizing quantum computing experiments.

## Features

- View experiment data organized by quantum computer backend
- Track state changes between experiments
- Visualize experiment results with interactive plots
- Automatic state log updates
- Support for multiple lab folders
- Per-lab-path state logs and experiment caches

## Installation

1. Clone the repository:
```bash
git clone git@github.com:i-qcc/lab_notebook_poc.git
cd lab_notebook_poc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python -m lab_notebook.app
```

By default, the application will:
- Scan the `/home/omrieoqm/.qualibrate/user_storage` directory for experiment data
- Store state logs in `~/.lab_notebook/state_logs`
- Store experiment caches in `~/.lab_notebook/experiment_cache`

To specify a custom lab data path:
```bash
python -m lab_notebook.app --lab-path /path/to/your/lab/data
```

Each lab data path will have its own:
- State log file (e.g., `1234_state_log.yml`)
- Experiment cache file (e.g., `1234_experiment_cache.json`)

The numeric ID (e.g., 1234) is automatically generated based on the lab path.

## Development

The application is built with:
- Flask for the web framework
- Plotly for interactive visualizations
- YAML for state logging

## License

[Add your license information here] 