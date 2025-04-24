import os
import json
import shutil
from datetime import datetime, timezone
from lab_notebook.app import update_experiment_data, get_state_changes

def create_test_experiment(base_path, exp_name, timestamp, state_data, patches=None):
    """Create a test experiment with the given state data"""
    # Create experiment directory
    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    
    # Create node.json with timestamp and patches
    node_data = {
        'created_at': timestamp.isoformat()
    }
    if patches:
        node_data['patches'] = patches
    
    with open(os.path.join(exp_path, 'node.json'), 'w') as f:
        json.dump(node_data, f)
    
    # Create quam_state directory
    quam_state_path = os.path.join(exp_path, 'quam_state')
    os.makedirs(quam_state_path, exist_ok=True)
    
    # Create wiring.json
    wiring_data = {
        'network': {
            'quantum_computer_backend': 'test_backend'
        }
    }
    with open(os.path.join(quam_state_path, 'wiring.json'), 'w') as f:
        json.dump(wiring_data, f)
    
    # Create state.json with the given state data
    with open(os.path.join(quam_state_path, 'state.json'), 'w') as f:
        json.dump(state_data, f)
    
    # Create a dummy PNG file
    with open(os.path.join(exp_path, 'test.png'), 'w') as f:
        f.write('dummy png content')

def test_state_changes():
    """Test state changes between experiments"""
    # Create a temporary test directory
    test_dir = 'test_lab_data'
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a lab folder
    lab_folder = os.path.join(test_dir, 'test_lab')
    os.makedirs(lab_folder, exist_ok=True)
    
    # Create a date folder
    date_folder = os.path.join(lab_folder, '2025-03-18')
    os.makedirs(date_folder, exist_ok=True)
    
    # Create two experiments with different states
    exp1_state = {
        'qubits': {
            'qubitA1': {
                'resonator': {
                    'operations': {
                        'readout': {
                            'integration_weights_angle': -9.141493827134486
                        }
                    }
                }
            }
        }
    }
    
    exp2_state = {
        'qubits': {
            'qubitA1': {
                'resonator': {
                    'operations': {
                        'readout': {
                            'integration_weights_angle': -21.716087221853375
                        }
                    }
                }
            }
        }
    }
    
    # Create patches for exp2
    exp2_patches = [{
        'op': 'replace',
        'path': '/qubits/qubitA1/resonator/operations/readout/integration_weights_angle',
        'old': -9.141493827134486,
        'value': -21.716087221853375
    }]
    
    # Create experiments with different timestamps
    create_test_experiment(
        date_folder,
        'exp1',
        datetime(2025, 3, 18, 10, 0, 0, tzinfo=timezone.utc),
        exp1_state
    )
    
    create_test_experiment(
        date_folder,
        'exp2',
        datetime(2025, 3, 18, 11, 0, 0, tzinfo=timezone.utc),
        exp2_state,
        patches=exp2_patches
    )
    
    # Set the lab data path to our test directory
    from lab_notebook.app import set_lab_data_path
    set_lab_data_path(test_dir)
    
    # Get experiment data
    experiments = update_experiment_data(full_refresh=True)
    
    # Verify we have two experiments
    assert len(experiments) == 2, f"Expected 2 experiments, got {len(experiments)}"
    
    # Find exp2 (newer experiment)
    exp2 = next(exp for exp in experiments if exp['folder'] == 'exp2')
    
    # Verify state changes
    expected_change = "qubits.qubitA1.resonator.operations.readout.integration_weights_angle : -9.141493827134486 -> -21.716087221853375"
    assert expected_change in exp2['state_changes'], f"Expected change '{expected_change}' not found in state changes: {exp2['state_changes']}"
    
    print("Test passed successfully!")
    print(f"Found state changes: {exp2['state_changes']}")
    
    # Clean up
    shutil.rmtree(test_dir)

if __name__ == '__main__':
    test_state_changes() 