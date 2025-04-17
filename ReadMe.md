# RL Beta Amyloid Refinement

## Google Drive Folder Link and Colab Notebook
https://drive.google.com/drive/folders/16_fQiZGjH3lXOmqZYGQq4EQGG4tykMFg?usp=drive_link
https://colab.research.google.com/drive/16rcgjxeV6LZ8aO0pEZUUi_mU2ZfyoX-n?usp=sharing

## Overview
This project implements a reinforcement learning environment for the refinement of beta amyloid structures using the Gymnasium framework. The environment simulates the manipulation of phi/psi angles of protein structures to optimize their conformation based on energy, RMSD (Root Mean Square Deviation), and SASA (Solvent Accessible Surface Area).

## Requirements
To run this project, you need to have Python installed along with the following packages:

```bash
biopython==1.81
numpy==1.24.3
gymnasium==0.29.1
freesasa
stable-baselines3
peptidebuilder
pdbfixer
openmm
```

You can install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Project Structure
- `amyloid_env.py`: Contains the main environment class and helper functions
- `train_rl.py`: Training script for the reinforcement learning model
- `inference.py`: Script for running inference on new structures
- `backbone_to_structure.py`: Utilities for converting backbone angles to full structures
- `test_env.py`: Test suite for the environment

## Key Components

### AmyloidEnv Class
The main environment class that inherits from `gym.Env`. Features include:
- State representation using φ/ψ angles
- Customizable action space for angle modifications
- Reward computation based on multiple factors
- Support for both training and inference modes

### Core Functions
- `load_structure(cif_path)`: Loads protein structures from CIF files and extracts:
  - φ/ψ angles
  - Atomic coordinates
  - B-factors
  - Hydrophobicity
  - SASA values
  - Residue IDs

- `pad_phi_psi(vec, fixed_dim)`: Utility function to standardize vector dimensions
- `compute_energy(state)`: Calculates energy based on dihedral angles
- `compute_rmsd(state, target)`: Computes RMSD between states
- `compute_sasa(state)`: Calculates solvent-accessible surface area

## Usage

### Training Mode
```python
from amyloid_env import AmyloidEnv

af2_path = "path/to/your/af2_file.cif"
exp_path = "path/to/your/exp_file.cif"
env = AmyloidEnv(af2_path, exp_path)

state = env.reset()
action = env.action_space.sample()  # Sample random action
next_state, reward, terminated, truncated, info = env.step(action)
```

### Inference Mode
```python
from inference import run_inference

refined_angles = run_inference(
    af2_cif_path="path/to/prediction.cif",
    max_steps=50,
    fixed_dim=100
)
```

### Testing
Run the test suite to validate environment functionality:
```bash
python test_env.py
```

## Data Organization
- Place CIF files in the `data` directory
- Inference outputs will be saved in the `inference_outputs` directory
- Trained models should be saved as "ppo_amyloid_final.zip"

## Model Details
The project uses Proximal Policy Optimization (PPO) from stable-baselines3 for training. The environment supports:
- Customizable maximum steps
- Adjustable vector dimensions
- Both training and inference modes
- Detailed step information including energy, RMSD, and SASA values


## Acknowledgments
- Biopython for protein structure manipulation
- Gymnasium for the reinforcement learning framework
- OpenMM for molecular mechanics calculations
- Stable-baselines3 for RL implementation
