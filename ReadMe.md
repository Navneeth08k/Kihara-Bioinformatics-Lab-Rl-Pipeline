# RL Beta Amyloid Refinement

## Overview
This project implements a reinforcement learning environment for the refinement of beta amyloid structures using the Gymnasium framework. The environment simulates the manipulation of phi/psi angles of protein structures to optimize their conformation based on energy, RMSD (Root Mean Square Deviation), and SASA (Solvent Accessible Surface Area).

## Requirements
To run this project, you need to have Python installed along with the following packages:

- `biopython==1.81`
- `numpy==1.24.3`
- `gymnasium==0.29.1`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
1. **Load Structures**: The environment requires CIF files for both the predicted and experimental structures. Ensure that your CIF files are organized in the `data` directory.

2. **Run Tests**: You can test the environment using the provided `test_env.py` script. This script will load sample pairs of CIF files and run a series of tests to validate the environment's functionality.

   ```bash
   python test_env.py
   ```

3. **Create and Use the Environment**: You can create an instance of the `AmyloidEnv` class in your Python code to interact with the environment. Here’s a basic example:

   ```python
   from amyloid_env import AmyloidEnv

   af2_path = "path/to/your/af2_file.cif"
   exp_path = "path/to/your/exp_file.cif"
   env = AmyloidEnv(af2_path, exp_path)

   state = env.reset()
   action = env.action_space.sample()  # Sample random action
   next_state, reward, done, info = env.step(action)
   ```

## Components
- **AmyloidEnv**: The main environment class that inherits from `gym.Env`. It handles the state representation, action application, and reward computation.
- **load_structure**: A utility function to load protein structures from CIF files and extract relevant features.
- **compute_energy**: A function to calculate the energy based on dihedral angles.
- **compute_rmsd**: A function to compute the RMSD between the current state and the target state.
- **compute_sasa**: A placeholder function for calculating the solvent-accessible surface area.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Biopython for protein structure manipulation.
- Gymnasium for providing the reinforcement learning framework.