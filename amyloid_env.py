import gymnasium as gym
from gymnasium import spaces
from Bio.PDB import MMCIFParser, PPBuilder, DSSP, Polypeptide
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import *
import numpy as np
import os
import tempfile
import csv
import glob

def load_structure(cif_path):
    """
    Loads a CIF file and returns a dictionary of protein features:
    - phi/psi angles (state representation)
    - atomic coordinates
    - B-factors
    - hydrophobicity
    - SASA (placeholder)
    - Residue IDs
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    
    model = structure[0]
    chain = next(model.get_chains())
    
    # Extract phi/psi angles
    ppb = PPBuilder()
    phi_psi_angles = []
    residue_ids = []
    for pp in ppb.build_peptides(chain):
        phi_psi = pp.get_phi_psi_list()
        for i, (phi, psi) in enumerate(phi_psi):
            if phi is not None and psi is not None:
                phi_psi_angles.extend([np.degrees(phi), np.degrees(psi)])
                residue_ids.append(pp[i].id[1])
    
    # Extract atomic coordinates and B-factors
    coords = []
    b_factors = []
    atom_types = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
            b_factors.append(atom.get_bfactor())
            atom_types.append(atom.get_id())
    
    # Extract hydrophobicity (Kyte-Doolittle scale)
    hydrophobicity = []
    for residue in chain:
        if is_aa(residue):
            try:
                res_code = protein_letters_3to1.get(residue.get_resname(), 'X')
                h_scale = {
                    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
                }
                hydrophobicity.append(h_scale.get(res_code, 0))
            except Exception as e:
                hydrophobicity.append(0)
    
    # Placeholder SASA calculation: assign a constant value per residue
    sasa = np.ones(len(residue_ids)) * 100
    
    return {
        'phi_psi': np.array(phi_psi_angles, dtype=np.float32),
        'coordinates': np.array(coords, dtype=np.float32),
        'b_factors': np.array(b_factors, dtype=np.float32),
        'atom_types': atom_types,
        'hydrophobicity': np.array(hydrophobicity, dtype=np.float32),
        'sasa': np.array(sasa, dtype=np.float32),
        'residue_ids': residue_ids
    }

def apply_action(state, action):
    """
    Modifies only the phi/psi angles.
    """
    new_phi_psi = state['phi_psi'] + action
    new_phi_psi = np.clip(new_phi_psi, -180, 180)
    new_state = state.copy()
    new_state['phi_psi'] = new_phi_psi
    return new_state

def compute_energy(state):
    return np.sum(np.abs(state['phi_psi']))

def compute_rmsd(state, target_state):
    return np.sqrt(np.mean((state['phi_psi'] - target_state['phi_psi']) ** 2))

def compute_sasa(state):
    return np.sum(state['sasa'])

class AmyloidEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, af2_pdb_path, exp_pdb_path, max_steps=50):
        super(AmyloidEnv, self).__init__()
        self.af2_pdb_path = af2_pdb_path
        self.exp_pdb_path = exp_pdb_path
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load structures
        self.af2_state = load_structure(self.af2_pdb_path)
        self.exp_state = load_structure(self.exp_pdb_path)
        
        # Find the minimum length of phi/psi angles between AF2 and experimental
        af2_phi_psi_len = len(self.af2_state['phi_psi'])
        exp_phi_psi_len = len(self.exp_state['phi_psi'])
        self.min_phi_psi_len = min(af2_phi_psi_len, exp_phi_psi_len)
        
        # Truncate both states to the minimum length
        self.af2_state['phi_psi'] = self.af2_state['phi_psi'][:self.min_phi_psi_len]
        self.exp_state['phi_psi'] = self.exp_state['phi_psi'][:self.min_phi_psi_len]
        
        # Initialize the current state
        self.state = self.af2_state.copy()
        self.state_dim = self.min_phi_psi_len
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-180, high=180, shape=(self.state_dim,), dtype=np.float32)
        
        # Set target state
        self.target_state = self.exp_state
    
    def reset(self):
        self.current_step = 0
        self.state = self.af2_state.copy()
        return self.state['phi_psi']
    
    def step(self, action):
        self.state = apply_action(self.state, action)
        self.current_step += 1

        reward = self._compute_reward(self.state)
        done = self.current_step >= self.max_steps or self._has_converged(self.state)
        info = {
            "step": self.current_step,
            "energy": compute_energy(self.state),
            "rmsd": compute_rmsd(self.state, self.target_state),
            "sasa": compute_sasa(self.state),
            "reward": reward
        }

        # Optional debug log
        print(f"[Step {self.current_step}] Reward: {reward:.3f}, RMSD: {info['rmsd']:.3f}, Energy: {info['energy']:.2f}")
        
        return self.state['phi_psi'], reward, done, info

    def _compute_reward(self, state):
        energy = compute_energy(state)
        rmsd = compute_rmsd(state, self.target_state)
        sasa = compute_sasa(state)
        
        beta_bonus = np.random.uniform(-1, 1)
        clash_penalty = np.random.uniform(0, 1)
        
        total_reward = (0.4 * (-energy) +
                        0.3 * (-rmsd) +
                        0.2 * (-sasa) +
                        0.1 * beta_bonus -
                        0.1 * clash_penalty)
        return total_reward
    
    def _has_converged(self, state):
        return False
    
    def render(self, mode="human"):
        print(f"Step {self.current_step} - Energy: {compute_energy(self.state):.2f}, RMSD: {compute_rmsd(self.state, self.target_state):.2f}")

def get_all_samples(base_dir="data"):
    """
    Scans the data directory and returns a list of (AF2_path, EXP_path) tuples.
    Assumes each sample folder has subdirectories 'AF2' and 'Experimental'.
    """
    samples = []
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        try:
            af2_dir = os.path.join(sample_path, "AF2")
            exp_dir = os.path.join(sample_path, "Experimental")
            af2_file = next(f for f in os.listdir(af2_dir) if f.endswith("_AF2.cif"))
            exp_file = next(f for f in os.listdir(exp_dir) if f.endswith(".cif") and not f.endswith("_AF2.cif"))
            af2_path = os.path.join(af2_dir, af2_file)
            exp_path = os.path.join(exp_dir, exp_file)
            samples.append((af2_path, exp_path))
        except Exception as e:
            print(f"Skipping {sample_folder}: {e}")
    return samples

def extract_and_save_features():
    """
    Extracts features from all AF2 CIF files and saves summary features to CSV.
    Also saves detailed phi/psi angles to a separate CSV file.
    """
    af2_files = []
    base_dir = "data"
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        af2_dir = os.path.join(sample_path, "AF2")
        if not os.path.exists(af2_dir):
            continue
        for file in os.listdir(af2_dir):
            if file.endswith(".cif"):
                af2_files.append((sample_folder, os.path.join(af2_dir, file)))
    
    print(f"Found {len(af2_files)} AF2 CIF files.")
    csv_file = "protein_features.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "File", "Num_Phi_Psi", "Num_Atoms", "Num_Residues", "Avg_B_Factor", "Avg_Hydrophobicity", "Total_SASA"])
        for sample_name, file_path in af2_files:
            try:
                print(f"Processing {file_path}...")
                features = load_structure(file_path)
                num_phi_psi = len(features['phi_psi'])
                num_atoms = len(features['coordinates'])
                num_residues = len(features['residue_ids'])
                avg_b_factor = np.mean(features['b_factors'])
                avg_hydrophobicity = np.mean(features['hydrophobicity'])
                total_sasa = np.sum(features['sasa'])
                writer.writerow([sample_name, os.path.basename(file_path), num_phi_psi, num_atoms, num_residues, f"{avg_b_factor:.2f}", f"{avg_hydrophobicity:.2f}", f"{total_sasa:.2f}"])
                print(f"  - Extracted {num_phi_psi} phi/psi angles, {num_atoms} atoms, {num_residues} residues")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Features saved to {csv_file}")
    save_detailed_angles(af2_files)

def save_detailed_angles(af2_files):
    """
    Saves detailed phi/psi angles for each sample to a CSV file.
    """
    angles_file = "protein_angles.csv"
    with open(angles_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "File", "Residue_ID", "Phi", "Psi"])
        for sample_name, file_path in af2_files:
            try:
                features = load_structure(file_path)
                phi_psi = features['phi_psi']
                residue_ids = features['residue_ids']
                for i in range(0, len(phi_psi), 2):
                    if i+1 < len(phi_psi):
                        writer.writerow([sample_name, os.path.basename(file_path), residue_ids[i//2] if i//2 < len(residue_ids) else "N/A", f"{phi_psi[i]:.2f}", f"{phi_psi[i+1]:.2f}"])
            except Exception as e:
                print(f"Error saving angles for {file_path}: {e}")
    print(f"Detailed angles saved to {angles_file}")

def test_environment_on_sample(af2_path, exp_path, max_steps=10):
    """
    Creates an environment for a given sample and runs a test episode.
    """
    print(f"\nTesting environment for sample:\n AF2: {af2_path}\n EXP: {exp_path}")
    
    try:
        # Load structures first to check for compatibility
        af2_features = load_structure(af2_path)
        exp_features = load_structure(exp_path)
        
        af2_phi_psi_len = len(af2_features['phi_psi'])
        exp_phi_psi_len = len(exp_features['phi_psi'])
        min_len = min(af2_phi_psi_len, exp_phi_psi_len)
        
        print(f"AF2 structure has {af2_phi_psi_len} phi/psi angles")
        print(f"Experimental structure has {exp_phi_psi_len} phi/psi angles")
        print(f"Using the first {min_len} angles for comparison")
        
        # Create and test the environment
        env = AmyloidEnv(af2_path, exp_path, max_steps=max_steps)
        state = env.reset()
        print("Initial state (first 10 angles):", state[:10])
        
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = env.action_space.sample()  # Replace with policy action later
            state, reward, done, info = env.step(action)
            step_count += 1
            print(f"Step {step_count}: Reward={reward:.3f}, RMSD={info['rmsd']:.3f}, Energy={info['energy']:.2f}")
        
        env.render()
        return True
    except Exception as e:
        print(f"Error testing environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting protein feature extraction...")
    extract_and_save_features()
    print("Extraction complete!\n")
    
    # Get all sample pairs for testing the environment
    sample_pairs = get_all_samples("data")
    print(f"Found {len(sample_pairs)} sample pairs for environment testing.\n")
    
    # Test the environment on each sample with a full episode (using random actions)
    for i, (af2_path, exp_path) in enumerate(sample_pairs):
        print(f"--- Testing Sample {i+1} ---")
        try:
            test_environment_on_sample(af2_path, exp_path, max_steps=10)
        except Exception as e:
            print(f"Error testing sample {i+1}: {e}")
