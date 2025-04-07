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
    - phi/psi angles
    - atomic coordinates
    - secondary structure
    - hydrophobicity
    - SASA
    - B-factors
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
                residue_ids.append(pp[i].id[1])  # Store residue ID
    
    # Extract atomic coordinates and B-factors
    coords = []
    b_factors = []
    atom_types = []
    
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
            b_factors.append(atom.get_bfactor())
            atom_types.append(atom.get_id())
    
    # Extract secondary structure (requires DSSP)
    # We'll use a simplified approach since DSSP requires a PDB file
    # For a real implementation, you'd convert CIF to PDB and use DSSP
    
    # Extract hydrophobicity (Kyte-Doolittle scale)
    hydrophobicity = []
    for residue in chain:
        if is_aa(residue):
            try:
                res_code = protein_letters_3to1.get(residue.get_resname(), 'X')
                # Simplified hydrophobicity scale
                h_scale = {
                    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
                }
                hydrophobicity.append(h_scale.get(res_code, 0))
            except:
                hydrophobicity.append(0)
    
    # Calculate a simple SASA approximation
    # In a real implementation, you'd use a proper SASA calculator
    sasa = np.ones(len(residue_ids)) * 100  # Placeholder
    
    # Return all features
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
    # Only modify phi/psi angles
    new_phi_psi = state['phi_psi'] + action
    new_phi_psi = np.clip(new_phi_psi, -180, 180)
    
    # Create a new state with updated phi/psi
    new_state = state.copy()
    new_state['phi_psi'] = new_phi_psi
    
    return new_state

def compute_energy(state):
    # Simplified energy calculation based on phi/psi angles
    return np.sum(np.abs(state['phi_psi']))

def compute_rmsd(state, target_state):
    # Simplified RMSD calculation based on phi/psi angles
    return np.sqrt(np.mean((state['phi_psi'] - target_state['phi_psi']) ** 2))

def compute_sasa(state):
    # Use the pre-calculated SASA
    return np.sum(state['sasa'])

class AmyloidEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, af2_pdb_path, exp_pdb_path, max_steps=50):
        super(AmyloidEnv, self).__init__()
        self.af2_pdb_path = af2_pdb_path
        self.exp_pdb_path = exp_pdb_path
        self.max_steps = max_steps
        self.current_step = 0
        
        self.state = load_structure(self.af2_pdb_path)
        self.state_dim = self.state['phi_psi'].shape[0]
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-180, high=180, shape=(self.state_dim,), dtype=np.float32)
        
        self.target_state = load_structure(self.exp_pdb_path)
    
    def reset(self):
        self.current_step = 0
        self.state = load_structure(self.af2_pdb_path)
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

# --- Helper to find all AF2 CIF files ---
def find_af2_cif_files(base_dir="data"):
    """
    Scans the data directory and returns a list of AF2 CIF file paths.
    """
    af2_files = []
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        
        # Look for AF2 directory
        af2_dir = os.path.join(sample_path, "AF2")
        if not os.path.exists(af2_dir):
            continue
            
        # Find all CIF files in AF2 directory
        for file in os.listdir(af2_dir):
            if file.endswith(".cif"):
                af2_files.append((sample_folder, os.path.join(af2_dir, file)))
    
    return af2_files

# --- Extract and save features to CSV ---
def extract_and_save_features():
    """
    Extract features from all AF2 CIF files and save to CSV
    """
    af2_files = find_af2_cif_files("data")
    print(f"Found {len(af2_files)} AF2 CIF files.")
    
    # Create CSV file
    csv_file = "protein_features.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Sample", "File", "Num_Phi_Psi", "Num_Atoms", "Num_Residues",
            "Avg_B_Factor", "Avg_Hydrophobicity", "Total_SASA"
        ])
        
        # Process each file
        for sample_name, file_path in af2_files:
            try:
                print(f"Processing {file_path}...")
                features = load_structure(file_path)
                
                # Calculate summary statistics
                num_phi_psi = len(features['phi_psi'])
                num_atoms = len(features['coordinates'])
                num_residues = len(features['residue_ids'])
                avg_b_factor = np.mean(features['b_factors'])
                avg_hydrophobicity = np.mean(features['hydrophobicity'])
                total_sasa = np.sum(features['sasa'])
                
                # Write to CSV
                writer.writerow([
                    sample_name,
                    os.path.basename(file_path),
                    num_phi_psi,
                    num_atoms,
                    num_residues,
                    f"{avg_b_factor:.2f}",
                    f"{avg_hydrophobicity:.2f}",
                    f"{total_sasa:.2f}"
                ])
                
                print(f"  - Extracted {num_phi_psi} phi/psi angles, {num_atoms} atoms, {num_residues} residues")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Features saved to {csv_file}")
    
    # Also save detailed phi/psi angles to a separate file
    save_detailed_angles(af2_files)

def save_detailed_angles(af2_files):
    """
    Save detailed phi/psi angles to a separate file
    """
    angles_file = "protein_angles.csv"
    with open(angles_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Sample", "File", "Residue_ID", "Phi", "Psi"])
        
        # Process each file
        for sample_name, file_path in af2_files:
            try:
                features = load_structure(file_path)
                
                # Extract phi/psi pairs
                phi_psi = features['phi_psi']
                residue_ids = features['residue_ids']
                
                # Write each phi/psi pair
                for i in range(0, len(phi_psi), 2):
                    if i+1 < len(phi_psi):
                        writer.writerow([
                            sample_name,
                            os.path.basename(file_path),
                            residue_ids[i//2] if i//2 < len(residue_ids) else "N/A",
                            f"{phi_psi[i]:.2f}",
                            f"{phi_psi[i+1]:.2f}"
                        ])
                
            except Exception as e:
                print(f"Error saving angles for {file_path}: {e}")
    
    print(f"Detailed angles saved to {angles_file}")

# --- Run extraction ---
if __name__ == "__main__":
    print("Starting protein feature extraction...")
    extract_and_save_features()
    print("Extraction complete!")
