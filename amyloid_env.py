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
import freesasa
from rebuild_structure import rebuild_structure_from_angles

### Data Extraction and Feature Calculation

def load_structure(cif_path):
    """
    Loads a CIF file and returns a dictionary of protein features:
      - phi/psi angles (state representation)
      - atomic coordinates 
      - B-factors
      - hydrophobicity
      - SASA (placeholder, constant value per residue)
      - Residue IDs
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    
    model = structure[0]
    chain = next(model.get_chains())
    
    # Extract phi/psi angles using PPBuilder.
    ppb = PPBuilder()
    phi_psi_angles = []
    residue_ids = []
    for pp in ppb.build_peptides(chain):
        phi_psi = pp.get_phi_psi_list()
        for i, (phi, psi) in enumerate(phi_psi):
            if phi is not None and psi is not None:
                phi_psi_angles.extend([np.degrees(phi), np.degrees(psi)])
                residue_ids.append(pp[i].id[1])
    
    # Extract atomic coordinates, B-factors, and atom types.
    coords = []
    b_factors = []
    atom_types = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
            b_factors.append(atom.get_bfactor())
            atom_types.append(atom.get_id())
    
    # Extract hydrophobicity using a simplified Kyte-Doolittle scale.
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
    
    # Placeholder SASA: assign a constant value (100) per residue.
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
    Applies an action by modifying only the phi/ψ angles.
    The action is added elementwise, and angles are clipped between -180 and 180.
    """
    new_phi_psi = state['phi_psi'] + action
    new_phi_psi = np.clip(new_phi_psi, -180, 180)
    new_state = state.copy()
    new_state['phi_psi'] = new_phi_psi
    return new_state

### Reward and Metric Functions

def compute_energy(state):
    """
    Computes an energy proxy based on the deviation from ideal beta-sheet dihedrals.
    Here, we assume ideal beta-sheet angles of phi = -120 and psi = 120.
    A harmonic penalty is applied.
    """
    phi_psi = state['phi_psi']
    ideal = np.array([-120, 120] * (len(phi_psi)//2), dtype=np.float32)
    k = 0.01  # arbitrary stiffness constant
    energy = 0.5 * k * np.sum((phi_psi - ideal)**2)
    return energy

def compute_rmsd(state, target_state):
    """
    Computes the RMSD between the phi/ψ angle vectors of the current and target states.
    If lengths differ, compares only up to the minimum length.
    """
    phi1 = state['phi_psi']
    phi2 = target_state['phi_psi']
    min_len = min(len(phi1), len(phi2))
    diff = phi1[:min_len] - phi2[:min_len]
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd

def compute_sasa(state, pdb_path=None):
    """
    Computes SASA.
    If a pdb_path is provided, a real SASA calculation (compute_sasa_real) may be invoked.
    Otherwise, returns the placeholder SASA value from the state.
    """
    if pdb_path is not None:
        return compute_sasa_real(pdb_path)
    else:
        return np.sum(state['sasa'])

def compute_sasa_real(pdb_path):
    """
    Computes the solvent-accessible surface area using FreeSASA.
    Expects pdb_path to be a valid PDB file.
    """
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        return result.totalArea()
    except Exception as e:
        print(f"Error computing SASA with freesasa on {pdb_path}: {e}")
        return 0.0

def compute_beta_sheet_fraction(phi_psi):
    """
    Estimates the fraction of residues in a beta-sheet conformation based on phi/ψ.
    Typical beta-sheet dihedrals: phi between -150 and -90, psi between 90 and 150.
    """
    length = len(phi_psi) - (len(phi_psi) % 2)
    count = 0
    total = length // 2
    for i in range(0, length, 2):
        phi = phi_psi[i]
        psi = phi_psi[i+1]
        if -150 <= phi <= -90 and 90 <= psi <= 150:
            count += 1
    return count / total if total > 0 else 0

def compute_clash_penalty(state):
    """
    Computes a simplistic clash penalty from atomic coordinates.
    For each pair of atoms closer than 2.0 Å, adds a penalty proportional to (2.0 - distance).
    (O(n^2) approach for demonstration.)
    """
    coords = state['coordinates']
    penalty = 0.0
    threshold = 2.0  # in Å
    n = len(coords)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                penalty += (threshold - dist)
    return penalty

def refined_compute_reward(state, target_state):
    """
    Computes the total reward using a weighted sum of:
      - Energy: deviation from ideal beta-sheet dihedrals.
      - RMSD: difference between current and target phi/ψ vectors.
      - SASA: total solvent-accessible surface area.
      - Beta-sheet fraction: similarity between current and target beta-sheet content.
      - Clash penalty: penalizing atomic clashes.
    Weights are arbitrarily chosen for tuning.
    """
    energy = compute_energy(state)
    rmsd = compute_rmsd(state, target_state)
    sasa = compute_sasa(state)  # using placeholder SASA
    
    phi_current = state['phi_psi']
    phi_target = target_state['phi_psi']
    min_len = min(len(phi_current), len(phi_target))
    beta_target = compute_beta_sheet_fraction(phi_target[:min_len])
    beta_current = compute_beta_sheet_fraction(phi_current[:min_len])
    beta_bonus = -abs(beta_target - beta_current)
    
    clash_penalty = compute_clash_penalty(state)
    
    total_reward = (0.4 * (-energy) +
                    0.3 * (-rmsd) +
                    0.2 * (-sasa) +
                    0.1 * beta_bonus -
                    0.05 * clash_penalty)
    return total_reward

### Environment Definition

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
    
    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility if needed.
        super().reset(seed=seed)
        self.current_step = 0
        self.state = load_structure(self.af2_pdb_path)
        # Return observation and an empty info dict per Gymnasium API.
        return self.state['phi_psi'], {}
    
    def step(self, action):
        self.state = apply_action(self.state, action)
        self.current_step += 1

        reward = self._compute_reward(self.state)
        terminated = self._has_converged(self.state)
        truncated = self.current_step >= self.max_steps
        info = {
            "step": self.current_step,
            "energy": compute_energy(self.state),
            "rmsd": compute_rmsd(self.state, self.target_state),
            "sasa": compute_sasa(self.state),
            "reward": reward
        }

        print(f"[Step {self.current_step}] Reward: {reward:.3f}, RMSD: {info['rmsd']:.3f}, Energy: {info['energy']:.2f}, SASA: {info['sasa']:.2f}")
        
        # Return five values as expected by Gymnasium: observation, reward, terminated, truncated, info
        return self.state['phi_psi'], reward, terminated, truncated, info


    def _compute_reward(self, state):
        return refined_compute_reward(state, self.target_state)
    
    def _has_converged(self, state):
        # You can add a convergence criterion based on RMSD or changes in state.
        return False
    
    def render(self, mode="human"):
        print(f"Step {self.current_step} - Energy: {compute_energy(self.state):.2f}, RMSD: {compute_rmsd(self.state, self.target_state):.2f}, SASA: {compute_sasa(self.state):.2f}")

### Helper Functions for Data Handling and Testing

def get_all_samples(base_dir="data"):
    """
    Scans the data directory and returns a list of (AF2_path, EXP_path) tuples.
    Assumes each sample folder has subdirectories "AF2" and "Experimental".
    Instead of looking for files with a "_AF2.cif" suffix, it takes the first .cif file found in the AF2 folder.
    """
    samples = []
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        try:
            af2_dir = os.path.join(sample_path, "AF2")
            exp_dir = os.path.join(sample_path, "Experimental")
            # Get all CIF files in the AF2 directory
            af2_files = [f for f in os.listdir(af2_dir) if f.endswith(".cif")]
            if not af2_files:
                raise ValueError("No CIF files found in AF2 directory")
            # Choose the first CIF file (or implement your own selection criteria)
            af2_file = af2_files[0]
            # Get all CIF files in the Experimental directory (excluding those meant for AF2 if any)
            exp_files = [f for f in os.listdir(exp_dir) if f.endswith(".cif")]
            if not exp_files:
                raise ValueError("No CIF files found in Experimental directory")
            exp_file = exp_files[0]
            af2_path = os.path.join(af2_dir, af2_file)
            exp_path = os.path.join(exp_dir, exp_file)
            samples.append((af2_path, exp_path))
        except Exception as e:
            print(f"Skipping {sample_folder}: {e}")
    return samples


def extract_and_save_features():
    """
    Extracts features from all AF2 CIF files and saves summary features to CSV.
    Also saves detailed phi/ψ angles to a separate CSV file.
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
                print(f"  - Extracted {num_phi_psi} phi/ψ angles, {num_atoms} atoms, {num_residues} residues")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Features saved to {csv_file}")
    save_detailed_angles(af2_files)

def save_detailed_angles(af2_files):
    """
    Saves detailed phi/ψ angles for each sample to a CSV file.
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
                        writer.writerow([sample_name, os.path.basename(file_path),
                                         residue_ids[i//2] if i//2 < len(residue_ids) else "N/A",
                                         f"{phi_psi[i]:.2f}", f"{phi_psi[i+1]:.2f}"])
            except Exception as e:
                print(f"Error saving angles for {file_path}: {e}")
    print(f"Detailed angles saved to {angles_file}")

def test_environment_on_sample(af2_path, exp_path, max_steps=10, output_dir="refined_outputs"):
    """
    Creates an environment for a given sample, runs a test episode,
    saves the refined phi/psi angles to CSV, and rebuilds the 3D structure.
    """
    print(f"\nTesting environment for sample:\n AF2: {af2_path}\n EXP: {exp_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load structures and setup environment
        env = AmyloidEnv(af2_path, exp_path, max_steps=max_steps)
        state = env.reset()
        print("Initial state (first 10 angles):", state[:10])
        
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = env.action_space.sample()  # Later: use learned PPO policy here
            state, reward, done, info = env.step(action)
            step_count += 1

        # === Save final phi/psi angles to CSV ===
        refined_angles = state  # this is just the phi/psi vector
        sample_name = os.path.splitext(os.path.basename(af2_path))[0]
        angle_csv = os.path.join(output_dir, f"{sample_name}_refined_angles.csv")

        with open(angle_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Residue_Index", "Phi", "Psi"])
            for i in range(0, len(refined_angles), 2):
                if i + 1 < len(refined_angles):
                    writer.writerow([i // 2 + 1, refined_angles[i], refined_angles[i + 1]])
        print(f"✅ Saved refined phi/psi angles to: {angle_csv}")

        # === Rebuild 3D structure ===
        output_pdb_path = os.path.join(output_dir, f"{sample_name}_reconstructed.pdb")
        rebuild_structure_from_angles(af2_path, refined_angles, output_pdb_path)
        print(f"✅ Reconstructed structure saved to: {output_pdb_path}")

        env.render()
        return True

    except Exception as e:
        print(f"❌ Error testing environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

### Main Script

if __name__ == "__main__":
    print("Starting protein feature extraction...")
    extract_and_save_features()
    print("Extraction complete!\n")
    
    # Get sample pairs for testing the environment
    sample_pairs = get_all_samples("data")
    print(f"Found {len(sample_pairs)} sample pairs for environment testing.\n")
    
    # Test the environment on each sample with a full episode (using random actions)
    for i, (af2_path, exp_path) in enumerate(sample_pairs):
        print(f"--- Testing Sample {i+1} ---")
        try:
            test_environment_on_sample(af2_path, exp_path, max_steps=10)
        except Exception as e:
            print(f"Error testing sample {i+1}: {e}")
