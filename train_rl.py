import os
import random
import numpy as np
import csv
import glob
import freesasa

import gymnasium as gym
from gymnasium import spaces
from Bio.PDB import MMCIFParser, PPBuilder, DSSP, Polypeptide
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa

# ==============================
# HELPER: Pad/Truncate the φ/ψ Vector
# ==============================

def pad_phi_psi(vec, fixed_dim):
    """
    Pads vec with zeros or truncates it so that it has length fixed_dim.
    """
    if len(vec) < fixed_dim:
        padded = np.concatenate([vec, np.zeros(fixed_dim - len(vec))])
    else:
        padded = vec[:fixed_dim]
    return padded

# ==============================
# DATA EXTRACTION & FEATURE FUNCTIONS
# ==============================

def load_structure(cif_path):
    """
    Loads a CIF file and returns a dictionary of protein features:
      - φ/ψ angles (state representation)
      - atomic coordinates 
      - B-factors
      - hydrophobicity
      - SASA (placeholder: constant value per residue)
      - Residue IDs
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    model = structure[0]
    chain = next(model.get_chains())
    
    ppb = PPBuilder()
    phi_psi_angles = []
    residue_ids = []
    for pp in ppb.build_peptides(chain):
        phi_psi = pp.get_phi_psi_list()
        for i, (phi, psi) in enumerate(phi_psi):
            if phi is not None and psi is not None:
                phi_psi_angles.extend([np.degrees(phi), np.degrees(psi)])
                residue_ids.append(pp[i].id[1])
    
    coords = []
    b_factors = []
    atom_types = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
            b_factors.append(atom.get_bfactor())
            atom_types.append(atom.get_id())
    
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
    Applies an action by modifying only the φ/ψ angles.
    """
    new_phi_psi = state['phi_psi'] + action
    new_phi_psi = np.clip(new_phi_psi, -180, 180)
    new_state = state.copy()
    new_state['phi_psi'] = new_phi_psi
    return new_state

# ==============================
# REWARD AND METRIC FUNCTIONS
# ==============================

def compute_energy(state):
    phi_psi = state['phi_psi']
    ideal = np.array([-120, 120] * (len(phi_psi)//2), dtype=np.float32)
    k = 0.01
    return 0.5 * k * np.sum((phi_psi - ideal)**2)

def compute_rmsd(state, target_state):
    phi1 = state['phi_psi']
    phi2 = target_state['phi_psi']
    min_len = min(len(phi1), len(phi2))
    diff = phi1[:min_len] - phi2[:min_len]
    return np.sqrt(np.mean(diff**2))

def compute_sasa(state, pdb_path=None):
    if pdb_path is not None:
        return compute_sasa_real(pdb_path)
    else:
        return np.sum(state['sasa'])

def compute_sasa_real(pdb_path):
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        return result.totalArea()
    except Exception as e:
        print(f"Error computing SASA with freesasa on {pdb_path}: {e}")
        return 0.0

def compute_beta_sheet_fraction(phi_psi):
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
    coords = state['coordinates']
    penalty = 0.0
    threshold = 2.0
    n = len(coords)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                penalty += (threshold - dist)
    return penalty

def refined_compute_reward(state, target_state):
    energy = compute_energy(state)
    rmsd = compute_rmsd(state, target_state)
    sasa = compute_sasa(state)
    
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

# ==============================
# ENVIRONMENT DEFINITION
# ==============================

class AmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, af2_pdb_path, exp_pdb_path, max_steps=50, fixed_dim=None):
        super(AmyloidEnv, self).__init__()
        self.af2_pdb_path = af2_pdb_path
        self.exp_pdb_path = exp_pdb_path
        self.max_steps = max_steps
        self.current_step = 0

        # Set render_mode and spec to satisfy gym wrappers.
        self.render_mode = None
        self.spec = type('EnvSpec', (), {"id": "AmyloidEnv-v0"})()

        initial_state = load_structure(self.af2_pdb_path)
        if fixed_dim is None:
            self.fixed_dim = len(initial_state['phi_psi'])
        else:
            self.fixed_dim = fixed_dim
        
        self.state = load_structure(self.af2_pdb_path)
        self.state['phi_psi'] = pad_phi_psi(self.state['phi_psi'], self.fixed_dim)
        self.state_dim = self.fixed_dim
        
        self.target_state = load_structure(self.exp_pdb_path)
        self.target_state['phi_psi'] = pad_phi_psi(self.target_state['phi_psi'], self.fixed_dim)
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.fixed_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-180, high=180, shape=(self.fixed_dim,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = load_structure(self.af2_pdb_path)
        self.state['phi_psi'] = pad_phi_psi(self.state['phi_psi'], self.fixed_dim)
        return self.state['phi_psi'], {}
    
    def step(self, action):
        self.state = apply_action(self.state, action)
        self.state['phi_psi'] = pad_phi_psi(self.state['phi_psi'], self.fixed_dim)
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
        return self.state['phi_psi'], reward, terminated, truncated, info
    
    def _compute_reward(self, state):
        return refined_compute_reward(state, self.target_state)
    
    def _has_converged(self, state):
        return False
    
    def render(self, mode="human"):
        print(f"Step {self.current_step} - Energy: {compute_energy(self.state):.2f}, RMSD: {compute_rmsd(self.state, self.target_state):.2f}, SASA: {compute_sasa(self.state):.2f}")

# ==============================
# META-ENVIRONMENT (Multiple Samples)
# ==============================

class MetaAmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, sample_pairs, max_steps=50, fixed_dim=None):
        super(MetaAmyloidEnv, self).__init__()
        self.sample_pairs = sample_pairs
        self.max_steps = max_steps
        self.current_env = None

        # Set attributes for compatibility with vectorized wrappers.
        self.num_envs = 1
        self.render_mode = None
        self.spec = type('EnvSpec', (), {"id": "MetaAmyloidEnv-v0"})()

        if fixed_dim is None:
            dims = []
            for af2_path, _ in self.sample_pairs:
                s = load_structure(af2_path)
                dims.append(len(s['phi_psi']))
            self.fixed_dim = max(dims)
        else:
            self.fixed_dim = fixed_dim

        example_af2, example_exp = self.sample_pairs[0]
        temp_env = AmyloidEnv(example_af2, example_exp, max_steps=self.max_steps, fixed_dim=self.fixed_dim)
        obs, _ = temp_env.reset()
        self.observation_space = spaces.Box(low=-180, high=180, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=obs.shape, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        chosen_pair = random.choice(self.sample_pairs)
        self.current_env = AmyloidEnv(chosen_pair[0], chosen_pair[1], max_steps=self.max_steps, fixed_dim=self.fixed_dim)
        return self.current_env.reset(seed=seed, options=options)
    
    def step(self, action):
        return self.current_env.step(action)
    
    def render(self, mode="human"):
        return self.current_env.render(mode)

# ==============================
# HELPER FUNCTIONS FOR DATA HANDLING
# ==============================

def get_all_samples(base_dir="data"):
    samples = []
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        try:
            af2_dir = os.path.join(sample_path, "AF2")
            exp_dir = os.path.join(sample_path, "Experimental")
            af2_files = [f for f in os.listdir(af2_dir) if f.endswith(".cif")]
            exp_files = [f for f in os.listdir(exp_dir) if f.endswith(".cif")]
            if not af2_files or not exp_files:
                raise ValueError("Missing CIF file in one of the required folders.")
            af2_path = os.path.join(af2_dir, af2_files[0])
            exp_path = os.path.join(exp_dir, exp_files[0])
            samples.append((af2_path, exp_path))
        except Exception as e:
            print(f"Skipping {sample_folder}: {e}")
    return samples

def split_samples(samples, test_fraction=0.2):
    random.shuffle(samples)
    n_test = max(1, int(len(samples) * test_fraction))
    test_samples = samples[:n_test]
    train_samples = samples[n_test:]
    return train_samples, test_samples

def extract_and_save_features():
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
                writer.writerow([sample_name, os.path.basename(file_path), num_phi_psi, num_atoms, num_residues,
                                 f"{avg_b_factor:.2f}", f"{avg_hydrophobicity:.2f}", f"{total_sasa:.2f}"])
                print(f"  - Extracted {num_phi_psi} φ/ψ angles, {num_atoms} atoms, {num_residues} residues")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Features saved to {csv_file}")
    save_detailed_angles(af2_files)

def save_detailed_angles(af2_files):
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

def test_environment_on_sample(af2_path, exp_path, max_steps=10):
    print(f"\nTesting environment for sample:\n  AF2: {af2_path}\n  EXP: {exp_path}")
    try:
        af2_features = load_structure(af2_path)
        exp_features = load_structure(exp_path)
        af2_phi_psi_len = len(af2_features['phi_psi'])
        exp_phi_psi_len = len(exp_features['phi_psi'])
        min_len = min(af2_phi_psi_len, exp_phi_psi_len)
        print(f"AF2 structure has {af2_phi_psi_len} φ/ψ angles")
        print(f"Experimental structure has {exp_phi_psi_len} φ/ψ angles")
        print(f"Using the first {min_len} angles for comparison")
        
        env = AmyloidEnv(af2_path, exp_path, max_steps=max_steps, fixed_dim=100)
        state, _ = env.reset()
        print("Initial state (first 10 angles):", state[:10])
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            print(f"Step {step_count}: Reward={reward:.3f}, RMSD={info['rmsd']:.3f}, Energy={info['energy']:.2f}, SASA={info['sasa']:.2f}")
        env.render()
        return True
    except Exception as e:
        print(f"Error testing environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ==============================
# TRAINING LOOP WITH META-ENVIRONMENT
# ==============================

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

class MetaAmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, sample_pairs, max_steps=50, fixed_dim=None):
        super(MetaAmyloidEnv, self).__init__()
        self.sample_pairs = sample_pairs
        self.max_steps = max_steps
        self.current_env = None
        
        self.num_envs = 1
        self.render_mode = None
        self.spec = type('EnvSpec', (), {"id": "MetaAmyloidEnv-v0"})()
        
        if fixed_dim is None:
            dims = []
            for af2_path, _ in self.sample_pairs:
                s = load_structure(af2_path)
                dims.append(len(s['phi_psi']))
            self.fixed_dim = max(dims)
        else:
            self.fixed_dim = fixed_dim
        
        example_af2, example_exp = self.sample_pairs[0]
        temp_env = AmyloidEnv(example_af2, example_exp, max_steps=self.max_steps, fixed_dim=self.fixed_dim)
        obs, _ = temp_env.reset()
        self.observation_space = spaces.Box(low=-180, high=180, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=obs.shape, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        chosen_pair = random.choice(self.sample_pairs)
        self.current_env = AmyloidEnv(chosen_pair[0], chosen_pair[1], max_steps=self.max_steps, fixed_dim=self.fixed_dim)
        return self.current_env.reset(seed=seed, options=options)
    
    def step(self, action):
        return self.current_env.step(action)
    
    def render(self, mode="human"):
        return self.current_env.render(mode)

def main():
    samples = get_all_samples("data")
    if len(samples) < 10:
        print(f"Warning: Expected at least 10 sample pairs, found {len(samples)}. Exiting.")
        return
    
    random.shuffle(samples)
    samples = samples[:10]
    train_samples, test_samples = split_samples(samples, test_fraction=0.2)
    
    print("Train Samples:")
    for pair in train_samples:
        print(pair)
    print("Test Samples:")
    for pair in test_samples:
        print(pair)
    
    env = MetaAmyloidEnv(train_samples, max_steps=50, fixed_dim=100)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    
    def make_test_env():
        chosen_pair = random.choice(test_samples)
        return AmyloidEnv(chosen_pair[0], chosen_pair[1], max_steps=50, fixed_dim=100)
    test_env = DummyVecEnv([make_test_env])
    
    checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./checkpoints/', name_prefix='ppo_amyloid')
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/',
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, clip_range=0.2)
    total_timesteps = 1000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    
    model.save("ppo_amyloid_final")
    print("Training complete. Model saved as 'ppo_amyloid_final'.")

if __name__ == "__main__":
    print("Starting protein feature extraction...")
    extract_and_save_features()
    print("Extraction complete!\n")
    
    sample_pairs = get_all_samples("data")
    print(f"Found {len(sample_pairs)} sample pairs in total.\n")
    
    train_samples, test_samples = split_samples(sample_pairs, test_fraction=0.2)
    print("Training and testing sample split:")
    print("Train Samples:")
    for pair in train_samples:
        print(pair)
    print("Test Samples:")
    for pair in test_samples:
        print(pair)
    
    for i, (af2_path, exp_path) in enumerate(sample_pairs):
        print(f"--- Testing Sample {i+1} ---")
        try:
            test_environment_on_sample(af2_path, exp_path, max_steps=10)
        except Exception as e:
            print(f"Error testing sample {i+1}: {e}")
    
    main()
