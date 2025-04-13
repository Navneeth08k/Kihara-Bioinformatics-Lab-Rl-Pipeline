import os
import random
import numpy as np
import csv
import glob
import freesasa

import gymnasium as gym
from gymnasium import spaces
from Bio.PDB import MMCIFParser, PPBuilder, DSSP, Polypeptide
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import *

# --- Helper: Pad/Truncate φ/ψ Vector ---
def pad_phi_psi(vec, fixed_dim):
    """
    Pads vec with zeros or truncates it so that it has length fixed_dim.
    """
    if len(vec) < fixed_dim:
        padded = np.concatenate([vec, np.zeros(fixed_dim - len(vec))])
    else:
        padded = vec[:fixed_dim]
    return padded

# --- Data Extraction and Feature Calculation ---
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
    
    # Extract φ/ψ angles using PPBuilder.
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
    Applies an action by modifying only the φ/ψ angles.
    The action is added elementwise, and the resulting angles are clipped between -180 and 180.
    """
    new_phi_psi = state['phi_psi'] + action
    new_phi_psi = np.clip(new_phi_psi, -180, 180)
    new_state = state.copy()
    new_state['phi_psi'] = new_phi_psi
    return new_state

# --- Reward and Metric Functions ---
def compute_energy(state):
    """
    Computes an energy proxy based on the deviation from ideal beta-sheet dihedrals.
    Assumes ideal beta-sheet angles: φ=-120 and ψ=120.
    """
    phi_psi = state['phi_psi']
    ideal = np.array([-120, 120] * (len(phi_psi)//2), dtype=np.float32)
    k = 0.01
    return 0.5 * k * np.sum((phi_psi - ideal)**2)

def compute_rmsd(state, target_state):
    """
    Computes the RMSD between the φ/ψ vectors of the current and target states.
    If lengths differ, uses only the minimum length.
    """
    phi1 = state['phi_psi']
    phi2 = target_state['phi_psi']
    min_len = min(len(phi1), len(phi2))
    diff = phi1[:min_len] - phi2[:min_len]
    return np.sqrt(np.mean(diff**2))

def compute_sasa(state, pdb_path=None):
    """
    Computes SASA.
    Returns the placeholder SASA from the state (or a real calculation if implemented).
    """
    if pdb_path is not None:
        return None  # Replace with a real SASA calculation if available.
    else:
        return np.sum(state['sasa'])

def compute_beta_sheet_fraction(phi_psi):
    """
    Estimates the fraction of residues in beta-sheet conformation based on φ/ψ.
    Assumes beta-sheet dihedrals are: φ in [-150, -90] and ψ in [90, 150].
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
    Computes a simplistic clash penalty based on atomic coordinates.
    For every pair of atoms closer than 2.0 Å, adds a penalty of (2.0 - distance).
    """
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
    """
    Computes the total reward as a weighted sum of:
      - Energy: deviation from ideal beta-sheet dihedrals.
      - RMSD: difference between current and target φ/ψ vectors.
      - SASA: total solvent-accessible surface area.
      - Beta-sheet fraction: similarity of beta-sheet content.
      - Clash penalty: penalizes atomic clashes.
    """
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

# --- Environment Definition ---
class AmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, af2_pdb_path, exp_pdb_path=None, max_steps=50, fixed_dim=None, inference_mode=False):
        """
        Environment for refining AF2 predictions.
        
        Parameters:
          af2_pdb_path: Path to the AF2 prediction CIF file.
          exp_pdb_path: Path to the experimental CIF file (for training). If None and inference_mode is True, it's ignored.
          max_steps: Maximum number of steps per episode.
          fixed_dim: The fixed length for the φ/ψ vector (if None, uses the length from loaded structure).
          inference_mode: If True, no experimental structure is required and target state is set to the initial state.
        """
        super(AmyloidEnv, self).__init__()
        self.af2_pdb_path = af2_pdb_path
        self.inference_mode = inference_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Set render_mode and a dummy EnvSpec.
        self.render_mode = None
        self.spec = type('EnvSpec', (), {"id": "AmyloidEnv-v0"})()

        # Load initial state and determine fixed dimension.
        initial_state = load_structure(self.af2_pdb_path)
        self.fixed_dim = len(initial_state['phi_psi']) if fixed_dim is None else fixed_dim

        # Load current state and pad/truncate φ/ψ.
        self.state = load_structure(self.af2_pdb_path)
        self.state['phi_psi'] = pad_phi_psi(self.state['phi_psi'], self.fixed_dim)
        self.state_dim = self.fixed_dim

        if not self.inference_mode:
            if exp_pdb_path is None:
                raise ValueError("For training mode, an experimental CIF must be provided.")
            self.target_state = load_structure(exp_pdb_path)
            self.target_state['phi_psi'] = pad_phi_psi(self.target_state['phi_psi'], self.fixed_dim)
        else:
            # In inference mode, target_state is not used (set it to the initial state).
            self.target_state = self.state.copy()
        
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
        if not self.inference_mode:
            reward = refined_compute_reward(self.state, self.target_state)
        else:
            reward = 0.0  # During inference, we do not compute a reward.
        terminated = False  # You can implement a convergence check here.
        truncated = self.current_step >= self.max_steps
        info = {
            "step": self.current_step,
            "energy": compute_energy(self.state),
            "rmsd": compute_rmsd(self.state, self.target_state) if not self.inference_mode else None,
            "sasa": compute_sasa(self.state),
            "reward": reward
        }
        print(f"[Step {self.current_step}] Reward: {reward:.3f}, RMSD: {info['rmsd']}, Energy: {info['energy']:.2f}, SASA: {info['sasa']:.2f}")
        return self.state['phi_psi'], reward, terminated, truncated, info
    
    def render(self, mode="human"):
        print(f"Step {self.current_step} - Energy: {compute_energy(self.state):.2f}, "
              f"RMSD: {compute_rmsd(self.state, self.target_state):.2f}, "
              f"SASA: {compute_sasa(self.state):.2f}")

# --- Meta-Environment for Multiple Samples ---
class MetaAmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, sample_pairs, max_steps=50, fixed_dim=None, inference_mode=False):
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
        temp_env = AmyloidEnv(example_af2, example_exp, max_steps=self.max_steps, fixed_dim=self.fixed_dim, inference_mode=inference_mode)
        obs, _ = temp_env.reset()
        self.observation_space = spaces.Box(low=-180, high=180, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=obs.shape, dtype=np.float32)
        self.inference_mode = inference_mode
    
    def reset(self, seed=None, options=None):
        chosen_pair = random.choice(self.sample_pairs)
        self.current_env = AmyloidEnv(chosen_pair[0], chosen_pair[1],
                                       max_steps=self.max_steps,
                                       fixed_dim=self.fixed_dim,
                                       inference_mode=self.inference_mode)
        return self.current_env.reset(seed=seed, options=options)
    
    def step(self, action):
        return self.current_env.step(action)
    
    def render(self, mode="human"):
        return self.current_env.render(mode)
