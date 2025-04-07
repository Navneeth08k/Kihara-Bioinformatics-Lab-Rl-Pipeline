import gymnasium as gym
from gymnasium import spaces
from Bio.PDB import MMCIFParser, PPBuilder
import numpy as np
import os

def load_structure(cif_path):
    """
    Loads a CIF file and returns a vector of phi/psi angles (in degrees).
    Only uses BioPython's MMCIFParser and assumes one chain.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    
    model = structure[0]
    chain = next(model.get_chains())  # You can change this to select a specific chain if needed
    
    ppb = PPBuilder()
    phi_psi_angles = []
    
    for pp in ppb.build_peptides(chain):
        phi_psi = pp.get_phi_psi_list()
        for phi, psi in phi_psi:
            if phi is not None and psi is not None:
                phi_psi_angles.extend([np.degrees(phi), np.degrees(psi)])
    
    return np.array(phi_psi_angles, dtype=np.float32)

def apply_action(state, action):
    new_state = state + action
    return np.clip(new_state, -180, 180)

def compute_energy(state):
    return np.sum(np.abs(state))

def compute_rmsd(state, target_state):
    return np.sqrt(np.mean((state - target_state) ** 2))

def compute_sasa(state):
    return np.sum(np.abs(state)) * 0.01

class AmyloidEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, af2_pdb_path, exp_pdb_path, max_steps=50):
        super(AmyloidEnv, self).__init__()
        self.af2_pdb_path = af2_pdb_path
        self.exp_pdb_path = exp_pdb_path
        self.max_steps = max_steps
        self.current_step = 0
        
        self.state = load_structure(self.af2_pdb_path)
        self.state_dim = self.state.shape[0]
        
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-180, high=180, shape=(self.state_dim,), dtype=np.float32)
        
        self.target_state = load_structure(self.exp_pdb_path)
    
    def reset(self):
        self.current_step = 0
        self.state = load_structure(self.af2_pdb_path)
        return self.state
    
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
        
        return self.state, reward, done, info

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

# --- Helper to find all sample paths ---
def get_all_samples(base_dir="data"):
    """
    Scans the data directory and returns a list of (AF2_path, EXP_path) tuples.
    Assumes structure: data/Sample N/AF2/{ID}_AF2.cif and Experimental/{ID}.cif
    """
    samples = []
    for sample_folder in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        try:
            sample_id = sample_folder.split(" ")[1]  # Extract number (e.g., "1" from "Sample 1")
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

# --- Run Test on All Samples ---
if __name__ == "__main__":
    sample_pairs = get_all_samples("data")
    print(f"Found {len(sample_pairs)} samples.")

    for i, (af2_path, exp_path) in enumerate(sample_pairs):
        print(f"\n--- Sample {i+1} ---")
        print("AF2:", af2_path)
        print("EXP:", exp_path)
        try:
            af2_angles = load_structure(af2_path)
            exp_angles = load_structure(exp_path)
            print("AF2 angles shape:", af2_angles.shape)
            print("EXP angles shape:", exp_angles.shape)
            env = AmyloidEnv(af2_path, exp_path)
            obs = env.reset()
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            print("Next obs (first 5 angles):", next_obs[:5])
            print("Reward:", reward)
            env.render()
        except Exception as e:
            print(f"Error with sample {i+1}: {e}")
