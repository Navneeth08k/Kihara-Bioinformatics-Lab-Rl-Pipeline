import os
import csv
import numpy as np
from stable_baselines3 import PPO

# Import the environment and helper functions from amyloid_env.py
from amyloid_env import AmyloidEnv, load_structure, pad_phi_psi

# Import your rebuild module (ensure this module is implemented)
from rebuild_structure import rebuild_structure_from_angles

def run_inference(af2_cif_path, max_steps=50, fixed_dim=100):
    """
    Runs inference on a new AlphaFold2 CIF file in inference mode.
    The environment will output refined φ/ψ angles, which are then saved in a CSV file
    with columns: Sample, Residue_ID, Phi, Psi.
    
    Parameters:
      af2_cif_path (str): Path to the AlphaFold2 prediction CIF file.
      max_steps (int): Maximum number of steps to run the policy.
      fixed_dim (int): The fixed dimension of the φ/ψ vector.
      
    Returns:
      np.ndarray: The refined φ/ψ angle vector.
    """
    # Create the environment in inference mode (no experimental target)
    env = AmyloidEnv(af2_cif_path, exp_pdb_path=None, max_steps=max_steps, fixed_dim=fixed_dim, inference_mode=True)
    
    # Load the trained PPO model (ensure "ppo_amyloid_final.zip" exists)
    model = PPO.load("ppo_amyloid_final", env=env)
    
    # Reset the environment
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        print(f"Inference Step {step}: Reward={reward:.3f}, Info: {info}")
    
    refined_angles = obs
    print("Refined φ/ψ angles (full vector):", refined_angles)
    
    # Load the original structure to get residue IDs (un-padded real length)
    original = load_structure(af2_cif_path)
    actual_length = len(original['phi_psi'])  # actual number of angles
    refined_angles = refined_angles[:actual_length]
    residue_ids = original['residue_ids']
    
    # Determine sample name from the AF2 CIF file name.
    sample_name = os.path.splitext(os.path.basename(af2_cif_path))[0]
    
    # Write refined angles to CSV
    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{sample_name}_refined_angles.csv")
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sample", "Residue_ID", "Phi", "Psi"])
        for i in range(0, len(refined_angles), 2):
            res_index = i // 2
            if res_index < len(residue_ids):
                phi = refined_angles[i]
                psi = refined_angles[i + 1]
                writer.writerow([sample_name, residue_ids[res_index], f"{phi:.2f}", f"{psi:.2f}"])
    print(f"✅ Refined φ/ψ angles saved to: {output_csv}")
    
    # Rebuild the 3D structure using your provided rebuild module.
    output_pdb = os.path.join(output_dir, f"{sample_name}_reconstructed.pdb")
    rebuild_structure_from_angles(af2_cif_path, refined_angles, output_pdb)
    print(f"✅ Reconstructed PDB saved to: {output_pdb}")
    
    env.render()
    return refined_angles

if __name__ == "__main__":
    # Replace with the path to your new AF2 prediction CIF file.
    af2_cif_path = "data/Sample 10/AF2/7yat.cif"
    run_inference(af2_cif_path, max_steps=50, fixed_dim=100)
