import os
import csv
import numpy as np
from stable_baselines3 import PPO
from amyloid_env import AmyloidEnv, load_structure, pad_phi_psi
from rebuild_structure import rebuild_structure_from_angles

def find_latest_cif(directory="inference_outputs"):
    cif_files = sorted([f for f in os.listdir(directory) if f.endswith(".cif")])
    if not cif_files:
        raise FileNotFoundError("❌ No AlphaFold CIF file found.")
    return os.path.join(directory, cif_files[-1])

def run_inference(af2_cif_path, max_steps=50, fixed_dim=None):
    if fixed_dim is None:
        fixed_dim = len(load_structure(af2_cif_path)['phi_psi'])  # Auto-detect dimensions

    env = AmyloidEnv(af2_cif_path, exp_pdb_path=None, max_steps=max_steps, fixed_dim=fixed_dim, inference_mode=True)
    model = PPO.load("ppo_amyloid_final", env=env, custom_objects={"observation_space": env.observation_space})


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
    original = load_structure(af2_cif_path)
    actual_length = len(original['phi_psi'])
    refined_angles = refined_angles[:actual_length]
    residue_ids = original['residue_ids']
    sample_name = os.path.splitext(os.path.basename(af2_cif_path))[0]

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

    output_pdb = os.path.join(output_dir, f"{sample_name}_reconstructed.pdb")
    rebuild_structure_from_angles(af2_cif_path, refined_angles, output_pdb)
    print(f"✅ Reconstructed PDB saved to: {output_pdb}")

    env.render()
    return refined_angles

if __name__ == "__main__":
    af2_cif_path = find_latest_cif()
    run_inference(af2_cif_path)
