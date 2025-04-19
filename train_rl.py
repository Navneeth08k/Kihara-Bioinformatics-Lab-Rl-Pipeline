
#!/usr/bin/env python
# ======================================================================
#  train_rl.py   –   PPO training driver for AmyloidEnv on GPU
#  (requires amyloid_env.py in the same folder)
# ======================================================================

import os, random, glob, importlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

# ----------------------------------------------------------------------
#  Import your full environment (must already exist)
# ----------------------------------------------------------------------
amyloid_env = importlib.import_module("amyloid_env")
AmyloidEnv, MetaAmyloidEnv = (
    amyloid_env.AmyloidEnv,
    amyloid_env.MetaAmyloidEnv,
)

# ----------------------------------------------------------------------
#  Robust util: locate (AF2, Experimental) structure pairs
# ----------------------------------------------------------------------
def first_structure(folder):
    for ext in ("*.cif", "*.pdb", "*.CIF", "*.PDB"):
        hits = glob.glob(os.path.join(folder, ext))
        if hits:
            return hits[0]
    return None

def find_pairs(root="data"):
    """Recursively find directories that contain both AF2/ and Experimental/."""
    pairs = []
    for cur, dirs, _ in os.walk(root):
        if {"AF2", "Experimental"}.issubset(set(dirs)):
            af2_file = first_structure(os.path.join(cur, "AF2"))
            exp_file = first_structure(os.path.join(cur, "Experimental"))
            if af2_file and exp_file:
                pairs.append((af2_file, exp_file))
            else:
                print("• Skipping", cur, "(no .cif/.pdb structures)")
    print(f"✓ Found {len(pairs)} sample pairs.")
    return pairs

# ----------------------------------------------------------------------
#  Helpers to build parallel Gym envs
# ----------------------------------------------------------------------
def make_env(pair, fixed_dim):
    return lambda: AmyloidEnv(
        pair[0],
        pair[1],
        max_steps=50,
        fixed_dim=fixed_dim,
        inference_mode=False,
    )

# ----------------------------------------------------------------------
#  Main training routine
# ----------------------------------------------------------------------
def main():
    pairs = find_pairs("data")
    if len(pairs) < 10:
        raise RuntimeError(
            "Need ≥10 sample pairs under /content/data/ to start training."
        )

    random.shuffle(pairs)
    split = int(0.2 * len(pairs))
    test_pairs, train_pairs = pairs[:split], pairs[split:]

    FIXED_DIM = 128
    N_ENVS = 8  # number of parallel CPU envs (tune to your VM)

    train_vec = VecMonitor(
        SubprocVecEnv([make_env(p, FIXED_DIM) for p in train_pairs[:N_ENVS]])
    )

    test_vec = SubprocVecEnv([make_env(test_pairs[0], FIXED_DIM)])

    ckpt_cb = CheckpointCallback(
        save_freq=25_000, save_path="./ckpts/", name_prefix="ppo_amyloid"
    )
    eval_cb = EvalCallback(
        test_vec,
        best_model_save_path="./logs/best/",
        log_path="./logs/",
        eval_freq=5_000,
        deterministic=True,
    )

    callbacks = CallbackList([ckpt_cb, eval_cb])

    model = PPO(
        "MlpPolicy",
        train_vec,
        verbose=1,
        tensorboard_log="./logs/tb/",
        learning_rate=3e-4,
        clip_range=0.2,
        device="cuda",  #  ← GPU!
    )

    TOTAL_TIMESTEPS = 2_800_000  # ≈12 h on Colab T4/A100
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save("ppo_amyloid_final")
    print("✔ Training completed. Model saved →  ppo_amyloid_final.zip")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
