#!/usr/bin/env python
# ----------------------------------------------------------------------
#  train_rl.py  –  PPO training driver for AmyloidEnv (GPU)
# ----------------------------------------------------------------------

import os, random, glob, importlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)

# ---------- import environment ----------------------------------------
amyloid_env = importlib.import_module("amyloid_env")
AmyloidEnv, MetaAmyloidEnv = amyloid_env.AmyloidEnv, amyloid_env.MetaAmyloidEnv

# ---------- utilities -------------------------------------------------
def first_structure(folder):
    for ext in ("*.cif", "*.pdb", "*.CIF", "*.PDB"):
        hits = glob.glob(os.path.join(folder, ext))
        if hits:
            return hits[0]
    return None

def find_pairs(root="data"):
    pairs = []
    for cur, dirs, _ in os.walk(root):
        if {"AF2", "Experimental"}.issubset(set(dirs)):
            af2 = first_structure(os.path.join(cur, "AF2"))
            exp = first_structure(os.path.join(cur, "Experimental"))
            if af2 and exp:
                pairs.append((af2, exp))
            else:
                print("• Skipping", cur, "(no .cif/.pdb)")
    print(f"✓ Found {len(pairs)} sample pairs.")
    return pairs

def make_env(pair, fixed_dim):
    return lambda: AmyloidEnv(pair[0], pair[1],
                              max_steps=50,
                              fixed_dim=fixed_dim,
                              inference_mode=False)

# ---------- TensorBoard callback --------------------------------------
class MetricTBCallback(BaseCallback):
    """Log custom metrics placed in env info dict."""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        keys = ["energy", "rmsd", "hydS", "polS", "reward"]
        acc = {k: [] for k in keys}
        for info in infos:
            for k in keys:
                val = info.get(k)            # safe get
                if val is not None:
                    acc[k].append(val)
        for k, lst in acc.items():
            if lst:
                self.logger.record(f"custom/{k}", float(np.mean(lst)))
        return True

# ---------- main ------------------------------------------------------
def main():
    pairs = find_pairs("data")
    if len(pairs) < 10:
        raise RuntimeError("Need ≥10 sample pairs under data/")

    random.shuffle(pairs)
    split = int(0.2 * len(pairs))
    test_pairs, train_pairs = pairs[:split], pairs[split:]

    FIXED_DIM, N_ENVS = 128, 8
    train_vec = VecMonitor(SubprocVecEnv([make_env(p, FIXED_DIM)
                                          for p in train_pairs[:N_ENVS]]))
    test_vec  = SubprocVecEnv([make_env(test_pairs[0], FIXED_DIM)])

    ckpt_cb = CheckpointCallback(25_000, "./ckpts/", name_prefix="ppo_amyloid")
    eval_cb = EvalCallback(
        test_vec,
        best_model_save_path="./logs/best/",
        log_path="./logs/",
        eval_freq=5_000,
        deterministic=True,
    )

 
    callbacks = CallbackList([ckpt_cb, eval_cb, MetricTBCallback()])

    model = PPO("MlpPolicy",
                train_vec,
                verbose=1,
                tensorboard_log="./logs/tb/",
                learning_rate=3e-4,
                clip_range=0.2,
                device="cuda")

    model.learn(total_timesteps=2_800_000, callback=callbacks)
    model.save("ppo_amyloid_final")
    print("✔ Training complete → ppo_amyloid_final.zip")

if __name__ == "__main__":
    main()
