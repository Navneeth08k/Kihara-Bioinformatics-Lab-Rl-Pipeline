


#!/usr/bin/env python
# ----------------------------------------------------------------------
#  train_rl.py  –  PPO training driver for AmyloidEnv (GPU)
# ----------------------------------------------------------------------

import os
import random
import glob
import importlib

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)

# ---------- import environment ----------------------------------------
amyloid_env = importlib.import_module("amyloid_env")
importlib.reload(amyloid_env)
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

def make_env(pair, fixed_dim, sasa_period=5):
    return lambda: AmyloidEnv(pair[0], pair[1],
                              max_steps=120,
                              fixed_dim=fixed_dim,
                              inference_mode=False,
                              sasa_period=sasa_period)

# ---------- TensorBoard callback --------------------------------------
class MetricTBCallback(BaseCallback):
    """Log custom metrics placed in env-info dict."""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        # metrics that the env now returns in info{}
        keys = [
            "energy", "rmsd",
            "hyd_buried", "pol_buried", "total_buried",
            "Δhyd", "Δpol",
            "clash", "reward",
        ]
        acc = {k: [] for k in keys}
        for info in infos:
            for k in keys:
                val = info.get(k)
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

    FIXED_DIM, N_ENVS, SASA_PERIOD = 128, 4, 5

    # --- Training environment: SubprocVecEnv → VecMonitor → VecNormalize ---
    train_base = SubprocVecEnv([
        make_env(p, FIXED_DIM, sasa_period=SASA_PERIOD)
        for p in train_pairs[:N_ENVS]
    ])
    train_mon  = VecMonitor(train_base)
    train_vec  = VecNormalize(train_mon, norm_obs=True, norm_reward=True, clip_reward=5.0)

    # --- Evaluation environment: wrap *exactly* as training, but no updates to stats ---
    eval_base = SubprocVecEnv([
        make_env(test_pairs[0], FIXED_DIM, sasa_period=SASA_PERIOD)
    ])
    eval_mon  = VecMonitor(eval_base)
    eval_vec  = VecNormalize(
        eval_mon,
        training=False,
        norm_obs=True,
        norm_reward=True
    )
    # sync normalization statistics from train → eval
    eval_vec.obs_rms = train_vec.obs_rms
    eval_vec.ret_rms = train_vec.ret_rms

    ckpt_cb = CheckpointCallback(50_000, "./ckpts/", name_prefix="ppo_amyloid")
    eval_cb = EvalCallback(
        eval_env=eval_vec,                     # make sure we're passing the VecNormalize wrapper
        best_model_save_path="./logs/best/",
        log_path="./logs/",
        eval_freq=25_000,
        deterministic=True,
    )
    callbacks = CallbackList([ckpt_cb, eval_cb, MetricTBCallback()])

    model = PPO(
        "MlpPolicy",
        train_vec,
        verbose=1,
        tensorboard_log="./logs/tb/",
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.02,
        n_steps=512 * N_ENVS,     # e.g. 512 steps total if N_ENVS=2
        device="cuda",
    )
    model.learn(total_timesteps=200_000, callback=callbacks)
    model.save("ppo_amyloid_final")
    print("✔ Training complete → ppo_amyloid_final.zip")

if __name__ == "__main__":
    main()


