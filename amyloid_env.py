# amyloid_env.py  –  Amyloid backbone‑refinement Gym environments
# Added: removal of padding bias, enriched observations with chemical context

import io
import os
import random
import tempfile

import numpy as np
import freesasa
import gymnasium as gym
from gymnasium import spaces

from Bio.PDB import MMCIFParser, PDBParser, PPBuilder, PDBIO
from Bio.PDB.Polypeptide import is_aa
from PeptideBuilder import make_structure

# mapping 3-letter → 1-letter
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# ─────────────────────────────────────────────────────────────────────────────
# helper: pad/truncate φ/ψ to fixed_dim
# ─────────────────────────────────────────────────────────────────────────────
def pad_phi_psi(vec, fixed_dim):
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[0] < fixed_dim:
        return np.concatenate([vec, np.zeros(fixed_dim - vec.shape[0], dtype=np.float32)])
    return vec[:fixed_dim]

# ─────────────────────────────────────────────────────────────────────────────
# robust per‑residue SASA extractor (works with all FreeSASA layouts)
# ─────────────────────────────────────────────────────────────────────────────
def _sasa_per_residue(structure):
    chain = next(structure[0].get_chains())
    pdb_buf = io.StringIO()
    writer = PDBIO()
    writer.set_structure(chain)
    writer.save(pdb_buf)

    with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as fh:
        fh.write(pdb_buf.getvalue())
        fh.flush()
        fs_struct = freesasa.Structure(fh.name)
        res_areas = freesasa.calc(fs_struct).residueAreas()

    sasa_map = {}
    sample_key = next(iter(res_areas.keys()))
    sample_val = res_areas[sample_key]

    if isinstance(sample_val, dict):  # dict-of-dicts
        for ch, inner in res_areas.items():
            for seq, area in inner.items():
                sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))
    elif isinstance(sample_key, str):  # "A:18:ARG"
        for key, area in res_areas.items():
            parts = key.split(":")
            if len(parts) == 3:
                ch, seq, _ = parts
                sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))
    else:  # ResidueId objects
        for rid, area in res_areas.items():
            ch = getattr(rid, "chain", None) or getattr(rid, "chainLabel")()
            seq = getattr(rid, "number", None) or getattr(rid, "resNumber")()
            sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))

    out = []
    for residue in chain:
        if is_aa(residue):
            key = (chain.id, residue.id[1])
            out.append(sasa_map.get(key, 0.0))
    return np.array(out, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# load a single CIF → features dict (angles, coords, masks, sasa)
# ─────────────────────────────────────────────────────────────────────────────
def load_structure(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)
    chain = next(structure[0].get_chains())

    # φ/ψ + residue indices
    phi_psi = []
    res_ids = []
    for pp in PPBuilder().build_peptides(chain):
        for residue, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            if phi is not None and psi is not None:
                phi_psi.extend([np.degrees(phi), np.degrees(psi)])
                res_ids.append(residue.id[1])

    # atom coords (for clash term)
    coords = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
    coords = np.asarray(coords, dtype=np.float32)

    # hydro/polar masks (per residue)
    hyd_set = {"LEU", "ILE", "VAL", "ALA", "MET"}
    pol_set = {"ARG", "LYS", "ASP", "GLU", "GLN", "ASN"}
    hydro_mask = []
    polar_mask = []
    for residue in chain:
        if is_aa(residue):
            name = residue.get_resname().upper()
            hydro_mask.append(1.0 if name in hyd_set else 0.0)
            polar_mask.append(1.0 if name in pol_set else 0.0)
    hydro_mask = np.asarray(hydro_mask, dtype=np.float32)
    polar_mask = np.asarray(polar_mask, dtype=np.float32)

    # real per-residue SASA
    sasa = _sasa_per_residue(structure)

    return {
        "phi_psi": np.asarray(phi_psi, dtype=np.float32),
        "coordinates": coords,
        "sasa": sasa,
        "hydro_mask": hydro_mask,
        "polar_mask": polar_mask,
        "residue_ids": res_ids,
    }

# ─────────────────────────────────────────────────────────────────────────────
# reward‑term calculations (masking out padding)
# ─────────────────────────────────────────────────────────────────────────────
def compute_energy(state):
    φψ = state["phi_psi"]
    ideal = np.tile([-120.0, 120.0], len(φψ)//2).astype(np.float32)
    m = state["mask"]
    return 0.005 * np.sum(((φψ - ideal)**2) * m)

def compute_rmsd(state, target):
    L = state["valid_len"]
    a = state["phi_psi"][:L]
    b = target["phi_psi"][:L]
    return float(np.sqrt(np.mean((a - b)**2)))

def compute_sasa_terms(state):
    s = state["sasa"]
    h = float((s * state["hydro_mask"]).sum())
    p = float((s * state["polar_mask"]).sum())
    return h, p

def beta_fraction(phi_psi, valid_len=None):
    if valid_len is not None:
        phi_psi = phi_psi[:valid_len]
    phi = phi_psi[0::2]
    psi = phi_psi[1::2]
    return float(((phi >= -150)&(phi <= -90)&(psi >= 90)&(psi <=150)).mean())

def clash_penalty(state, thr=2.0):
    xyz = state["coordinates"]
    d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)

    # consider only the *largest* overlap per atom and cap it
    per_atom = np.clip(thr - d, 0.0, 0.5).max(axis=1)   # ← CHANGE
    return float(per_atom.sum())                        # ← CHANGE


def reward_func(state, target, delta_hS, delta_pS):
    E = compute_energy(state) / 1000.0
    R = compute_rmsd(state, target) / 10.0
    # delta‐SASA shaping
    Δh = delta_hS   # positive when hydrophobic burial increases
    Δp = delta_pS   # positive when polar exposure increases
    βs = beta_fraction(state["phi_psi"], state["valid_len"])
    βt = beta_fraction(target["phi_psi"], state["valid_len"])
    βdiff = -abs(βs - βt)
    clash = clash_penalty(state)
    

    # curriculum
    if self.global_step < 50_000:
        return 1.0 * Δh + 0.5 * Δp

    # full reward
    return (
        -0.10 * E
        -0.05 * R
        +1.0 * Δh
        +0.5 * Δp
        +0.10 * βdiff
        -0.01 * clash
    )

# ─────────────────────────────────────────────────────────────────────────────
# apply_action helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_action(state, action):
    new = state.copy()
    new["phi_psi"] = pad_phi_psi(state["phi_psi"] + action, len(action))
    return new

# ─────────────────────────────────────────────────────────────────────────────
# Gym Environment
# ─────────────────────────────────────────────────────────────────────────────
class AmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        af2_cif,
        exp_cif=None,
        max_steps=50,
        fixed_dim=128,
        inference_mode=False,
        sasa_period=5,
    ):
        super().__init__()
        self.af2_cif = af2_cif
        self.exp_cif = exp_cif
        self.max_steps = max_steps if max_steps is not None else 120
        self.fixed_dim = fixed_dim
        self.i_mode = inference_mode
        self.sasa_period = 1 if sasa_period is None else sasa_period
        self.current_step = 0
        self.global_step = 0

        # ----- initial state -----
        st0 = load_structure(self.af2_cif)
        raw_phi = st0["phi_psi"]
        valid_len = raw_phi.shape[0]
        st0["valid_len"] = valid_len
        mask = np.zeros(self.fixed_dim, dtype=np.float32)
        mask[:valid_len] = 1.0
        st0["mask"] = mask
        st0["phi_psi"] = pad_phi_psi(raw_phi, self.fixed_dim)
        self.state = st0
        # initialize previous SASA for delta‐SASA shaping
        h0, p0 = compute_sasa_terms(self.state)
        self.prev_hS, self.prev_pS = h0, p0


        # ----- target state -----
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("seq", self.af2_cif)
        chain = next(structure[0].get_chains())
        self.sequence = ''.join(
            three_to_one.get(res.get_resname(), 'A')
            for res in chain if is_aa(res)
        )
        if self.i_mode or self.exp_cif is None:
            tgt = st0.copy()
        else:
            tgt0 = load_structure(self.exp_cif)
            raw_phi_t = tgt0["phi_psi"]
            tgt0["valid_len"] = raw_phi_t.shape[0]
            mask_t = np.zeros(self.fixed_dim, dtype=np.float32)
            mask_t[:tgt0["valid_len"]] = 1.0
            tgt0["mask"] = mask_t
            tgt0["phi_psi"] = pad_phi_psi(raw_phi_t, self.fixed_dim)
            tgt = tgt0
        self.target = tgt

        # action space
        self.action_space = spaces.Box(-20.0, 20.0, (self.fixed_dim,), np.float32)

        # observation space: [φψ_norm | hydro_mask_pad | polar_mask_pad]
        low = np.concatenate([
            np.full(self.fixed_dim, -1.0, dtype=np.float32),
            np.zeros(self.fixed_dim, dtype=np.float32)
        ])
        high = np.concatenate([
            np.ones(self.fixed_dim, dtype=np.float32),
            np.ones(self.fixed_dim, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def _build_obs(self):
        φψ_norm = self.state["phi_psi"] / 180.0
        orig_hm = self.state["hydro_mask"]
        orig_pm = self.state["polar_mask"]
        pad_res = self.fixed_dim // 2

        # safely truncate or pad
        hm = np.zeros(pad_res, dtype=np.float32)
        pm = np.zeros(pad_res, dtype=np.float32)

        # only copy up to pad_res entries
        end_h = min(orig_hm.shape[0], pad_res)
        hm[:end_h] = orig_hm[:end_h]

        end_p = min(orig_pm.shape[0], pad_res)
        pm[:end_p] = orig_pm[:end_p]

        return np.concatenate([φψ_norm, hm, pm], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        st = load_structure(self.af2_cif)
        raw_phi = st["phi_psi"]
        valid_len = raw_phi.shape[0]
        st["valid_len"] = valid_len
        mask = np.zeros(self.fixed_dim, dtype=np.float32)
        mask[:valid_len] = 1.0
        st["mask"] = mask
        st["phi_psi"] = pad_phi_psi(raw_phi, self.fixed_dim)
        self.state = st
        # reset previous SASA
        h0, p0 = compute_sasa_terms(self.state)
        self.prev_hS, self.prev_pS = h0, p0
        obs = self._build_obs()
        return obs, {}

    def _maybe_recompute_sasa(self):
        if not self.sasa_period or (self.current_step % self.sasa_period != 0):
            return
        phi_psi_vec = self.state["phi_psi"]
        max_pairs = len(phi_psi_vec) // 2
        N = min(len(self.sequence), max_pairs)
        if N == 0:
            return
        seq = self.sequence[:N]
        flat = phi_psi_vec[:2 * N]
        phi_list = np.radians(flat[0::2])
        psi_actual = np.radians(flat[1::2])
        psi_im1 = [np.radians(120.0)] + psi_actual[:-1].tolist()
        model = make_structure(seq, phi=phi_list, psi_im1=psi_im1)
        if model is None:
            return
        buf = io.StringIO()
        writer = PDBIO()
        writer.set_structure(model)
        writer.save(buf)
        xyz_list = [atom.get_coord() for atom in model.get_atoms()]
        self.state["coordinates"] = np.asarray(xyz_list, dtype=np.float32)
        with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as fh:
            fh.write(buf.getvalue())
            fh.flush()
            struct_tmp = PDBParser(QUIET=True).get_structure("tmp", fh.name)
            sasa_vec = _sasa_per_residue(struct_tmp)
        self.state["sasa"] = sasa_vec
        hyd_set = {"L", "I", "V", "A", "M"}
        pol_set = {"R", "K", "D", "E", "Q", "N"}
        self.state["hydro_mask"] = np.array([1.0 if aa in hyd_set else 0.0 for aa in seq], dtype=np.float32)
        self.state["polar_mask"] = np.array([1.0 if aa in pol_set else 0.0 for aa in seq], dtype=np.float32)

    def step(self, action):
        """
        Apply an action (delta φ/ψ), rebuild SASA/clash as needed, and return
        (obs, reward, done, trunc, info) with a two‐stage curriculum:
         - First 20k steps: maximize ΔhydS & ΔpolS only
         - Thereafter: full reward with lighter penalties & supercharged SASA
        """
        # 1) Increment global and episode steps
        self.global_step  += 1
        self.current_step += 1

        # 2) Apply the action and rebuild backbone / SASA
        self.state = apply_action(self.state, action)
        self._maybe_recompute_sasa()

        # 3) Compute raw terms
        E    = compute_energy(self.state)                          # raw, unscaled
        R    = compute_rmsd(self.state, self.target)               # raw, unscaled
        hS, pS = compute_sasa_terms(self.state)
        delta_h = self.prev_hS - hS
        delta_p = pS - self.prev_pS
        beta_s = beta_fraction(self.state["phi_psi"], self.state["valid_len"])
        beta_t = beta_fraction(self.target["phi_psi"], self.state["valid_len"])
        beta_diff = -abs(beta_s - beta_t)
        clash = clash_penalty(self.state)

        # 4) Compute reward with curriculum
        if self.global_step < 20_000:
            # Stage 1: pure SASA shaping
            rew = 0.50 * delta_h + 0.20 * delta_p
        else:
            # Stage 2: full reward
            rew = (
                -0.10 * (E / 1000.0)      # energy /1000
                -0.05 * (R / 10.0)        # RMSD /10
                + 0.50 * delta_h          # supercharged hydrophobic burial
                + 0.20 * delta_p          # supercharged polar exposure
                + 0.10 * beta_diff        # beta‐sheet matching
                - 0.01 * clash            # capped clash penalty
            )

        # 5) Update previous SASA for next delta
        self.prev_hS, self.prev_pS = hS, pS

        # 6) Build info dict for TensorBoard
        info = {
            "step":    self.current_step,
            "global_step": self.global_step,
            "energy":  E,
            "rmsd":    None if self.i_mode else R,
            "hydS":    hS,
            "polS":    pS,
            "ΔhydS":   delta_h,
            "ΔpolS":   delta_p,
            "beta":    beta_diff,
            "clash":   clash,
            "reward":  rew,
        }

        # 7) Check truncation
        done  = False
        trunc = (self.current_step >= self.max_steps)

        # 8) Build next observation
        obs = self._build_obs()

        return obs, rew, done, trunc, info

    


    def render(self, mode="human"):
        print(f"[{self.current_step:02d}] E={compute_energy(self.state):.1f} "
              f"RMSD={compute_rmsd(self.state, self.target):.1f} "
              f"hydS={compute_sasa_terms(self.state)[0]:.1f} "
              f"polS={compute_sasa_terms(self.state)[1]:.1f}")

class MetaAmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, pairs, **kwargs):
        super().__init__()
        self.pairs = pairs
        self.kw = kwargs
        self.current_env = None
        env0 = AmyloidEnv(*pairs[0], **kwargs)
        self.observation_space = env0.observation_space
        self.action_space = env0.action_space

    def reset(self, **kwargs):
        af2, exp = random.choice(self.pairs)
        self.current_env = AmyloidEnv(af2, exp, **self.kw)
        return self.current_env.reset(**kwargs)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode="human"):
        return self.current_env.render(mode)
