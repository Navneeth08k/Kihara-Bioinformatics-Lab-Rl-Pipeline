"""
amyloid_env.py  –  Amyloid backbone‑refinement Gym environments
Upgrades: real per‑residue SASA via FreeSASA (pre‑computed once)
"""

import os, random, io, tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tempfile
from Bio.PDB import MMCIFParser, PPBuilder, PDBIO
from Bio.PDB.Polypeptide import is_aa
import freesasa                          # ← FreeSASA python bindings

# ------------------------------------------------------------------
# helper: pad / truncate φψ vector
# ------------------------------------------------------------------
def pad_phi_psi(vec, fixed_dim):
    if len(vec) < fixed_dim:
        return np.concatenate([vec, np.zeros(fixed_dim - len(vec))])
    return vec[:fixed_dim]

# ------------------------------------------------------------------
# structure loader – now with real SASA
# ------------------------------------------------------------------
# ---------- helper for real SASA (patched) ------------------------------
def _sasa_per_residue(structure):
    chain = next(structure[0].get_chains())

    # ---- write chain to an in‑memory string ----
    pdb_buf = io.StringIO()
    io_writer = PDBIO()
    io_writer.set_structure(chain)
    io_writer.save(pdb_buf)
    pdb_text = pdb_buf.getvalue()

    # ---- FreeSASA needs a filename, so use a temp file ----
    with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as fh:
        fh.write(pdb_text)
        fh.flush()                        # make sure contents hit disk
        fs_struct = freesasa.Structure(fh.name)

    result = freesasa.calc(fs_struct)
    areas  = result.residueAreas()

    sasa_list = []
    for res in chain:
        if not is_aa(res):
            continue
        key = (chain.id, res.get_resname(), res.id[1])
        sasa_list.append(areas.get(key, {}).get("total", 0.0))

    return np.array(sasa_list, dtype=np.float32)


def load_structure(cif_path):
    """Return dict with angles, coords, per‑residue SASA, masks, …"""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)
    chain = next(structure[0].get_chains())

    # ---------- φ/ψ -------------------------------------------------
    phi_psi, res_ids = [], []
    for pp in PPBuilder().build_peptides(chain):
        for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            if phi and psi:
                phi_psi.extend([np.degrees(phi), np.degrees(psi)])
                res_ids.append(res.id[1])

    # ---------- atom coords ----------------------------------------
    coords, b_factors = [], []
    for res in chain:
        for atom in res:
            coords.append(atom.get_coord())
            b_factors.append(atom.get_bfactor())

    # ---------- residue masks --------------------------------------
    hyd_set = {"LEU", "ILE", "VAL", "ALA", "MET"}
    pol_set = {"ARG", "LYS", "ASP", "GLU", "GLN", "ASN"}

    hydro_mask, polar_mask = [], []
    for res in chain:
        if is_aa(res):
            name = res.get_resname().upper()
            hydro_mask.append(1 if name in hyd_set else 0)
            polar_mask.append(1 if name in pol_set else 0)

    hydro_mask = np.asarray(hydro_mask, dtype=np.float32)
    polar_mask = np.asarray(polar_mask, dtype=np.float32)

    # ---------- real SASA ------------------------------------------
    sasa = _sasa_per_residue(structure)  # length == len(hydro_mask)

    return dict(
        phi_psi=np.asarray(phi_psi, dtype=np.float32),
        coordinates=np.asarray(coords, dtype=np.float32),
        b_factors=np.asarray(b_factors, dtype=np.float32),
        sasa=sasa,
        hydro_mask=hydro_mask,
        polar_mask=polar_mask,
        residue_ids=res_ids,
    )

# ------------------------------------------------------------------
# metric helpers  (unchanged except SASA now real)
# ------------------------------------------------------------------
def compute_energy(state):
    phi = state["phi_psi"]
    ideal = np.array([-120, 120] * (len(phi) // 2), dtype=np.float32)
    return 0.005 * np.sum((phi - ideal) ** 2)

def compute_rmsd(state, target):
    a, b = state["phi_psi"], target["phi_psi"]
    m = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2)))

def compute_sasa(state):
    s = state["sasa"]
    h = float(np.sum(s * state["hydro_mask"]))
    p = float(np.sum(s * state["polar_mask"]))
    return h, p

def beta_fraction(phi_psi):
    return sum(
        -150 <= phi <= -90 and 90 <= psi <= 150
        for phi, psi in zip(phi_psi[0::2], phi_psi[1::2])
    ) / (len(phi_psi) // 2)

def clash_penalty(state, thr=2.0):
    xyz = state["coordinates"]
    pen = 0.0
    for i in range(len(xyz)):
        d = np.linalg.norm(xyz[i + 1 :] - xyz[i], axis=1)
        pen += np.sum(np.clip(thr - d, 0, None))
    return pen

# ------------------------------------------------------------------
# reward: scaled as in previous answer
# ------------------------------------------------------------------
def reward_func(state, target):
    N = len(state["hydro_mask"])
    E_raw = compute_energy(state)
    rmsd = compute_rmsd(state, target)
    hydS, polS = compute_sasa(state)
    beta = -abs(beta_fraction(state["phi_psi"]) - beta_fraction(target["phi_psi"]))
    clash = clash_penalty(state)

    E = E_raw / (1_000.0)     # scale constants chosen from observed ranges
    rms = rmsd / 10.0
    hS = hydS / 200.0
    pS = polS / 200.0

    return (
        -0.40 * E
        -0.30 * rms
        + 0.15 * (-hS + pS / 5.0)
        + 0.10 * beta
        - 0.05 * clash
    )

# ------------------------------------------------------------------
# env classes (same as before, condensed print)
# ------------------------------------------------------------------
def apply_action(state, action):
    new = state.copy()
    new_phi = np.clip(state["phi_psi"] + action, -180, 180)
    new["phi_psi"] = new_phi
    return new

class AmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, af2_cif, exp_cif, max_steps=50, fixed_dim=128, inference_mode=False):
        super().__init__()
        self.af2_cif, self.exp_cif = af2_cif, exp_cif
        self.max_steps, self.fixed_dim = max_steps, fixed_dim
        self.i_mode = inference_mode

        self.state = load_structure(af2_cif)
        self.state["phi_psi"] = pad_phi_psi(self.state["phi_psi"], fixed_dim)
        self.target = self.state if inference_mode else load_structure(exp_cif)
        self.target["phi_psi"] = pad_phi_psi(self.target["phi_psi"], fixed_dim)

        self.action_space = spaces.Box(-5.0, 5.0, (fixed_dim,), np.float32)
        self.observation_space = spaces.Box(-180, 180, (fixed_dim,), np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = load_structure(self.af2_cif)
        self.state["phi_psi"] = pad_phi_psi(self.state["phi_psi"], self.fixed_dim)
        return self.state["phi_psi"], {}

    def step(self, action):
        self.state = apply_action(self.state, action)
        self.current_step += 1
        reward = 0.0 if self.i_mode else reward_func(self.state, self.target)
        trunc = self.current_step >= self.max_steps
        hydS, polS = compute_sasa(self.state)
        rmsd_val = None if self.i_mode else compute_rmsd(self.state, self.target)
        print(f"[{self.current_step:02d}] E={compute_energy(self.state):.1f} "
              f"RMSD={rmsd_val} hydS={hydS:.1f} polS={polS:.1f} R={reward:+.2f}")
        return self.state["phi_psi"], reward, False, trunc, {}

class MetaAmyloidEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, pairs, max_steps=50, fixed_dim=128, inference_mode=False):
        super().__init__()
        self.pairs = pairs
        self.max_steps, self.fixed_dim = max_steps, fixed_dim
        self.i_mode = inference_mode
        self.action_space = spaces.Box(-5.0, 5.0, (fixed_dim,), np.float32)
        self.observation_space = spaces.Box(-180, 180, (fixed_dim,), np.float32)
        self.current_env = None

    def reset(self, seed=None, options=None):
        pair = random.choice(self.pairs)
        self.current_env = AmyloidEnv(pair[0], pair[1], self.max_steps, self.fixed_dim, self.i_mode)
        return self.current_env.reset(seed=seed, options=options)

    def step(self, action):   return self.current_env.step(action)
    def render(self, mode="human"): self.current_env.render(mode)
