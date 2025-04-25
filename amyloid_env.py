# amyloid_env.py  –  Amyloid backbone‑refinement Gym environments
# Added: real per‑residue SASA via FreeSASA, recomputed every sasa_period steps

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
from PeptideBuilder import PeptideBuilder, make_structure


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
    """
    Given a Bio.PDB structure, return a 1D float32 numpy array of
    total SASA (Å²) for each amino-acid in the chain, in PDB order.
    """
    # take only the first chain
    chain = next(structure[0].get_chains())

    # write that chain out to a PDB buffer
    pdb_buf = io.StringIO()
    writer = PDBIO()
    writer.set_structure(chain)
    writer.save(pdb_buf)

    # run freesasa
    with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as fh:
        fh.write(pdb_buf.getvalue())
        fh.flush()
        fs_struct = freesasa.Structure(fh.name)
        res_areas = freesasa.calc(fs_struct).residueAreas()

    # flatten whatever freesasa gives us into a mapping (chain, resSeq)->area.total
    sasa_map = {}
    sample_key = next(iter(res_areas.keys()))
    sample_val = res_areas[sample_key]

    # dict-of-dicts: {'A': {'18': ResidueArea, ...}}
    if isinstance(sample_val, dict):
        for ch, inner in res_areas.items():
            for seq, area in inner.items():
                sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))

    # flat "A:18:ARG" → split on “:”
    elif isinstance(sample_key, str):
        for key, area in res_areas.items():
            parts = key.split(":")
            if len(parts) == 3:
                ch, seq, _ = parts
                sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))

    # ResidueId objects (newer freesasa)
    else:
        for rid, area in res_areas.items():
            ch = getattr(rid, "chain", None) or getattr(rid, "chainLabel")()
            seq = getattr(rid, "number", None) or getattr(rid, "resNumber")()
            sasa_map[(ch, int(seq))] = float(getattr(area, "total", 0.0))

    # now build the vector in the exact residue order Bio.PDB uses
    out = []
    for residue in chain:
        if is_aa(residue):
            key = (chain.id, residue.id[1])
            out.append(sasa_map.get(key, 0.0))
    return np.array(out, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# load a single CIF→ features dict (angles, coords, masks, sasa)
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

    # hydro/polar masks
    hyd_set = {"LEU", "ILE", "VAL", "ALA", "MET"}
    pol_set = {"ARG", "LYS", "ASP", "GLU", "GLN", "ASN"}
    hydro_mask = []
    polar_mask = []
    for residue in chain:
        if is_aa(residue):
            name = residue.get_resname().upper()
            hydro_mask.append(1 if name in hyd_set else 0)
            polar_mask.append(1 if name in pol_set else 0)
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
# energy, RMSD, SASA, beta‐fraction & clash terms (all scaled for reward)
# ─────────────────────────────────────────────────────────────────────────────
def compute_energy(state):
    phi = state["phi_psi"]
    ideal = np.tile([-120.0, 120.0], len(phi)//2).astype(np.float32)
    return 0.005 * np.sum((phi - ideal)**2)

def compute_rmsd(state, target):
    a, b = state["phi_psi"], target["phi_psi"]
    m = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:m] - b[:m])**2)))

def compute_sasa_terms(state):
    s = state["sasa"]
    h = float((s * state["hydro_mask"]).sum())
    p = float((s * state["polar_mask"]).sum())
    return h, p

def beta_fraction(phi_psi):
    phi = phi_psi[0::2]; psi = phi_psi[1::2]
    return float(((phi >= -150)&(phi <= -90)&(psi >= 90)&(psi <=150)).mean())

def clash_penalty(state, thr=2.0):
    xyz = state["coordinates"]
    # pairwise distances
    d = np.linalg.norm(xyz[:,None] - xyz[None,:], axis=-1)
    return float(np.clip(thr - d, 0.0, None).sum() * 0.1)

# scaled reward combination
def reward_func(state, target):
    E = compute_energy(state) / 1000.0
    R = compute_rmsd(state, target) / 10.0
    hS, pS = compute_sasa_terms(state)
    hS /= 200.0; pS /= 200.0
    βdiff = -abs(beta_fraction(state["phi_psi"]) - beta_fraction(target["phi_psi"]))
    clash = clash_penalty(state)

    return (
        -0.40 * E
        -0.30 * R
        -0.15 * hS
        +0.10 * pS
        +0.10 * βdiff
        -0.05 * clash
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gym environment
# ─────────────────────────────────────────────────────────────────────────────
def apply_action(state, action):
    new = state.copy()
    new["phi_psi"] = pad_phi_psi(state["phi_psi"] + action, len(action))
    return new

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

        # initial & target states
        st0 = load_structure(af2_cif)
        st0["phi_psi"] = pad_phi_psi(st0["phi_psi"], fixed_dim)
        self.state = st0

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("seq", af2_cif)
        chain = next(structure[0].get_chains())
        self.sequence = ''.join(
          three_to_one.get(res.get_resname(), 'A')
          for res in chain if is_aa(res)
        )

        if inference_mode or exp_cif is None:
            self.target = st0.copy()
        else:
            tgt = load_structure(exp_cif)
            tgt["phi_psi"] = pad_phi_psi(tgt["phi_psi"], fixed_dim)
            self.target = tgt

        self.action_space = spaces.Box(-20.0, 20.0, (fixed_dim,), np.float32)
        self.observation_space = spaces.Box(-180.0, 180.0, (fixed_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        st = load_structure(self.af2_cif)
        st["phi_psi"] = pad_phi_psi(st["phi_psi"], self.fixed_dim)
        self.state = st
        return self.state["phi_psi"], {}

    def _maybe_recompute_sasa(self):
      if not self.sasa_period or (self.current_step % self.sasa_period != 0):
          return

      phi_psi_vec = self.state["phi_psi"]
      max_pairs = len(phi_psi_vec) // 2
      N = min(len(self.sequence), max_pairs)
      if N == 0:
          return

      seq = self.sequence[:N]  # true one-letter sequence
      flat = phi_psi_vec[:2 * N]
      phi_list = np.radians(flat[0::2])                     # length N
      psi_actual = np.radians(flat[1::2])                   # length N
      psi_im1 = [np.radians(120.0)] + psi_actual[:-1].tolist()  # also length N

      if len(phi_list) != len(psi_im1) or len(seq) != len(phi_list):
          if self.current_step == 1:
              print(f"[SASA] Skipping: φ={len(phi_list)} ψ_im1={len(psi_im1)} seq={len(seq)}")
          return

      model = make_structure(seq, phi=phi_list, psi_im1=psi_im1)
      if model is None:
          print("[SASA] make_structure() returned None; skipping")
          return

      buf = io.StringIO()
      writer = PDBIO()
      writer.set_structure(model)
      writer.save(buf)

      xyz_list = []
      for atom in model.get_atoms():
          xyz_list.append(atom.get_coord())
      self.state["coordinates"] = np.asarray(xyz_list, dtype=np.float32)

      with tempfile.NamedTemporaryFile("w+", suffix=".pdb") as fh:
          fh.write(buf.getvalue())
          fh.flush()
          struct_tmp = PDBParser(QUIET=True).get_structure("tmp", fh.name)
          sasa_vec = _sasa_per_residue(struct_tmp)

      self.state["sasa"] = sasa_vec

      # rebuild hydrophobic/polar masks based on sequence
      hyd_set = {"L", "I", "V", "A", "M"}
      pol_set = {"R", "K", "D", "E", "Q", "N"}
      self.state["hydro_mask"] = np.array([1 if aa in hyd_set else 0 for aa in seq], dtype=np.float32)
      self.state["polar_mask"] = np.array([1 if aa in pol_set else 0 for aa in seq], dtype=np.float32)




    def step(self, action):
        self.current_step += 1
        self.state = apply_action(self.state, action)
        self._maybe_recompute_sasa()

        if self.i_mode:
            rew = 0.0
        else:
            rew = reward_func(self.state, self.target)

        hS, pS = compute_sasa_terms(self.state)
        info = {
            "step": self.current_step,
            "energy": compute_energy(self.state),
            "rmsd": None if self.i_mode else compute_rmsd(self.state, self.target),
            "hydS": hS,
            "polS": pS,
            "reward": rew,
        }

        done = False
        trunc = (self.current_step >= self.max_steps)
        obs = self.state["phi_psi"]
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

        # infer spaces from one instance
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
