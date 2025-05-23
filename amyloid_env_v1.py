#This amyloid_env_v1 file is an environment where rather than calculating all the metrics and rewards based on
#strictly the backbone, we are rebuilding the whole protein as this gives us a more full picture
#of how all these metrics impact rmsd etc... The thinking is that this will give a stronger
#signal for learning.

import io
import os
import random
import tempfile

import numpy as np
import freesasa
import gymnasium as gym
from gymnasium import spaces

from Bio.PDB.Structure import Structure
from Bio.PDB.Model     import Model
from Bio.PDB.Chain     import Chain

from Bio.PDB import MMCIFParser, PDBParser, PPBuilder, PDBIO
from Bio.PDB.Polypeptide import is_aa
from PeptideBuilder import make_structure

from Bio.PDB.Superimposer import Superimposer         # new
from scipy.spatial.distance import cdist              # new
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
# or simple χ1/χ2 dihedral check; reward: -0.01 * n_outliers
from Bio.PDB.vectors import calc_dihedral # <-- NEW

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
    """
    Compute per‐residue SASA for **one chain** (the first chain), 
    but *in the context* of all its neighbors (if any).
    """
    # 1) Identify the chain we care about
    model  = structure[0]                  # first Model of the Structure
    # ── PDBIO only supports single‐char chain IDs ──
    for i, chain in enumerate(model.get_chains()):
        # map to 'A','B','C',… wrapping around if needed
        chain.id = chr(ord('A') + (i % 26))
    chain0 = next(model.get_chains())     # grab chain A
    
    # 2) Write *entire* model (all chains) to PDB for FreeSASA
    pdb_buf = io.StringIO()
    writer  = PDBIO()
    writer.set_structure(model)           # ← note: model, not chain0
    writer.save(pdb_buf)
    
    # 3) Run FreeSASA on that PDB
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as fh:
        fh.write(pdb_buf.getvalue())
        fh.flush()
        fs_struct = freesasa.Structure(fh.name)
        res_areas = freesasa.calc(fs_struct).residueAreas()
    
    # 4) Build a mapping (chainID, resSeq) → area.total
    sasa_map = {}
    sample_key, sample_val = next(iter(res_areas.items()))
    if isinstance(sample_val, dict):
        for ch, inner in res_areas.items():
            for seq, area in inner.items():
                sasa_map[(ch, int(seq))] = float(area.total)
    else:
        for key, area in res_areas.items():
            # "A:18:ARG" or similar
            parts = key.split(":")
            if len(parts) == 3:
                ch, seq, _ = parts
                sasa_map[(ch, int(seq))] = float(area.total)
    
    # 5) Extract per‐residue SASA *for that first chain* in order
    out = []
    for residue in chain0:
        if is_aa(residue):
            key = (chain0.id, residue.id[1])
            out.append(sasa_map.get(key, 0.0))
    return np.array(out, dtype=np.float32)

def strand_register_offset(pred_chain, exp_chain, axis=0):
    pred_z = np.array([a.get_coord()[axis] for a in pred_chain.get_atoms()
                       if a.get_name() == "CA"])
    exp_z  = np.array([a.get_coord()[axis] for a in exp_chain.get_atoms()
                       if a.get_name() == "CA"])
    n = min(len(pred_z), len(exp_z))
    off = np.round(np.mean(pred_z[:n] - exp_z[:n]) / 3.4)   # ~3.4 Å per residue
    return abs(off)

def broken_hbonds(struct):
    atoms = [a for a in struct.get_atoms() if a.element in ("O","N")]
    ns = NeighborSearch(atoms)
    broken = sum(1 for a in atoms if len(ns.search(a.get_coord(), 2.7)) == 1) # only itself
    return broken

# ─────────────────────────────────────────────────────────────────────────────
# load a single CIF → features dict (angles, coords, masks, sasa)
# ─────────────────────────────────────────────────────────────────────────────
def load_structure(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)
    chain = next(structure[0].get_chains())

    # φ/ψ  residue indices
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




# ─────────────────────────────────────────────────────────────────────────────
# apply_action helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_action(state, action):
    new = state.copy()
    new["phi_psi"] = pad_phi_psi(state["phi_psi"]  + action, len(action))
    return new


# ───────────────────────── fibril helpers ──────────────────────────
FIBRIL_LATTICE = np.array([4.9, 0.0, 0.0])   # Å  ← replace with your true lattice

def build_full_chain(seq, phi, psi):
    """Backboneside-chains with PeptideBuilder."""
    model = make_structure(seq, phi=phi, psi_im1=[np.radians(120)] + psi[:-1].tolist())
    if model is None:
        raise ValueError("PeptideBuilder failed.")
    buf, io_obj = io.StringIO(), PDBIO()
    io_obj.set_structure(model); io_obj.save(buf)
    return PDBParser(QUIET=True).get_structure("full", io.StringIO(buf.getvalue()))

def replicate_chain(chain_struct, n_copy=3):
    """Return list of translated Bio.PDB models (including original)."""
    copies = [chain_struct]
    for i in range(1, n_copy):
        copy = chain_struct.copy()
        for atom in copy.get_atoms():
            atom.set_coord(atom.get_coord() + i * FIBRIL_LATTICE)
        copies.append(copy)
    return copies

# ───────── correct χ1 outlier counter ─────────
def rotamer_outliers(struct):
    """
    Count χ1 rotamer outliers.
    Uses N-CA-CB-X  dihedral, where X is CG/OG1/SG… depending on residue.
    A value outside ±60° (gauche±) or 180° (trans) ± 20° is flagged.
    """
    # which fourth atom to use for each residue
    chi1_atom = {
        "SER": "OG",  "THR": "OG1", "CYS": "SG",
        "ILE": "CG1", "VAL": "CG1",
        # default CG for the rest
    }

    out = 0
    for res in struct[0].get_residues():
        if not is_aa(res) or res.get_resname() in ("GLY",):  # no χ1
            continue
        try:
            n  = res["N"].get_vector()
            ca = res["CA"].get_vector()
            cb = res["CB"].get_vector()
            cg_name = chi1_atom.get(res.get_resname(), "CG")
            cg = res[cg_name].get_vector()
        except KeyError:
            continue                   # missing atoms → skip

        chi1 = calc_dihedral(n, ca, cb, cg) * 180.0 / np.pi   # radians→deg
        # bring to −180…180 then test windows
        chi1 = (chi1 + 180) % 360 - 180
        if (abs(chi1) > 100) or (20 < abs(chi1) < 140):
            out += 1
    return out


def assemble_models(models):
    """
    Merge several single‐chain Structure objects into one Structure
    with multiple chains (chain IDs A, B, C, …).
    """
    # create a new top‐level Structure and one Model
    comb = Structure("combined")
    mdl  = Model(0)
    comb.add(mdl)

    for idx, m in enumerate(models):
        # m is a Structure; grab its first (and only) chain
        orig_chain = next(m.get_chains())
        # give it a new chain ID (A, B, C...)
        new_chain = Chain(chr(ord("A")  + idx))
        # deep‐copy residues into the new chain
        for residue in orig_chain:
            new_chain.add(residue.copy())
        mdl.add(new_chain)

    return comb

def buried_interface_sasa(copies):
    single = sum(_sasa_per_residue(m) for m in copies)      # Σ chain SASA
    assembled = _sasa_per_residue(assemble_models(copies))  # complex SASA
    return float(single.sum() - assembled.sum())

def trimer_rmsd(pred_copies, exp_trimer):
    """
    Compute Cα‐only RMSD between predicted trimer (list of Structures)
    and experimental multimer (Structure).
    """
    sup = Superimposer()

    # 1) collect experimental Cα Atom objects
    fixed_atoms = [
        atom
        for atom in exp_trimer.get_atoms()
        if atom.get_name() == "CA"
    ]

    # 2) collect predicted Cα Atom objects across all copies
    moving_atoms = [
        atom
        for m in pred_copies
        for atom in m.get_atoms()
        if atom.get_name() == "CA"
    ]

    # 3) trim to the same length to avoid size mismatches
    n = min(len(fixed_atoms), len(moving_atoms))
    fixed_atoms  = fixed_atoms[:n]
    moving_atoms = moving_atoms[:n]

    # 4) superimpose and return the RMSD
    sup.set_atoms(fixed_atoms, moving_atoms)
    return float(sup.rms)


def inter_chain_clash(copies, thr=2.0):
    xyz = np.concatenate([np.vstack([a.get_coord() for a in m.get_atoms()]) for m in copies])
    # index window to avoid self-pairs
    idx_split = np.cumsum([len(list(m.get_atoms())) for m in copies])
    clashes = 0.0
    start = 0
    for end in idx_split[:-1]:
        d = cdist(xyz[start:end], xyz[end:], "euclidean")
        clashes += np.clip(thr - d, 0.0, 0.5).sum()
        start = end
    return float(clashes)

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
        # normalization / clipping caps (per‐residue)
        self.hb_cap_per_res      = 0.1   # clip H-bond diffs at 0.1/bond-per-res
        self.rot_cap_per_res     = 0.2   # clip rotamer diffs at 0.2/outlier-per-res
        self.max_clash_ic        = 600   # expected max inter‐chain clashes for normalization
        
        # curriculum phase boundaries (in env steps)
        self.phase1_steps        = 50_000
        self.phase2_steps        = 150_000
        # phase3 = everything beyond phase2
        # ————————————————————————————————————————————————————————————————

        # ----- initial state -----
        st0 = load_structure(self.af2_cif)
        raw_phi = st0["phi_psi"]
        valid_len = raw_phi.shape[0]
        st0["valid_len"] = valid_len
        mask = np.zeros(self.fixed_dim, dtype=np.float32)
        mask[:valid_len] = 1.0
        st0["mask"] = mask
        st0["phi_psi"] = pad_phi_psi(raw_phi, self.fixed_dim)
        st0["hydro_mask"]= pad_phi_psi(st0["hydro_mask"], self.fixed_dim)
        st0["polar_mask"]= pad_phi_psi(st0["polar_mask"], self.fixed_dim)
        st0["sasa"] = pad_phi_psi(st0["sasa"], self.fixed_dim)
        
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

       

        # load experimental trimer once for RMSD alignment
        # in __init__, after you set self.sequence
        if self.exp_cif and not self.i_mode:
            parser_exp = MMCIFParser(QUIET=True)
            self.exp_trimer = parser_exp.get_structure("exp", self.exp_cif)
        else:
            self.exp_trimer = None
        
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

        # now we have exp_trimer and sequence – safe to build trimer once
        self._rebuild_trimer()
        self.prev_hyd = self.state["hyd_buried"]
        self.prev_pol = self.state["pol_buried"]
        self.prev_buried = self.state["buried"]

        # action space
        self.action_space = spaces.Box(-20.0, 20.0, (self.fixed_dim,), np.float32)
        self.render_mode = None

        core_len = 2 * self.fixed_dim          # φ/ψ  hydro_mask
        low_core = np.concatenate([
            np.full(self.fixed_dim, -1.0, dtype=np.float32),   # φ/ψ_norm
            np.zeros(self.fixed_dim, dtype=np.float32),        # hydro_mask
        ])
        high_core = np.concatenate([
            np.ones(self.fixed_dim, dtype=np.float32),         # φ/ψ_norm
            np.ones(self.fixed_dim, dtype=np.float32),         # hydro_mask
        ])

        low  = np.concatenate([low_core,  np.array([0.0, 0.0])])
        high = np.concatenate([high_core, np.array([1.0, 1.0])])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        



    def _core_obs(self):
        """
        Base observation = [φ/ψ (-180…180 → –1…1) | hydro_mask]  (length 2·fixed_dim)
        """
        phi_norm   = self.state["phi_psi"] / 180.0                     # (fixed_dim,)
        hydro_pad  = pad_phi_psi(self.state["hydro_mask"], self.fixed_dim)
        return np.concatenate([phi_norm, hydro_pad]).astype(np.float32)


    def _build_obs(self):
        core  = self._core_obs()
        extra = np.array([
            self.state.get("buried", 0.0)  / 300.0,    # rough normalisation
            self.state.get("R_rmsd", 0.0)  / 10.0,
        ], dtype=np.float32)
        return np.concatenate([core, extra], axis=0)



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
        # pad the newly loaded masks & sasa so they match fixed_dim
        self.state["hydro_mask"] = pad_phi_psi(self.state["hydro_mask"], self.fixed_dim)
        self.state["polar_mask"] = pad_phi_psi(self.state["polar_mask"], self.fixed_dim)
        self.state["sasa"]       = pad_phi_psi(self.state["sasa"],       self.fixed_dim)
        self._rebuild_trimer()
        self.prev_hyd = self.state["hyd_buried"]
        self.prev_pol = self.state["pol_buried"]
        self.prev_buried = self.state["buried"]

        # reset previous SASA
        h0, p0 = compute_sasa_terms(self.state)
        self.prev_hS, self.prev_pS = h0, p0
        obs = self._build_obs()
        return obs, {}

    def _rebuild_trimer(self):
        """Rebuild full-atom monomer → replicate → cache SASA, RMSD, clash."""
        
        phi_psi_vec = self.state["phi_psi"]
        max_pairs   = len(phi_psi_vec) // 2
        N           = min(len(self.sequence), max_pairs)
        if N == 0:
            return
        seq  = self.sequence[:N]
        phi  = np.radians(phi_psi_vec[:2*N:2])
        psi  = np.radians(phi_psi_vec[1:2*N:2])

        mono = build_full_chain(seq, phi, psi)
        copies = replicate_chain(mono, n_copy=3)
        mono_sasa = _sasa_per_residue(mono)              # (N,)
        tri_sasa  = _sasa_per_residue(assemble_models(copies))

        # ⬇︎ keep monomer SASA current so compute_sasa_terms is meaningful
        # pad the updated monomer SASA to fixed_dim so compute_sasa_terms works
        self.state["sasa"] = pad_phi_psi(mono_sasa, self.fixed_dim)

        # interface burial per residue
        buried_per_res = mono_sasa - tri_sasa

        # split into hydrophobic vs polar
        hyd_buried = float((buried_per_res * self.state["hydro_mask"][:len(buried_per_res)]).sum())
        pol_buried = float((buried_per_res * self.state["polar_mask"][:len(buried_per_res)]).sum())

        self.state.update({
            "hyd_buried": hyd_buried,
            "pol_buried": pol_buried,
            "total_buried": float(buried_per_res.sum()),
            # keep clash/R_rmsd you already cache
        })
        
        self.state["copies"] = copies                       # cache for clash
        self.state["buried"] = buried_interface_sasa(copies)

        if self.exp_trimer:
            self.state["R_rmsd"] = trimer_rmsd(copies, self.exp_trimer)
        else:
            self.state["R_rmsd"] = 0.0
        
        # -------- NEW quality terms ----------------------------------------
        if self.exp_trimer:
            exp_chain  = next(self.exp_trimer.get_chains())
            pred_chain = next(copies[0].get_chains())            # first replica
            self.state["reg_off"]  = strand_register_offset(pred_chain, exp_chain)
        else:
            self.state["reg_off"]  = 0.0

        self.state["hb_err"]   = broken_hbonds(copies[0])        # monomer quality
        self.state["rot_out"]  = rotamer_outliers(copies[0])
        # -------------------------------------------------------------------


        self.state["clash_ic"] = inter_chain_clash(copies)


    def step(self, action):
        """
        Apply an action (delta φ/ψ), rebuild SASA/clash as needed, and return
        (obs, reward, done, trunc, info) under a 3-phase curriculum:
          1) 0–50k steps: burial shaping only
          2) 50–150k: +register/RMSD & burial/H-bonds
          3) >150k: +rotamer & clash penalties
        """
        # 1) bookkeeping & apply action
        self.global_step  += 1
        self.current_step += 1
        self.state = apply_action(self.state, action)
        if self.global_step % self.sasa_period == 0:
            self._rebuild_trimer()

        # 2) compute raw signals
        E        = compute_energy(self.state)
        R        = compute_rmsd(self.state, self.target)
        hS, pS   = compute_sasa_terms(self.state)
        Δhyd     = self.state["hyd_buried"] - self.prev_hyd
        Δpol     = self.prev_pol - self.state["pol_buried"]
        reg_raw  = abs(self.state["reg_off"])
        hb_raw   = self.state["hb_err"]
        rot_raw  = self.state["rot_out"]
        clash_raw= self.state["clash_ic"]

        # update SASA baseline
        self.prev_hyd, self.prev_pol = self.state["hyd_buried"], self.state["pol_buried"]

        # 3) normalize & clip
        valid_len   = max(self.state["valid_len"], 1)
        reg_norm    = reg_raw / valid_len
        hb_norm     = min(hb_raw / valid_len,   self.hb_cap_per_res)
        rot_norm    = min(rot_raw / valid_len,  self.rot_cap_per_res)
        clash_norm  = min(clash_raw / self.max_clash_ic, 1.0)
        rmsd_norm   = self.state["R_rmsd"] / 180.0
        energy_norm = (E / 1000.0) / 10.0

        dh_per_res = max(Δhyd, 0.0) / valid_len
        dp_per_res = max(Δpol, 0.0) / valid_len

        # 4) curriculum‐aware reward
        gs = self.global_step
        if gs < self.phase1_steps:
            # Phase 1: burial shaping only
            rew = +0.10 * dh_per_res + 0.05 * dp_per_res

        elif gs < self.phase2_steps:
            # Phase 2: add register/RMSD & H-bond/SASA
            rew = (
                +0.10 * dh_per_res
                +0.05 * dp_per_res
                -1.00 * reg_norm
                -1.00 * hb_norm
                -1.00 * rmsd_norm
                -1.00 * clash_norm    # optional here or in Phase 3
                -0.10 * energy_norm
            )

        else:
            # Phase 3: full penalties (incl. rotamers)
            rew = (
                +0.10 * dh_per_res
                +0.05 * dp_per_res
                -1.00 * reg_norm
                -1.00 * hb_norm
                -0.50 * rot_norm
                -1.00 * clash_norm
                -1.00 * rmsd_norm
                -0.10 * energy_norm
            )

        # 5) truncation & info dict
        done  = False
        trunc = (self.current_step >= self.max_steps)
        info = {
            "step":        self.current_step,
            "global_step": self.global_step,
            "energy":      E,
            "rmsd":        R,
            "hyd_buried":  self.state["hyd_buried"],
            "pol_buried":  self.state["pol_buried"],
            "reg_off":     self.state["reg_off"],
            "hb_err":      self.state["hb_err"],
            "rot_out":     self.state["rot_out"],
            "clash":       self.state["clash_ic"],
            "Δhyd":        Δhyd,
            "Δpol":        Δpol,
            "reward":      rew,
        }
        info.update({
            "dh_per_res":   dh_per_res,
            "dp_per_res":   dp_per_res,
            "reg_norm":     reg_norm,
            "hb_norm":      hb_norm,
            "rot_norm":     rot_norm,
            "clash_norm":   clash_norm,
            "rmsd_norm":    rmsd_norm,
            "energy_norm":  energy_norm,
        })
        # 6) build obs & return
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
        self.render_mode = None


    def reset(self, **kwargs):
        af2, exp = random.choice(self.pairs)
        self.current_env = AmyloidEnv(af2, exp, **self.kw)
        return self.current_env.reset(**kwargs)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode="human"):
        return self.current_env.render(mode)
