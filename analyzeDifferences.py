#!/usr/bin/env python3
"""
analyze_all_pairs.py: Batch analyze AF2 vs experimental for all data pairs.

Usage:
    python analyze_all_pairs.py --data-root data --out-dir analysis_results

This script:
 1. Finds all sample pairs under the data root (folders containing AF2/ and Experimental/).
 2. For each pair, aligns AF2 prediction to experimental structure, extracts per-residue φ/ψ, Cα distances, SASA.
 3. Writes a CSV per pair into the out directory, named by sample folder name.
"""
import os
import argparse
import glob
import io
import tempfile
import csv

import numpy as np
import freesasa
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder, PDBIO
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.Polypeptide import is_aa

# ─────────────────────────────────────────────────────────────────────────────
# Reuse helper functions from analyze_af2_vs_exp.py
# ─────────────────────────────────────────────────────────────────────────────

def find_pairs(root):
    pairs = []
    for cur, dirs, _ in os.walk(root):
        if {'AF2', 'Experimental'}.issubset(set(dirs)):
            af2 = first_structure(os.path.join(cur, 'AF2'))
            exp = first_structure(os.path.join(cur, 'Experimental'))
            if af2 and exp:
                pairs.append((cur, af2, exp))
    return pairs


def first_structure(folder):
    for ext in ('*.cif', '*.pdb', '*.CIF', '*.PDB'):
        hits = glob.glob(os.path.join(folder, ext))
        if hits:
            return hits[0]
    return None


def load_structure(path):
    if path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', path)


def sasa_per_residue(structure):
    model = structure[0]
    for i, chain in enumerate(model.get_chains()):
        chain.id = chr(ord('A') + (i % 26))
    pdb_buf = io.StringIO()
    io_obj = PDBIO()
    io_obj.set_structure(model)
    io_obj.save(pdb_buf)
    with tempfile.NamedTemporaryFile('w+', suffix='.pdb') as fh:
        fh.write(pdb_buf.getvalue())
        fh.flush()
        fs_struct = freesasa.Structure(fh.name)
        res_areas = freesasa.calc(fs_struct).residueAreas()
    sasa_map = {}
    for key, area in res_areas.items():
        if isinstance(key, tuple):
            ch, resnum, _ = key
            sasa_map[(ch, resnum)] = float(area.total)
        else:
            parts = key.split(':')
            if len(parts) >= 2:
                ch, resnum = parts[0], int(parts[1])
                sasa_map[(ch, resnum)] = float(area.total)
    chain0 = next(model.get_chains())
    out = []
    for res in chain0:
        if is_aa(res):
            out.append(sasa_map.get((chain0.id, res.id[1]), 0.0))
    return np.array(out, dtype=np.float32)


def extract_phi_psi(structure):
    chain = next(structure[0].get_chains())
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        return [], [], []
    pp = peptides[0]
    angles = pp.get_phi_psi_list()
    res_ids, phi, psi = [], [], []
    for res, (phi_val, psi_val) in zip(pp, angles):
        if phi_val is not None and psi_val is not None:
            res_ids.append(res.id[1])
            phi.append(np.degrees(phi_val))
            psi.append(np.degrees(psi_val))
    return res_ids, phi, psi


def align_structures(pred, ref):
    sup = Superimposer()
    pred_ca = [a for a in pred.get_atoms() if a.get_name()=='CA']
    ref_ca  = [a for a in ref.get_atoms()  if a.get_name()=='CA']
    n = min(len(pred_ca), len(ref_ca))
    sup.set_atoms(ref_ca[:n], pred_ca[:n])
    sup.apply(pred.get_atoms())


def extract_ca_distances(pred, ref):
    pred_chain = next(pred.get_chains())
    ref_chain  = next(ref.get_chains())
    pred_map = {res.id[1]: res['CA'].get_coord() for res in pred_chain if 'CA' in res}
    ref_map  = {res.id[1]: res['CA'].get_coord() for res in ref_chain if 'CA' in res}
    dists = {resnum: float(np.linalg.norm(pred_map[resnum] - ref_map[resnum]))
             for resnum in set(pred_map) & set(ref_map)}
    return dists

# ─────────────────────────────────────────────────────────────────────────────
# Main batch processing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir',   required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = find_pairs(args.data_root)
    for sample_dir, af2_path, exp_path in pairs:
        name = os.path.basename(sample_dir.rstrip('/'))
        out_csv = os.path.join(args.out_dir, f"{name}_differences.csv")

        struct_pred = load_structure(af2_path)
        struct_exp  = load_structure(exp_path)
        align_structures(struct_pred, struct_exp)

        ids_pred, phi_pred, psi_pred = extract_phi_psi(struct_pred)
        _,       phi_exp,  psi_exp   = extract_phi_psi(struct_exp)
        ca_dists  = extract_ca_distances(struct_pred, struct_exp)
        sasa_pred = sasa_per_residue(struct_pred)
        sasa_exp  = sasa_per_residue(struct_exp)

        with open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['residue','phi_pred','phi_exp','dphi',
                             'psi_pred','psi_exp','dpsi','ca_dist',
                             'sasa_pred','sasa_exp','dsasa'])
            for i,resnum in enumerate(ids_pred):
                pe = phi_exp[i] if i<len(phi_exp) else ''
                se = psi_exp[i] if i<len(psi_exp) else ''
                dphi = phi_pred[i] - pe if pe!='' else ''
                dpsi = psi_pred[i] - se if se!='' else ''
                cad  = ca_dists.get(resnum, '')
                sa1  = sasa_pred[i] if i<len(sasa_pred) else ''
                sa2  = sasa_exp[i]  if i<len(sasa_exp)  else ''
                ds   = sa1 - sa2 if sa1!='' and sa2!='' else ''
                writer.writerow([resnum, phi_pred[i], pe, dphi,
                                 psi_pred[i], se, dpsi, cad,
                                 sa1, sa2, ds])
        print(f"Wrote {out_csv}")

if __name__ == '__main__':
    main()
