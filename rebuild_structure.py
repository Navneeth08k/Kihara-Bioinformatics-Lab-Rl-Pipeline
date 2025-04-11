import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBIO import PDBIO
import PeptideBuilder
from PeptideBuilder import Geometry, make_structure

# Manual 3-letter → 1-letter conversion
three_to_one_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
def three_to_one(resname):
    return three_to_one_map.get(resname.upper(), 'X')

def extract_sequence_from_cif(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    chain = next(structure[0].get_chains())
    sequence = ""
    for residue in chain:
        if is_aa(residue, standard=True):
            sequence += three_to_one(residue.get_resname())
    return sequence

def rebuild_structure_from_angles(original_cif_path, new_phi_psi, output_pdb_path):
    sequence = extract_sequence_from_cif(original_cif_path)
    num_residues = len(sequence)
    usable_residues = num_residues - 2
    expected_len = usable_residues * 2

    if len(new_phi_psi) < expected_len:
        raise ValueError(f"Not enough angles. Expected {expected_len}, got {len(new_phi_psi)}")

    # Build residue-by-residue geometry manually
    phi_angles = []
    psi_angles = []

    # Add default value for first residue's phi (in radians)
    phi_angles.append(np.radians(-120.0))  # Default beta-sheet phi

    # Add angles for middle residues
    for i in range(1, num_residues - 1):
        phi = new_phi_psi[(i - 1) * 2]
        psi = new_phi_psi[(i - 1) * 2 + 1]
        phi_angles.append(np.radians(phi))
        psi_angles.append(np.radians(psi))

    # Add default value for last residue's psi (in radians)
    psi_angles.append(np.radians(120.0))  # Default beta-sheet psi

    # Build and save structure
    structure = make_structure(sequence, phi=phi_angles, psi_im1=psi_angles)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)
    print(f"✅ Structure saved to {output_pdb_path}")
