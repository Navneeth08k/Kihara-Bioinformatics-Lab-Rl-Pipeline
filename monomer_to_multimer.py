from Bio.PDB import PDBParser, PDBIO, Model, Chain
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
import os

INPUT_PDB = "inference_outputs/7yat_reconstructed_minimized.pdb"
OUTPUT_PDB = "inference_outputs/7yat_fibril_multimer.pdb"
NUM_CHAINS = 24
TRANSLATION_Z = 4.7  # Angstroms, typical inter-sheet spacing

parser = PDBParser(QUIET=True)
structure = parser.get_structure("monomer", INPUT_PDB)
monomer_chain = list(structure[0].get_chains())[0]

# Create a new structure
builder = StructureBuilder()
builder.init_structure("fibril")
builder.init_model(0)

for i in range(NUM_CHAINS):
    # Deep copy the chain
    new_chain = Chain.Chain(chr(65 + i))  # Chain IDs A, B, C...
    for residue in monomer_chain:
        new_residue = residue.copy()
        for atom in new_residue:
            coord = atom.get_coord()
            coord[2] += i * TRANSLATION_Z  # Shift along Z-axis
            atom.set_coord(coord)
        new_chain.add(new_residue)
    
    builder.structure[0].add(new_chain)

# Save the multimer
io = PDBIO()
io.set_structure(builder.get_structure())
io.save(OUTPUT_PDB)

print(f"✅ Fibril multimer saved: {OUTPUT_PDB}")

