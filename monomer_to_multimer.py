import os
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Chain
from Bio.PDB.StructureBuilder import StructureBuilder

def main(num_chains=24):
    OUTPUT_DIR = "inference_outputs"
    TRANSLATION_Z = 4.7  # Å — inter-sheet spacing

    # Find latest minimized structure
    pdb_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_reconstructed_minimized.pdb")])
    if not pdb_files:
        raise FileNotFoundError("❌ No minimized PDB found in inference_outputs!")

    INPUT_PDB = os.path.join(OUTPUT_DIR, pdb_files[-1])
    basename = pdb_files[-1].replace("_reconstructed_minimized.pdb", "")
    OUTPUT_PDB = os.path.join(OUTPUT_DIR, f"{basename}_fibril_multimer.pdb")

    # Parse and copy monomer
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("monomer", INPUT_PDB)
    monomer_chain = list(structure[0].get_chains())[0]

    # Build multimer structure
    builder = StructureBuilder()
    builder.init_structure("fibril")
    builder.init_model(0)

    for i in range(num_chains):
        new_chain = Chain.Chain(chr(65 + i))  # A, B, C...
        for residue in monomer_chain:
            new_residue = residue.copy()
            for atom in new_residue:
                coord = atom.get_coord()
                coord[2] += i * TRANSLATION_Z
                atom.set_coord(coord)
            new_chain.add(new_residue)
        builder.structure[0].add(new_chain)

    # Save result
    io = PDBIO()
    io.set_structure(builder.get_structure())
    io.save(OUTPUT_PDB)
    print(f"✅ Fibril multimer saved: {OUTPUT_PDB}")

# Prevent auto-running
if __name__ == "__main__":
    main()
