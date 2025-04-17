import os
import glob
import numpy as np
import pandas as pd

from pdbfixer import PDBFixer
from PeptideBuilder import make_structure
from Bio.PDB.PDBIO import PDBIO
from openmm.app import Modeller, ForceField, Simulation, PDBFile, NoCutoff
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond

def main():
    # ==== Define constants and output locations ====
    BASE_DIR = "inference_outputs"
    os.makedirs(BASE_DIR, exist_ok=True)

    # ==== Step 1: Load latest refined CSV ====
    csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith("_refined_angles.csv")])
    if not csv_files:
        raise FileNotFoundError("❌ No refined_angles.csv file found in inference_outputs!")

    CSV_PATH = os.path.join(BASE_DIR, csv_files[-1])
    basename = csv_files[-1].replace("_refined_angles.csv", "")

    # ==== Step 2: Detect and parse FASTA ====
    fasta_candidates = sorted(glob.glob("*.fasta") + glob.glob("*.fa"))
    if not fasta_candidates:
        raise FileNotFoundError("❌ No FASTA file found in current directory.")
    fasta_path = fasta_candidates[-1]

    with open(fasta_path, "r") as f:
        lines = f.readlines()
        SEQUENCE = "".join([line.strip() for line in lines if not line.startswith(">")])

    # ==== Step 3: Define output filenames ====
    BACKBONE_PDB = os.path.join(BASE_DIR, f"{basename}_reconstructed.pdb")
    FINAL_PDB = os.path.join(BASE_DIR, f"{basename}_reconstructed_minimized.pdb")

    # ==== Step 4: Load angles and construct geometry ====
    df = pd.read_csv(CSV_PATH)
    DEFAULT_PHI_START = -120.0
    DEFAULT_PSI_END = 120.0

    phi_angles = [np.radians(DEFAULT_PHI_START)]
    psi_angles = []

    for i in range(1, len(SEQUENCE) - 1):
        phi = df.iloc[i - 1]["Phi"]
        psi = df.iloc[i - 1]["Psi"]
        phi_angles.append(np.radians(phi))
        psi_angles.append(np.radians(psi))
    psi_angles.append(np.radians(DEFAULT_PSI_END))

    # ==== Step 5: Build backbone ====
    structure = make_structure(SEQUENCE, phi=phi_angles, psi_im1=psi_angles)
    io = PDBIO()
    io.set_structure(structure)
    io.save(BACKBONE_PDB)
    print(f"✅ Backbone saved: {BACKBONE_PDB}")

    # ==== Step 6: Fix and minimize structure ====
    fixer = PDBFixer(filename=BACKBONE_PDB)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    try:
        modeller = Modeller(fixer.topology, fixer.positions)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff)
        integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picosecond)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance=10.0, maxIterations=500)

        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(FINAL_PDB, 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)

        print(f"✅ Final structure saved: {FINAL_PDB}")
    except Exception as e:
        print(f"❌ ERROR during dry minimization: {e}")

# Prevent running on import
if __name__ == "__main__":
    main()
