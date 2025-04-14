import os
import sys
import numpy as np
import pandas as pd

# Add local pdbfixer path
sys.path.append('./pdbfixer')
from pdbfixer import PDBFixer

from PeptideBuilder import make_structure
from Bio.PDB.PDBIO import PDBIO
from openmm.app import Modeller, ForceField, Simulation, PDBFile, NoCutoff
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond, nanometer

# ==== Constants ====
DEFAULT_PHI_START = -120.0
DEFAULT_PSI_END = 120.0
SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDV"
CSV_PATH = "inference_outputs/7yat_refined_angles.csv"
OUTPUT_DIR = "inference_outputs"
BACKBONE_PDB = os.path.join(OUTPUT_DIR, "7yat_reconstructed.pdb")
FINAL_PDB = os.path.join(OUTPUT_DIR, "7yat_reconstructed_minimized.pdb")

# ==== Step 1: Load angles ====
df = pd.read_csv(CSV_PATH)

phi_angles = [np.radians(DEFAULT_PHI_START)]
psi_angles = []

for i in range(1, len(SEQUENCE) - 1):
    phi = df.iloc[i - 1]["Phi"]
    psi = df.iloc[i - 1]["Psi"]
    phi_angles.append(np.radians(phi))
    psi_angles.append(np.radians(psi))
psi_angles.append(np.radians(DEFAULT_PSI_END))

# ==== Step 2: Build backbone ====
structure = make_structure(SEQUENCE, phi=phi_angles, psi_im1=psi_angles)
os.makedirs(OUTPUT_DIR, exist_ok=True)
io = PDBIO()
io.set_structure(structure)
io.save(BACKBONE_PDB)
print(f"✅ Backbone saved: {BACKBONE_PDB}")

# ==== Step 3: Fix structure ====
fixer = PDBFixer(filename=BACKBONE_PDB)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(pH=7.0)

forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# === Step 4: Dry Minimization (NO solvent) ===
try:
    print("🛠 Preparing dry structure (no water)...")
    modeller = Modeller(fixer.topology, fixer.positions)

    print("🔧 Building system...")
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=None)

    print("⚙️ Creating integrator...")
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picosecond)

    print("🧪 Setting up simulation...")
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    print("🔬 Minimizing dry structure...")
    simulation.minimizeEnergy(tolerance=10.0, maxIterations=500)

    print("💾 Extracting minimized positions...")
    positions = simulation.context.getState(getPositions=True).getPositions()

    print("📦 Saving final minimized structure...")
    with open(FINAL_PDB, 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)

    print(f"✅ Final structure saved: {FINAL_PDB}")
except Exception as e:
    print(f"❌ ERROR during dry minimization: {e}")
