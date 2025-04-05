#!/usr/bin/env python3

import os
import glob
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFIO

def convert_pdb_to_cif(pdb_file_path):
    """Convert a PDB file to CIF format"""
    # Create parser and IO objects
    parser = PDBParser(QUIET=True)
    mmcif_io = MMCIFIO()
    
    # Get file name without extension
    file_name = os.path.splitext(os.path.basename(pdb_file_path))[0]
    
    # Get the directory of the pdb file
    dir_path = os.path.dirname(pdb_file_path)
    
    # Output path for the CIF file (same directory as the PDB)
    cif_file_path = os.path.join(dir_path, f"{file_name}.cif")
    
    try:
        # Parse the PDB file
        structure = parser.get_structure(file_name, pdb_file_path)
        
        # Write as CIF file
        mmcif_io.set_structure(structure)
        mmcif_io.save(cif_file_path)
        
        print(f"Converted: {pdb_file_path} -> {cif_file_path}")
        return True
    except Exception as e:
        print(f"Error converting {pdb_file_path}: {str(e)}")
        return False

def find_and_convert_pdb_files():
    """Find all PDB files in sample folders and convert them to CIF format"""
    # Get the current directory where the script is run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the data directory
    data_dir = os.path.join(script_dir, "data")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
        
    # Find all sample folders within the data directory
    sample_dirs = glob.glob(os.path.join(data_dir, "Sample *"))
    
    if not sample_dirs:
        print(f"No Sample folders found in {data_dir}")
        return
    
    # Initialize counters
    total_files = 0
    converted_files = 0
    
    # Iterate through each sample folder
    for sample_dir in sample_dirs:
        # Find all PDB files recursively in the sample folder
        pdb_files = glob.glob(os.path.join(sample_dir, "**", "*.pdb"), recursive=True)
        
        total_files += len(pdb_files)
        
        # Convert each PDB file
        for pdb_file in pdb_files:
            if convert_pdb_to_cif(pdb_file):
                converted_files += 1
    
    print(f"\nSummary: Converted {converted_files} of {total_files} PDB files to CIF format")

if __name__ == "__main__":
    print("Starting PDB to CIF conversion...")
    find_and_convert_pdb_files()
    print("Conversion complete!") 