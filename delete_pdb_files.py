import os
import glob

def delete_pdb_file(pdb_file_path):
    """Delete a PDB file if its corresponding CIF file exists"""
    # Get the potential CIF file path
    file_name = os.path.splitext(os.path.basename(pdb_file_path))[0]
    dir_path = os.path.dirname(pdb_file_path)
    cif_file_path = os.path.join(dir_path, f"{file_name}.cif")
    
    # Check if the corresponding CIF file exists
    if os.path.exists(cif_file_path):
        try:
            # Delete the PDB file
            os.remove(pdb_file_path)
            print(f"Deleted: {pdb_file_path}")
            return True
        except Exception as e:
            print(f"Error deleting {pdb_file_path}: {str(e)}")
            return False
    else:
        print(f"Skipping {pdb_file_path}: No corresponding CIF file found")
        return False

def find_and_delete_pdb_files():
    """Find all PDB files and delete them if they have corresponding CIF files"""
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
    deleted_files = 0
    
    # Iterate through each sample folder
    for sample_dir in sample_dirs:
        # Find all PDB files recursively in the sample folder
        pdb_files = glob.glob(os.path.join(sample_dir, "**", "*.pdb"), recursive=True)
        
        total_files += len(pdb_files)
        
        # Delete each PDB file if it has a corresponding CIF file
        for pdb_file in pdb_files:
            if delete_pdb_file(pdb_file):
                deleted_files += 1
    
    print(f"\nSummary: Deleted {deleted_files} of {total_files} PDB files")

if __name__ == "__main__":
    print("Starting PDB file deletion...")
    user_input = input("This will delete PDB files that have corresponding CIF files. Continue? (y/n): ")
    
    if user_input.lower() == 'y':
        find_and_delete_pdb_files()
        print("Deletion complete!")
    else:
        print("Operation cancelled.") 