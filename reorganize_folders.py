#!/usr/bin/env python3

import os
import shutil
import glob

def reorganize_folders():
    """Move all Sample folders into a 'data' directory"""
    # Get the current directory where the script is run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(script_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Find all Sample folders in the current directory
    sample_dirs = glob.glob(os.path.join(script_dir, "Sample *"))
    
    # Move each Sample folder to the data directory
    moved_folders = 0
    for sample_dir in sample_dirs:
        # Get the sample folder name without path
        sample_name = os.path.basename(sample_dir)
        
        # Define the destination path
        destination = os.path.join(data_dir, sample_name)
        
        try:
            # Move the folder
            shutil.move(sample_dir, destination)
            print(f"Moved: {sample_dir} -> {destination}")
            moved_folders += 1
        except Exception as e:
            print(f"Error moving {sample_dir}: {str(e)}")
    
    print(f"\nReorganization complete: Moved {moved_folders} Sample folders to {data_dir}")

if __name__ == "__main__":
    print("Starting folder reorganization...")
    user_input = input("This will move all Sample folders into a 'data' directory. Continue? (y/n): ")
    
    if user_input.lower() == 'y':
        reorganize_folders()
        print("Reorganization complete!")
    else:
        print("Operation cancelled.") 