import os
import json
import argparse
from pathlib import Path

def generate_mapping(mesh_dir):
    """
    Scans a directory and creates a mapping using the EXACT folder name as the key.
    Example: 
    - Folder: 'apple_1' -> Key: 'apple_1'
    - Folder: 'apple_2' -> Key: 'apple_2'
    """
    target_path = Path(mesh_dir).resolve()
    
    if not target_path.exists():
        print(f"Error: Directory not found: {target_path}")
        return

    print(f"Scanning directory: {target_path}")

    # 1. Get all folder names
    folders = sorted([f.name for f in target_path.iterdir() if f.is_dir()])
    
    # 2. Assign IDs
    # This creates a dictionary like: {'apple_1': 0, 'apple_2': 1, 'banana_1': 2}
    mapping = {name: i+1 for i, name in enumerate(folders)}

    print(f"Found {len(mapping)} unique folders.")
    if mapping:
        print("Sample keys:", list(mapping.keys())[:5])

    # 3. Save
    output_file = target_path / "type_mapping.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4: Makes it multi-line and pretty
            # sort_keys=True: Ensures the keys are alphabetical
            # ensure_ascii=False: Keeps special characters readable
            json.dump(mapping, f, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"✅ Success! Mapping saved to: {output_file}")
    except IOError as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    
    generate_mapping(args.dir)