import os
import zipfile
import rarfile
from pathlib import Path

def unpack_archives(source_dir):
    """
    Unpacks all ZIP and RAR files from the source directory
    and extracts them to the same directory.
    
    Args:
        source_dir (str): Path to the directory containing archives
    """
    source_path = Path(source_dir)
    
    # Check if directory exists
    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' does not exist!")
        return
    
    # Find all ZIP and RAR files
    archives = list(source_path.glob("*.zip")) + list(source_path.glob("*.rar"))
    
    if not archives:
        print(f"No ZIP or RAR files found in '{source_dir}'")
        return
    
    print(f"Found {len(archives)} archive(s) to extract")
    print("-" * 50)
    
    extracted_count = 0
    failed_count = 0
    
    for archive in archives:
        try:
            print(f"Extracting: {archive.name}")
            
            if archive.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(source_path)
            elif archive.suffix.lower() == '.rar':
                with rarfile.RarFile(archive, 'r') as rar_ref:
                    rar_ref.extractall(source_path)
            
            extracted_count += 1
            print(f"✓ Successfully extracted: {archive.name}")
            
        except Exception as e:
            failed_count += 1
            print(f"✗ Failed to extract {archive.name}: {str(e)}")
    
    print("-" * 50)
    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    # Directory containing your archives
    source_directory = r"E:\TradingData\forex"
    
    print("WinRAR/ZIP Archive Unpacker")
    print("=" * 50)
    print(f"Source directory: {source_directory}\n")
    
    unpack_archives(source_directory)
    
    input("\nPress Enter to exit...")