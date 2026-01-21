import os
import zipfile
import rarfile
from pathlib import Path

def unpack_archives(source_dir, create_subfolders=False, delete_after_extract=False):
    """
    Unpacks all ZIP and RAR files from the source directory.
    
    Args:
        source_dir (str): Path to the directory containing archives
        create_subfolders (bool): If True, creates a subfolder for each archive
        delete_after_extract (bool): If True, deletes archive after successful extraction
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
            
            # Determine extraction path
            if create_subfolders:
                extract_path = source_path / archive.stem  # Use archive name without extension
                extract_path.mkdir(exist_ok=True)
            else:
                extract_path = source_path
            
            # Extract based on file type
            if archive.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif archive.suffix.lower() == '.rar':
                with rarfile.RarFile(archive, 'r') as rar_ref:
                    rar_ref.extractall(extract_path)
            
            extracted_count += 1
            print(f"✓ Successfully extracted to: {extract_path}")
            
            # Delete archive if requested
            if delete_after_extract:
                archive.unlink()
                print(f"  → Deleted archive: {archive.name}")
            
        except Exception as e:
            failed_count += 1
            print(f"✗ Failed to extract {archive.name}: {str(e)}")
    
    print("-" * 50)
    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Failed: {failed_count}")
    
    # Show extracted files info
    if extracted_count > 0:
        csv_files = list(source_path.rglob("*.csv"))
        parquet_files = list(source_path.rglob("*.parquet"))
        print(f"\nFound in directory:")
        print(f"  CSV files: {len(csv_files)}")
        print(f"  Parquet files: {len(parquet_files)}")

if __name__ == "__main__":
    # CONFIGURATION - Adjust these settings
    directories = [
        r"E:\TradingData\crypto",
        r"E:\TradingData\indices"
    ]
    
    # Set to True if you want each dataset in its own folder
    create_subfolders = False
    
    # Set to True to delete ZIP/RAR files after extraction (saves space)
    delete_after_extract = False
    
    print("Archive Unpacker for Trading Data")
    print("=" * 50)
    print(f"Create subfolders: {create_subfolders}")
    print(f"Delete after extract: {delete_after_extract}\n")
    
    # Process each directory
    for directory in directories:
        print(f"\n{'='*50}")
        print(f"Processing: {directory}")
        print('='*50)
        unpack_archives(directory, create_subfolders, delete_after_extract)
    
    print("\n" + "="*50)
    print("All directories processed!")
    input("\nPress Enter to exit...")