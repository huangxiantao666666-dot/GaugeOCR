"""
Script to extract images from MeasureBench dataset.

This script processes the parquet files in dataset/measurebench/data/
and extracts all images to a folder structure for further processing.

MeasureBench contains:
- real_world: 1272 real-world instrument images
- synthetic_test: 1170 synthetic instrument images with randomized readings
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io


def load_parquet_files(parquet_dir: str):
    """
    Load all parquet files from the specified directory.
    
    Args:
        parquet_dir: Directory containing parquet files
        
    Returns:
        Dictionary with split names as keys and DataFrames as values
    """
    parquet_files = list(Path(parquet_dir).glob('*.parquet'))
    
    if len(parquet_files) == 0:
        print(f"No parquet files found in {parquet_dir}")
        return {}
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    splits = {}
    for parquet_file in parquet_files:
        print(f"\nLoading: {parquet_file.name}")
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Determine split name from filename
        split_name = parquet_file.stem.split('-')[0]  # e.g., 'real_world' or 'synthetic_test'
        
        splits[split_name] = df
        
        print(f"  Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
    
    return splits


def extract_images_from_split(df: pd.DataFrame, output_dir: str, split_name: str):
    """
    Extract images from a DataFrame and save them to disk.
    
    Args:
        df: DataFrame containing image data
        output_dir: Base output directory
        split_name: Name of the split (e.g., 'real_world', 'synthetic_test')
        
    Returns:
        Dictionary mapping question_id to image path
    """
    # Create output directory for this split
    split_output_dir = Path(output_dir) / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Extracting images from {split_name} split")
    print(f"{'='*60}")
    
    mapping = {}
    saved_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images", unit="img"):
        try:
            question_id = row.get('question_id', f'{split_name}_{idx:05d}')
            image_type = row.get('image_type', 'unknown')
            
            # Get image data
            image_data = row.get('image')
            
            if image_data is None:
                print(f"  Warning: No image data for {question_id}")
                error_count += 1
                continue
            
            # Handle different image formats
            if isinstance(image_data, dict):
                # Image is stored as a dictionary with 'bytes' or 'path'
                if 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                elif 'path' in image_data:
                    # Load from path
                    image_bytes = open(image_data['path'], 'rb').read()
                else:
                    print(f"  Warning: Unknown image format for {question_id}")
                    error_count += 1
                    continue
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            elif isinstance(image_data, Image.Image):
                # Already a PIL Image
                pil_image = image_data
            else:
                print(f"  Warning: Unexpected image type: {type(image_data)}")
                error_count += 1
                continue
            
            # Convert bytes to PIL Image if needed
            if not isinstance(image_data, Image.Image):
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    print(f"  Error decoding image {question_id}: {str(e)}")
                    error_count += 1
                    continue
            
            # Determine image extension based on image_type or default to jpg
            if image_type and image_type.lower() in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
                ext = image_type.lower()
                if ext == 'jpeg':
                    ext = 'jpg'
            else:
                ext = 'jpg'
            
            # Generate output filename
            # Format: {split}_{question_id}_{image_type}.{ext}
            safe_question_id = str(question_id).replace('/', '_').replace('\\', '_')
            output_filename = f"{split_name}_{safe_question_id}.{ext}"
            output_path = split_output_dir / output_filename
            
            # Save image
            # Convert RGBA to RGB if saving as JPEG
            if ext == 'jpg' and pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
            pil_image.save(output_path)
            
            # Store mapping
            mapping[question_id] = {
                'image_path': str(output_path),
                'image_type': image_type,
                'split': split_name,
                'question': row.get('question', ''),
                'design': row.get('design', '')
            }
            
            saved_count += 1
            
        except Exception as e:
            print(f"  Error processing row {idx}: {str(e)}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Extraction complete for {split_name}")
    print(f"  Total images: {len(df)}")
    print(f"  Successfully saved: {saved_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {split_output_dir}")
    print(f"{'='*60}")
    
    return mapping


def save_mapping_to_file(mapping: dict, output_path: str):
    """
    Save the mapping dictionary to a JSON file.
    
    Args:
        mapping: Dictionary mapping question_id to image info
        output_path: Path to save the JSON file
    """
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nMapping saved to: {output_path}")


def main():
    """Main function to process MeasureBench dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract images from MeasureBench dataset parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract images from default location
  python mbench_2image.py
  
  # Specify custom input and output directories
  python mbench_2image.py -i /path/to/parquet -o /path/to/output
  
  # Specify custom mapping file location
  python mbench_2image.py --mapping /path/to/mapping.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input directory containing parquet files (default: dataset/measurebench/data)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for extracted images (default: dataset/measurebench_images)'
    )
    
    parser.add_argument(
        '--mapping',
        type=str,
        default=None,
        help='Path to save the mapping JSON file (default: data/mbench_image_mapping.json)'
    )
    
    args = parser.parse_args()
    
    # Determine paths
    base_dir = Path(__file__).parent.parent
    
    if args.input:
        parquet_dir = Path(args.input)
    else:
        parquet_dir = base_dir / 'dataset' / 'measurebench' / 'data'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_dir / 'dataset' / 'measurebench_images'
    
    if args.mapping:
        mapping_output = Path(args.mapping)
    else:
        mapping_output = base_dir / 'data' / 'mbench_image_mapping.json'
    
    print("="*60)
    print("MeasureBench Image Extraction Tool")
    print("="*60)
    print(f"\nInput directory: {parquet_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mapping file: {mapping_output}")
    
    # Check if parquet files exist
    if not parquet_dir.exists():
        print(f"\nERROR: Input directory not found: {parquet_dir}")
        print("Please ensure the MeasureBench dataset directory exists")
        sys.exit(1)
    
    # Check if there are parquet files in the directory, if not, try 'data' subdirectory
    parquet_files = list(parquet_dir.glob('*.parquet'))
    if len(parquet_files) == 0:
        data_subdir = parquet_dir / 'data'
        if data_subdir.exists():
            print(f"\nNote: No parquet files in root, looking in subdirectory: {data_subdir}")
            parquet_dir = data_subdir
        else:
            print(f"\nERROR: No parquet files found in {parquet_dir}")
            print("Please ensure the MeasureBench dataset parquet files are in the input directory or its 'data' subdirectory")
            sys.exit(1)
    
    # Load parquet files
    splits = load_parquet_files(str(parquet_dir))
    
    if len(splits) == 0:
        print("\nNo data loaded. Exiting.")
        sys.exit(1)
    
    # Process each split
    all_mappings = {}
    for split_name, df in splits.items():
        split_mapping = extract_images_from_split(df, str(output_dir), split_name)
        all_mappings.update(split_mapping)
    
    # Ensure mapping directory exists
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save combined mapping
    save_mapping_to_file(all_mappings, str(mapping_output))
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total splits processed: {len(splits)}")
    print(f"Total images extracted: {len(all_mappings)}")
    print(f"Output directory: {output_dir}")
    print(f"Mapping file: {mapping_output}")
    print("="*60)


if __name__ == '__main__':
    main()
