"""
Filter MeasureBench images to select only gauge and dial images.

This script filters the extracted MeasureBench images based on:
- image_type: contains 'gauge' or related terms
- design: contains 'dial' or related terms

Output: Filtered images organized by split (real_world/synthetic_test)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def load_mapping(mapping_path: str) -> dict:
    """
    Load the image mapping from JSON file.
    
    Args:
        mapping_path: Path to the mapping JSON file
        
    Returns:
        Dictionary mapping question_id to image info
    """
    print(f"Loading mapping from: {mapping_path}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"  Loaded {len(mapping)} images")
    return mapping


def is_gauge_or_dial(image_info: dict) -> bool:
    """
    Check if an image filename contains gauge or dial keywords.
    
    Args:
        image_info: Dictionary containing image metadata
        
    Returns:
        True if image filename contains gauge/dial keywords
    """
    # Get the image filename from the path
    image_path = image_info.get('image_path', '')
    filename = Path(image_path).name.lower()
    
    # Keywords to look for in filename
    gauge_keywords = ['meter', 'gauge', 'pressure', 'fuel', 'oil', 'speedometer', 'tachometer', 
                      'voltmeter', 'ammeter', 'thermometer', 'hygrometer', 'barometer',
                      'manometer', 'flowmeter', 'odometer']
    
    dial_keywords = ['dial', 'pointer', 'analog', 'clock', 'meter']
    
    # Check if any keyword is in the filename
    for keyword in gauge_keywords + dial_keywords:
        if keyword in filename:
            return True
    
    return False


def filter_and_copy_images(mapping: dict, source_base: str, output_dir: str):
    """
    Filter images and copy them to output directory.
    
    Args:
        mapping: Dictionary mapping question_id to image info
        source_base: Base directory where images are stored
        output_dir: Output directory for filtered images
        
    Returns:
        Dictionary with filtering statistics
    """
    # Statistics
    stats = {
        'real_world': {'total': 0, 'selected': 0},
        'synthetic_test': {'total': 0, 'selected': 0}
    }
    
    # Create output directories
    real_world_dir = Path(output_dir) / 'real_world'
    synthetic_test_dir = Path(output_dir) / 'synthetic_test'
    
    real_world_dir.mkdir(parents=True, exist_ok=True)
    synthetic_test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Filtering images...")
    print(f"{'='*60}")
    
    selected_count = 0
    
    # Process each image
    for question_id, image_info in tqdm(mapping.items(), desc="Filtering images", unit="img"):
        split = image_info.get('split', 'unknown')
        
        # Update total count
        if split in stats:
            stats[split]['total'] += 1
        
        # Check if this is a gauge/dial image
        if is_gauge_or_dial(image_info):
            selected_count += 1
            
            # Get source path (make it absolute if relative)
            source_path_str = image_info['image_path']
            source_path = Path(source_path_str)
            
            # If source path is relative, make it absolute relative to current working directory
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            
            # Determine destination directory
            if split == 'real_world':
                dest_dir = real_world_dir
            elif split == 'synthetic_test':
                dest_dir = synthetic_test_dir
            else:
                continue
            
            # Generate destination filename
            dest_filename = f"{split}_{question_id}.jpg"
            dest_path = dest_dir / dest_filename
            
            # Copy the image
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                
                # Update selected count
                if split in stats:
                    stats[split]['selected'] += 1
            else:
                print(f"  Warning: Source image not found: {source_path}")
    
    print(f"\nTotal selected: {selected_count}")
    print(f"Stats: {stats}")
    
    return stats


def save_filtered_mapping(mapping: dict, output_path: str, filtered_ids: set):
    """
    Save a filtered mapping containing only gauge/dial images.
    
    Args:
        mapping: Original mapping dictionary
        output_path: Path to save filtered mapping
        filtered_ids: Set of question_ids that passed the filter
    """
    filtered_mapping = {
        qid: info for qid, info in mapping.items() 
        if qid in filtered_ids
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltered mapping saved to: {output_path}")
    print(f"  Contains {len(filtered_mapping)} images")


def main():
    """Main function to filter MeasureBench images."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter MeasureBench images to select gauge and dial types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter images with default paths
  python filter_mbench_gauge_dial.py
  
  # Specify custom paths
  python filter_mbench_gauge_dial.py \\
    --mapping data/mbench_image_mapping.json \\
    --source dataset/measurebench_images \\
    --output dataset/measurebench_gauge_dial
        """
    )
    
    parser.add_argument(
        '--mapping',
        type=str,
        default='data/mbench_image_mapping.json',
        help='Path to the image mapping JSON file'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='dataset/measurebench_images',
        help='Source directory containing extracted images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='dataset/measurebench_gauge_dial',
        help='Output directory for filtered images'
    )
    
    parser.add_argument(
        '--save-mapping',
        type=str,
        default=None,
        help='Path to save filtered mapping (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine paths
    base_dir = Path(__file__).parent.parent
    
    mapping_path = args.mapping if Path(args.mapping).is_absolute() else base_dir / args.mapping
    source_base = args.source if Path(args.source).is_absolute() else base_dir / args.source
    output_dir = args.output if Path(args.output).is_absolute() else base_dir / args.output
    
    if args.save_mapping:
        save_mapping_path = args.save_mapping if Path(args.save_mapping).is_absolute() else base_dir / args.save_mapping
    else:
        save_mapping_path = output_dir / 'filtered_mapping.json'
    
    print("="*60)
    print("MeasureBench Gauge & Dial Image Filter")
    print("="*60)
    print(f"\nMapping file: {mapping_path}")
    print(f"Source directory: {source_base}")
    print(f"Output directory: {output_dir}")
    print(f"Save mapping to: {save_mapping_path}")
    
    # Check if mapping file exists
    if not mapping_path.exists():
        print(f"\nERROR: Mapping file not found: {mapping_path}")
        print("Please run mbench_2image.py first to extract images and generate mapping")
        sys.exit(1)
    
    # Check if source directory exists
    if not source_base.exists():
        print(f"\nERROR: Source directory not found: {source_base}")
        print("Please run mbench_2image.py first to extract images")
        sys.exit(1)
    
    # Load mapping
    mapping = load_mapping(str(mapping_path))
    
    # Filter and copy images
    stats = filter_and_copy_images(mapping, str(source_base), str(output_dir))
    
    # Collect filtered question IDs
    filtered_ids = set()
    for question_id, image_info in mapping.items():
        if is_gauge_or_dial(image_info):
            filtered_ids.add(question_id)
    
    # Save filtered mapping
    save_filtered_mapping(mapping, str(save_mapping_path), filtered_ids)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"\nReal World:")
    print(f"  Total images: {stats['real_world']['total']}")
    print(f"  Filtered (gauge/dial): {stats['real_world']['selected']}")
    if stats['real_world']['total'] > 0:
        percentage = stats['real_world']['selected'] / stats['real_world']['total'] * 100
        print(f"  Percentage: {percentage:.1f}%")
    
    print(f"\nSynthetic Test:")
    print(f"  Total images: {stats['synthetic_test']['total']}")
    print(f"  Filtered (gauge/dial): {stats['synthetic_test']['selected']}")
    if stats['synthetic_test']['total'] > 0:
        percentage = stats['synthetic_test']['selected'] / stats['synthetic_test']['total'] * 100
        print(f"  Percentage: {percentage:.1f}%")
    
    total_selected = stats['real_world']['selected'] + stats['synthetic_test']['selected']
    total_images = stats['real_world']['total'] + stats['synthetic_test']['total']
    
    print(f"\nOverall:")
    print(f"  Total images: {total_images}")
    print(f"  Filtered (gauge/dial): {total_selected}")
    if total_images > 0:
        percentage = total_selected / total_images * 100
        print(f"  Percentage: {percentage:.1f}%")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
