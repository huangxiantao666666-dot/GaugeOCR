"""
Clock Detection and Cropping Module using YOLOv26
================================================
This module uses YOLOv26 to detect clocks (gauges) in images and crop them out.

Features:
1. Initialize YOLOv26 model with custom weights
2. Detect all clocks in an image
3. Crop each detected clock and save as new images
4. Batch process all images in a folder
5. Create mapping dictionary (source file -> cropped files)

Usage:
    from utils.detect import ClockDetector
    
    # Initialize detector
    detector = ClockDetector(weights_path="checkpoints/yolo26m.pt")
    
    # Process a single image
    mapping = detector.process_image("input.jpg", output_dir="output")
    
    # Process all images in a folder
    mapping = detector.process_folder("input_folder", output_dir="output_folder")
"""

import matplotlib.pyplot as plt
import os
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from tqdm import tqdm


class ClockDetector:
    """
    Clock Detector using YOLOv26.
    
    This class provides functionality to:
    1. Load YOLOv26 model with custom weights
    2. Detect clocks (gauges) in images
    3. Crop detected clocks and save them
    4. Batch process folders of images
    """
    
    def __init__(self, weights_path: str = "checkpoints/yolo26m.pt", 
                 confidence_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize the YOLOv26 clock detector.
        
        Args:
            weights_path: Path to YOLOv26 weights file (.pt)
            confidence_threshold: Confidence threshold for detections (0.0-1.0)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        
        # Auto-select device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load YOLOv26 model
        print(f"Loading YOLOv26 model from: {weights_path}")
        self.model = YOLO(weights_path)
        print("Model loaded successfully!")
        
        # Move model to specified device
        self.model.to(self.device)
        
    def detect_clocks(self, image_path: str) -> List[Dict]:
        """
        Detect all clocks in an image.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            List of dictionaries, each containing:
                - box: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_id: Class ID of the detection
        """
        try:
            # Run inference with verbose=False to suppress YOLO output
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            
            # Get the first result (single image)
            result = results[0]
            
            # Extract detections
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confs = result.boxes.conf.cpu().numpy()   # Confidence scores
                classes = result.boxes.cls.cpu().numpy()  # Class IDs
                
                # Get class names from model
                class_names = self.model.names
                
                for i in range(len(boxes)):
                    class_id = int(classes[i])
                    class_name = class_names[class_id]
                    
                    # Only keep detections that are 'clock' or similar categories
                    # Common clock/gauge class names: 'clock', 'gauge', 'meter', '仪表', '钟表'
                    clock_keywords = ['clock', 'gauge', 'meter', '仪表', '钟表', '压力表', '电压表', '电流表']
                    is_clock = any(keyword in class_name.lower() for keyword in clock_keywords)
                    
                    if is_clock:
                        detections.append({
                            'box': boxes[i].tolist(),
                            'confidence': float(confs[i]),
                            'class_id': class_id,
                            'class_name': class_name
                        })
            
            return detections
            
        except Exception as e:
            print(f"  ERROR processing {image_path}: {str(e)}")
            return []
    
    def crop_and_save(self, image_path: str, detections: List[Dict], 
                     output_dir: str, base_name: str, show_progress: bool = False) -> List[str]:
        """
        Crop detected clocks and save them as new images.
        
        Args:
            image_path: Path to the original image
            detections: List of detection dictionaries
            output_dir: Directory to save cropped images
            base_name: Base name for the cropped images
            show_progress: Whether to print progress information
        
        Returns:
            List of paths to saved cropped images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        saved_paths = []
        
        for i, det in enumerate(detections):
            # Extract bounding box
            x1, y1, x2, y2 = map(int, det['box'])
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Calculate crop dimensions
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            # Check if crop dimensions are smaller than 0.5x original dimensions
            should_crop = (crop_width < w * 0.5) or (crop_height < h * 0.5)
            
            if should_crop:
                # Crop the region
                cropped = image[y1:y2, x1:x2]
                
                # Generate output filename
                output_filename = f"{base_name}_clock_{i+1}.jpg"
                output_path = os.path.join(output_dir, output_filename)

                # Save the cropped image
                cv2.imwrite(output_path, cropped)
                saved_paths.append(output_path)
                
                if show_progress:
                    print(f"  Saved (cropped): {output_filename} (box: [{x1}, {y1}, {x2}, {y2}], "
                          f"size: {crop_width}x{crop_height}, conf: {det['confidence']:.3f})")
            else:
                # Copy the original image
                output_filename = f"{base_name}_full_{i+1}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                cv2.imwrite(output_path, image)
                saved_paths.append(output_path)
                
                if show_progress:
                    print(f"  Saved (full): {output_filename} (box: [{x1}, {y1}, {x2}, {y2}], "
                          f"size: {crop_width}x{crop_height} >= 0.5x original, conf: {det['confidence']:.3f})")
        
        return saved_paths
    
    def process_image(self, image_path: str, output_dir: str, show_progress: bool = False) -> Dict[str, List[str]]:
        """
        Process a single image: detect clocks and crop them.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save cropped images
            show_progress: Whether to print progress information
        
        Returns:
            Dictionary mapping: {source_image: [cropped_image_paths]}
        """
        if show_progress:
            print(f"\nProcessing image: {image_path}")
        
        # Get base name for output files
        base_name = Path(image_path).stem
        
        # Detect clocks
        detections = self.detect_clocks(image_path)
        
        if len(detections) == 0:
            if show_progress:
                print(f"  No clocks detected in {image_path}")
            return {image_path: []}
        
        # Crop and save
        saved_paths = self.crop_and_save(image_path, detections, output_dir, base_name, show_progress=False)
        
        if show_progress:
            print(f"Detected {len(detections)} clock(s) in {image_path}")
            for saved_path in saved_paths:
                print(f"  Saved: {os.path.basename(saved_path)}")
        
        return {image_path: saved_paths}
    
    def process_folder(self, input_folder: str, output_folder: str, 
                      extensions: List[str] = None) -> Dict[str, List[str]]:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to input folder containing images
            output_folder: Path to output folder for cropped images
            extensions: List of image extensions to process (default: ['.jpg', '.jpeg', '.png', '.bmp'])
        
        Returns:
            Dictionary mapping: {source_image: [cropped_image_paths]}
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp',
                         '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP']
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
        
        if len(image_files) == 0:
            print(f"No images found in {input_folder}")
            return {}
        
        print(f"Found {len(image_files)} image(s) in {input_folder}")
        
        # Process each image with progress bar
        all_mappings = {}
        failed_images = []
        
        with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
            for img_path in image_files:
                img_path_str = str(img_path)
                try:
                    # Process image without individual progress output
                    mappings = self.process_image(img_path_str, output_folder, show_progress=False)
                    all_mappings.update(mappings)
                    
                    # Update progress bar with detection info
                    num_detections = len(mappings[list(mappings.keys())[0]])
                    if num_detections > 0:
                        pbar.set_postfix_str(f"Detected: {num_detections}")
                    else:
                        pbar.set_postfix_str("No detection")
                    
                except Exception as e:
                    print(f"\nERROR processing {img_path_str}: {str(e)}")
                    failed_images.append(img_path_str)
                
                pbar.update(1)
        
        # Print summary
        total_cropped = sum(len(v) for v in all_mappings.values())
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"  Images processed: {len(all_mappings)}")
        print(f"  Total clocks cropped: {total_cropped}")
        if len(failed_images) > 0:
            print(f"  Failed images: {len(failed_images)}")
        print(f"  Output directory: {output_folder}")
        print(f"{'='*60}")
        
        return all_mappings
    
    def save_mapping(self, mapping: Dict[str, List[str]], output_path: str):
        """
        Save the mapping dictionary to a text file.
        
        Args:
            mapping: Dictionary mapping source images to cropped images
            output_path: Path to save the mapping file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Clock Detection Mapping\n")
            f.write("="*60 + "\n\n")
            
            for source, cropped_list in mapping.items():
                f.write(f"Source: {source}\n")
                if len(cropped_list) == 0:
                    f.write("  No clocks detected\n")
                else:
                    f.write(f"  Detected {len(cropped_list)} clock(s):\n")
                    for cropped in cropped_list:
                        f.write(f"    - {cropped}\n")
                f.write("\n")
        
        print(f"Mapping saved to: {output_path}")


def main():
    """
    Example usage of the ClockDetector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect and crop clocks from images')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or folder path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for cropped images')
    parser.add_argument('--weights', type=str, default='checkpoints/yolo26m.pt',
                       help='Path to YOLOv26 weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--save-mapping', action='store_true',
                       help='Save mapping dictionary to file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ClockDetector(weights_path=args.weights, confidence_threshold=args.conf)
    
    # Check if input is a file or folder
    if os.path.isfile(args.input):
        mapping = detector.process_image(args.input, args.output)
    elif os.path.isdir(args.input):
        mapping = detector.process_folder(args.input, args.output)
    else:
        print(f"Invalid input path: {args.input}")
        return
    
    # Save mapping if requested
    if args.save_mapping:
        mapping_path = os.path.join(args.output, 'mapping.txt')
        detector.save_mapping(mapping, mapping_path)
    
    # Print mapping
    print("\nMapping Summary:")
    for source, cropped in mapping.items():
        print(f"  {Path(source).name} -> {[Path(c).name for c in cropped]}")


if __name__ == "__main__":
    main()
