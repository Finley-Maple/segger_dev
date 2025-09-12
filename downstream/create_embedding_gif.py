#!/usr/bin/env python3
"""
Script to generate a GIF animation from tx_embeddings_by_gene_type images across training epochs.

This script searches for epoch-based directories containing tx_embeddings_by_gene_type.png files
and creates an animated GIF showing the evolution of transcript embeddings during training.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import re
import glob

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
    sys.exit(1)

MODEL_DIR = Path('/dkfz/cluster/gpu/data/OE0606/fengyun/segger_model/segger_colon_seq/segger_with_embeddings/version_5')
MODEL_DIR2 = Path('/dkfz/cluster/gpu/data/OE0606/fengyun/segger_model_align/segger_colon_seq/segger_with_embeddings/version_3')

# Example usage
# python create_embedding_gif.py --base_dir /dkfz/cluster/gpu/data/OE0606/fengyun/segger_model/segger_colon_seq/segger_with_embeddings/version_5 --output tx_embeddings_evolution.gif --start_epoch 0 --end_epoch 90 --duration 2000 --resize 800x600 --image_name tx_embeddings_by_spatial_distance.png --search_all
# python create_embedding_gif.py --base_dir /dkfz/cluster/gpu/data/OE0606/fengyun/segger_model_align/segger_colon_seq/segger_with_embeddings/version_3 --output tx_embeddings_evolution.gif --start_epoch 0 --end_epoch 90 --duration 2000 --resize 800x600 --image_name tx_embeddings_by_spatial_distance.png --search_all

def find_epoch_images(base_directory: Path, 
                     image_name: str = "tx_embeddings_by_gene_type.png",
                     start_epoch: int = 0, 
                     end_epoch: int = 90) -> List[Tuple[int, Path]]:
    """
    Find all epoch-based images in the directory structure.
    
    Args:
        base_directory: Base directory to search for epoch directories
        image_name: Name of the image file to look for
        start_epoch: Starting epoch number
        end_epoch: Ending epoch number
        
    Returns:
        List of tuples (epoch_number, image_path) sorted by epoch
    """
    epoch_images = []
    
    # Search patterns for epoch directories
    search_patterns = [
        base_directory / "embedding_plots" / f"epoch_*" / image_name,
        base_directory / f"epoch_*" / image_name,
        base_directory / "*" / "embedding_plots" / f"epoch_*" / image_name,
        base_directory / "*" / "*" / "embedding_plots" / f"epoch_*" / image_name,
    ]
    
    for pattern in search_patterns:
        for image_path in glob.glob(str(pattern)):
            image_path = Path(image_path)
            
            # Extract epoch number from path
            epoch_match = re.search(r'epoch_(\d+)', str(image_path))
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                
                # Check if epoch is in desired range
                if start_epoch <= epoch_num <= end_epoch:
                    epoch_images.append((epoch_num, image_path))
    
    # Remove duplicates and sort by epoch
    epoch_images = list(set(epoch_images))
    epoch_images.sort(key=lambda x: x[0])
    
    return epoch_images


def create_gif_from_images(image_paths: List[Path], 
                          output_path: Path,
                          duration: int = 500,
                          loop: int = 0,
                          resize_to: Optional[Tuple[int, int]] = None) -> None:
    """
    Create a GIF from a list of image paths.
    
    Args:
        image_paths: List of image file paths in order
        output_path: Output path for the GIF
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
        resize_to: Optional tuple (width, height) to resize images
    """
    if not image_paths:
        raise ValueError("No images provided for GIF creation")
    
    # Load and process images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if requested
            if resize_to:
                img = img.resize(resize_to, Image.Resampling.LANCZOS)
            
            images.append(img)
            
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    # Create GIF
    print(f"Creating GIF with {len(images)} frames...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"GIF saved to: {output_path}")


def main():
    """Main function to generate GIF from epoch-based embedding images."""
    parser = argparse.ArgumentParser(
        description="Generate GIF from tx_embeddings_by_gene_type images across training epochs"
    )
    
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="./logs",
        help="Base directory to search for epoch images (default: ./logs)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="tx_embeddings_evolution.gif",
        help="Output GIF filename (default: tx_embeddings_evolution.gif)"
    )
    
    parser.add_argument(
        "--start_epoch", 
        type=int, 
        default=0,
        help="Starting epoch number (default: 0)"
    )
    
    parser.add_argument(
        "--end_epoch", 
        type=int, 
        default=90,
        help="Ending epoch number (default: 90)"
    )
    
    parser.add_argument(
        "--duration", 
        type=int, 
        default=500,
        help="Duration of each frame in milliseconds (default: 500)"
    )
    
    parser.add_argument(
        "--resize", 
        type=str, 
        default=None,
        help="Resize images to WIDTHxHEIGHT (e.g., 800x600)"
    )
    
    parser.add_argument(
        "--image_name", 
        type=str, 
        default="tx_embeddings_by_gene_type.png",
        help="Name of the image file to look for (default: tx_embeddings_by_gene_type.png)"
    )
    
    parser.add_argument(
        "--search_all", 
        action="store_true",
        help="Search all subdirectories for epoch images"
    )
    
    args = parser.parse_args()
    
    # Parse resize option
    resize_to = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize_to = (width, height)
        except ValueError:
            print(f"Error: Invalid resize format '{args.resize}'. Use WIDTHxHEIGHT (e.g., 800x600)")
            sys.exit(1)
    
    # Convert base directory to Path
    base_dir = Path(args.base_dir).resolve()
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    print(f"Searching for epoch images in: {base_dir}")
    print(f"Looking for: {args.image_name}")
    print(f"Epoch range: {args.start_epoch} to {args.end_epoch}")
    
    # Find epoch images
    if args.search_all:
        # Search entire directory tree
        epoch_images = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == args.image_name:
                    file_path = Path(root) / file
                    # Extract epoch from path
                    epoch_match = re.search(r'epoch_(\d+)', str(file_path))
                    if epoch_match:
                        epoch_num = int(epoch_match.group(1))
                        if args.start_epoch <= epoch_num <= args.end_epoch:
                            epoch_images.append((epoch_num, file_path))
        
        epoch_images.sort(key=lambda x: x[0])
    else:
        epoch_images = find_epoch_images(base_dir, args.image_name, args.start_epoch, args.end_epoch)
    
    if not epoch_images:
        print(f"No epoch images found in {base_dir}")
        print("\nTip: Make sure you have run training with EmbeddingVisualizationCallback")
        print("     or try using --search_all to search the entire directory tree")
        
        # Show available directories for debugging
        print(f"\nAvailable subdirectories in {base_dir}:")
        try:
            for item in base_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
        except Exception:
            pass
        
        sys.exit(1)
    
    print(f"Found {len(epoch_images)} epoch images:")
    for epoch, path in epoch_images[:5]:  # Show first 5
        print(f"  Epoch {epoch}: {path}")
    if len(epoch_images) > 5:
        print(f"  ... and {len(epoch_images) - 5} more")
    
    # Extract just the image paths in order
    image_paths = [path for epoch, path in epoch_images]
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create GIF
    try:
        create_gif_from_images(
            image_paths=image_paths,
            output_path=output_path,
            duration=args.duration,
            loop=0,  # Infinite loop
            resize_to=resize_to
        )
        
        print(f"\nSuccess! Created GIF with {len(image_paths)} frames")
        print(f"Output: {output_path.resolve()}")
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
