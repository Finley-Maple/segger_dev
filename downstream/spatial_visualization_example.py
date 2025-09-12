#!/usr/bin/env python3
"""
Example demonstrating the new spatial region visualization functionality.

This script shows how to use the modified post_embedding_visualization.py
with spatial region filtering.
"""

import subprocess
import sys

def run_spatial_visualization():
    """Example of running spatial visualization."""
    
    # Example 1: Visualize a specific spatial region
    print("=" * 60)
    print("EXAMPLE: Spatial Region Visualization")
    print("=" * 60)
    print("Command to run spatial visualization for a specific region:")
    print()
    
    cmd = [
        "python", "post_embedding_visualization.py",
        "--dataset", "colon",
        "--model_type", "seq", 
        "--spatial_region", "2000", "3000", "2000", "2500",
        "--min_transcripts", "5"
    ]
    
    print(" ".join(cmd))
    print()
    print("This will:")
    print("  1. Load the colon dataset with seq model")
    print("  2. Find all batches containing transcripts in region x=[2000,3000], y=[2000,2500]")
    print("  3. Require at least 5 transcripts per batch in the region")
    print("  4. Combine batches from train/test/val datasets")
    print("  5. Generate embedding visualizations for this spatial region")
    print()
    
    # Example 2: Default behavior (CSV files)
    print("=" * 60)
    print("EXAMPLE: Default Behavior (CSV files)")
    print("=" * 60)
    print("Command to run with CSV batch indices (original behavior):")
    print()
    
    cmd2 = [
        "python", "post_embedding_visualization.py",
        "--dataset", "colon",
        "--model_type", "seq"
    ]
    
    print(" ".join(cmd2))
    print()
    print("This will:")
    print("  1. Load the colon dataset with seq model") 
    print("  2. Look for CSV files: train_region_batches.csv, test_region_batches.csv, val_region_batches.csv")
    print("  3. If CSV files found, use those batch indices")
    print("  4. If no CSV files, fall back to random batch selection")
    print("  5. Generate embedding visualizations")
    print()
    
    print("=" * 60)
    print("USAGE SUMMARY")
    print("=" * 60)
    print("New spatial region parameters:")
    print("  --spatial_region X_MIN X_MAX Y_MIN Y_MAX")
    print("  --min_transcripts N (default: 1)")
    print()
    print("When --spatial_region is specified:")
    print("  → Automatically computes batch indices for the region")
    print("  → Combines batches from train/test/val datasets")
    print("  → No need for CSV files")
    print()
    print("When --spatial_region is NOT specified:")
    print("  → Uses original behavior (CSV files or random selection)")


if __name__ == "__main__":
    run_spatial_visualization()
