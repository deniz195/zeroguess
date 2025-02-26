#!/usr/bin/env python
"""
Utility script to clean up old visualization files from the visualization output directory.

Usage:
    python cleanup_visualizations.py [--days N] [--all]

Arguments:
    --days N : Remove files older than N days (default: 7)
    --all    : Remove all visualization files
"""

import os
import sys
import glob
import time
import argparse
import datetime
import shutil
from pathlib import Path


# Path to the visualization output directory
VISUALIZATION_OUTPUT_DIR = os.path.join("tests", "output", "visualizations")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up old visualization files from tests"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--days", 
        type=int, 
        default=7, 
        help="Remove files older than N days (default: 7)"
    )
    group.add_argument(
        "--all", 
        action="store_true", 
        help="Remove all visualization files"
    )
    
    return parser.parse_args()


def cleanup_visualizations(days=None, remove_all=False):
    """Clean up visualization files.
    
    Args:
        days (int, optional): Remove files older than this many days.
        remove_all (bool): If True, remove all files regardless of age.
    """
    # Ensure the directory exists
    visualization_dir = Path(VISUALIZATION_OUTPUT_DIR)
    if not visualization_dir.exists():
        print(f"Visualization directory does not exist: {visualization_dir}")
        return
    
    # Get all PNG files in the directory
    files = list(visualization_dir.glob("*.png"))
    if not files:
        print("No visualization files found.")
        return
    
    print(f"Found {len(files)} visualization files.")
    
    if remove_all:
        # Remove all files
        for file_path in files:
            file_path.unlink()
        print(f"Removed all {len(files)} visualization files.")
        return
    
    if days is not None:
        # Calculate the cutoff time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Count files to remove
        old_files = [f for f in files if f.stat().st_mtime < cutoff_time]
        
        if not old_files:
            print(f"No visualization files older than {days} days found.")
            return
        
        # Remove old files
        for file_path in old_files:
            file_path.unlink()
        
        print(f"Removed {len(old_files)} visualization files older than {days} days.")


if __name__ == "__main__":
    args = parse_args()
    
    if args.all:
        cleanup_visualizations(remove_all=True)
    else:
        cleanup_visualizations(days=args.days) 