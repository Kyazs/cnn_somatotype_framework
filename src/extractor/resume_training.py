#!/usr/bin/env python3
"""
Resume Training Script

Use this script to easily resume training from an emergency checkpoint.
Run this script when your training gets interrupted.
"""

import os
import sys
import glob
from datetime import datetime

# Add parent directory to path for imports
sys.path.append("..")
from utils import MODEL_FILES_DIR

def find_checkpoints():
    """Find available emergency checkpoints."""
    emergency_dir = os.path.join(MODEL_FILES_DIR, "emergency_checkpoints")
    
    if not os.path.exists(emergency_dir):
        print("No emergency checkpoint directory found.")
        return []
    
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(emergency_dir, "*.keras"))
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return []
    
    # Get file info
    checkpoints = []
    for file_path in checkpoint_files:
        stat = os.stat(file_path)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / (1024 * 1024)
        checkpoints.append({
            'path': file_path,
            'filename': os.path.basename(file_path),
            'modified': mod_time,
            'size_mb': size_mb
        })
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    
    return checkpoints

def show_training_log():
    """Show recent training progress from log file."""
    emergency_dir = os.path.join(MODEL_FILES_DIR, "emergency_checkpoints")
    log_file = os.path.join(emergency_dir, "training_progress.txt")
    
    if os.path.exists(log_file):
        print("\n" + "="*60)
        print("RECENT TRAINING PROGRESS")
        print("="*60)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Show last 10 lines
            for line in lines[-10:]:
                print(line.strip())
        print("="*60 + "\n")
    else:
        print("No training log found.")

def main():
    print("="*60)
    print("TRAINING RESUMPTION HELPER")
    print("="*60)
    
    # Show training log
    show_training_log()
    
    # Find checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoints available for resumption.")
        print("You'll need to start training from scratch.")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    print()
    
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint['filename']}")
        print(f"   Modified: {checkpoint['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {checkpoint['size_mb']:.1f} MB")
        print(f"   Path: {checkpoint['path']}")
        print()
    
    print("To resume training:")
    print("1. Run the main training script (extractor_model_training.py)")
    print("2. When prompted, choose 'y' to resume from checkpoint")
    print("3. The most recent checkpoint will be loaded automatically")
    print()
    
    print("TIPS:")
    print("- Check system resources before resuming")
    print("- Monitor GPU memory usage during training")  
    print("- Training will continue from the last saved epoch")
    print("- All training history before the checkpoint will be lost")
    print()

if __name__ == "__main__":
    main()