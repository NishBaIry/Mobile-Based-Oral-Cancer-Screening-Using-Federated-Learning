#!/usr/bin/env python3
"""
Complete FL Client Script - Train and Upload
Automates the entire FL client workflow for easy demonstration
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import glob
import argparse
from dotenv import load_dotenv

def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_training(dataset_path, epochs, batch_size, learning_rate, client_id, client_name):
    """Run the fine-tuning script"""
    print_banner("STEP 1: Fine-tuning MobileNetV2 Model")
    
    print("🔄 Starting model training...")
    print(f"   Client: {client_name} ({client_id})")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print("   This will fine-tune the model on local hospital data\n")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    training_script = script_dir / "mobilenetv2_finetune_federated.py"
    
    if not training_script.exists():
        print(f"❌ Training script not found: {training_script}")
        return False
    
    # Set environment variables for the training script
    env = os.environ.copy()
    env['CLIENT_ID'] = client_id
    env['CLIENT_NAME'] = client_name
    
    # Run the training script with arguments
    result = subprocess.run(
        [
            sys.executable, 
            str(training_script),
            "--dataset-path", dataset_path,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate)
        ],
        cwd=str(script_dir),
        env=env
    )
    
    if result.returncode != 0:
        print("\n❌ Training failed!")
        return False
    
    print("\n✅ Training completed successfully!")
    return True

def find_latest_weights():
    """Find the most recently saved weights file"""
    script_dir = Path(__file__).parent.absolute()
    weights_dir = script_dir / "client_weights"
    
    if not weights_dir.exists():
        print("❌ No weights directory found!")
        return None
    
    # Find all weight files
    weight_files = list(weights_dir.glob("*_weights.h5"))
    
    if not weight_files:
        print("❌ No weight files found!")
        return None
    
    # Get the most recent one
    latest_file = max(weight_files, key=lambda p: p.stat().st_mtime)
    return latest_file

def upload_to_server(client_id, client_name):
    """Upload the trained weights to FL server"""
    print_banner("STEP 2: Uploading Weights to FL Server")
    
    print("📤 Finding latest trained weights...")
    
    # Find the latest weights file
    weights_file = find_latest_weights()
    
    if weights_file:
        print(f"   Found: {weights_file.name}")
    else:
        print("\n❌ No weights file found to upload!")
        return False
    
    # Extract model name from weights file (remove _weights.h5 suffix)
    model_name = weights_file.stem.replace('_weights', '')
    
    print(f"\n🚀 Uploading to FL server...")
    print(f"   Model: {model_name}")
    print(f"   Client: {client_name} ({client_id})")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    upload_script = script_dir / "fl_client_send.py"
    
    # Set environment variables for upload script
    env = os.environ.copy()
    env['CLIENT_ID'] = client_id
    env['CLIENT_NAME'] = client_name
    
    # Run the upload script with the model name
    result = subprocess.run(
        [sys.executable, str(upload_script), '--model', model_name],
        cwd=str(script_dir),
        env=env
    )
    
    if result.returncode != 0:
        print("\n❌ Upload failed!")
        return False
    
    print("\n✅ Weights uploaded successfully!")
    return True

def main():
    """Main workflow"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FL Client - Train and Upload')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--client-id', type=str, default='client1', help='Unique client identifier')
    parser.add_argument('--client-name', type=str, default='Client 1', help='Human-readable client name')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  🏥 FEDERATED LEARNING CLIENT - AUTOMATED WORKFLOW")
    print("="*70)
    print(f"\n  Client: {args.client_name} ({args.client_id})")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
    print("\n  This script will:")
    print("    1. Fine-tune MobileNetV2 on local hospital data")
    print("    2. Upload trained weights to FL server")
    print("    3. Wait for global model aggregation")
    print("\n" + "="*70)
    
    # Step 1: Train
    if not run_training(
        args.dataset_path, 
        args.epochs, 
        args.batch_size, 
        args.learning_rate,
        args.client_id,
        args.client_name
    ):
        print("\n❌ Workflow failed at training step!")
        sys.exit(1)
    
    # Wait a moment
    time.sleep(1)
    
    # Step 2: Upload
    if not upload_to_server(args.client_id, args.client_name):
        print("\n❌ Workflow failed at upload step!")
        sys.exit(1)
    
    # Success!
    print_banner("✅ FEDERATED LEARNING CLIENT WORKFLOW COMPLETE!")
    print("📊 Summary:")
    print(f"   ✓ Client: {args.client_name} ({args.client_id})")
    print("   ✓ Model fine-tuned on local data")
    print("   ✓ Weights uploaded to FL server")
    print("   ✓ Server will aggregate when threshold reached\n")
    print("🔍 Check server logs for aggregation status")
    print("📱 Mobile apps will auto-download the updated model\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
