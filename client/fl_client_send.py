import os
import sys
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION FROM .env and ENVIRONMENT
# ============================================
FL_SERVER_HOST = os.getenv('FL_SERVER_HOST', '192.168.1.100')
FL_SERVER_PORT = os.getenv('FL_SERVER_PORT', '5000')

# Client ID/Name: Use from environment (set by GUI/CLI) or fallback to .env
CLIENT_ID = os.getenv('CLIENT_ID', 'client1')
CLIENT_NAME = os.getenv('CLIENT_NAME', 'Client 1')
SEND_FULL_WEIGHTS = os.getenv('SEND_FULL_WEIGHTS', 'False').lower() == 'true'

# API endpoints
FL_UPLOAD_ENDPOINT = os.getenv('FL_UPLOAD_ENDPOINT', '/api/upload_weights')
FL_STATUS_ENDPOINT = os.getenv('FL_STATUS_ENDPOINT', '/api/status')

# Construct full URLs
FL_SERVER_URL = f"http://{FL_SERVER_HOST}:{FL_SERVER_PORT}"
UPLOAD_URL = f"{FL_SERVER_URL}{FL_UPLOAD_ENDPOINT}"
STATUS_URL = f"{FL_SERVER_URL}{FL_STATUS_ENDPOINT}"

print("🌐 FEDERATED LEARNING CLIENT - UPLOAD WEIGHTS")
print("="*70)
print(f"📡 Server: {FL_SERVER_URL}")
print(f"🏥 Client: {CLIENT_NAME} ({CLIENT_ID})")
print("="*70)

# ============================================
# PARSE ARGUMENTS
# ============================================
parser = argparse.ArgumentParser(description='Upload model weights to FL server')
parser.add_argument('--model', type=str, required=False, 
                    help='Model name (e.g., mobilenetv2_finetuned_20251220_192542). If not provided, uses latest.')
parser.add_argument('--send-full', action='store_true', 
                    help='Send full weights instead of deltas')
parser.add_argument('--dry-run', action='store_true',
                    help='Test without actually sending')

args = parser.parse_args()

# Auto-detect latest model if not provided
if args.model:
    MODEL_NAME = args.model
else:
    print("\n🔍 Auto-detecting latest model...")
    weights_dir = Path('client_weights')
    if not weights_dir.exists():
        print("❌ Error: client_weights/ directory not found!")
        sys.exit(1)
    
    # Find latest weights file
    weights_files = sorted(weights_dir.glob('*_weights.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not weights_files:
        print("❌ Error: No weights files found in client_weights/")
        sys.exit(1)
    
    # Extract model name from filename (remove _weights.h5)
    MODEL_NAME = weights_files[0].stem.replace('_weights', '')
    print(f"✅ Found latest model: {MODEL_NAME}")
SEND_FULL = args.send_full or SEND_FULL_WEIGHTS

# ============================================
# LOCATE FILES
# ============================================
print(f"\n📂 Looking for model files...")

# Check in client_weights directory first, then current directory
weights_dir = Path('client_weights')

if SEND_FULL:
    weights_file = f"{MODEL_NAME}_weights.h5"
    upload_type = "full_weights"
    print(f"   Sending: FULL WEIGHTS")
else:
    weights_file = f"{MODEL_NAME}_deltas.npz"
    upload_type = "weight_deltas"
    print(f"   Sending: WEIGHT DELTAS (recommended)")

# Look in client_weights first, then current directory
weights_path = weights_dir / weights_file
if not weights_path.exists():
    weights_path = Path(weights_file)

if not weights_path.exists():
    print(f"❌ Error: {weights_file} not found!")
    print(f"\nSearched in:")
    print(f"   - client_weights/{weights_file}")
    print(f"   - ./{weights_file}")
    print(f"\nAvailable files in client_weights/:")
    if weights_dir.exists():
        for f in weights_dir.iterdir():
            if MODEL_NAME in f.name and (f.suffix == '.h5' or f.suffix == '.npz'):
                print(f"   - {f.name}")
    print(f"\nAvailable files in current directory:")
    for f in Path('.').iterdir():
        if MODEL_NAME in f.name and (f.suffix == '.h5' or f.suffix == '.npz'):
            print(f"   - {f.name}")
    sys.exit(1)

report_file = weights_dir / f"{MODEL_NAME}_report.txt"
if not report_file.exists():
    report_file = Path(f"{MODEL_NAME}_report.txt")
report_exists = report_file.exists()

print(f"✅ Found: {weights_path}")
if report_exists:
    print(f"✅ Found: {report_file}")

# Get file size
file_size_mb = weights_path.stat().st_size / (1024 * 1024)
print(f"   Size: {file_size_mb:.2f} MB")

# ============================================
# CHECK SERVER STATUS
# ============================================
print(f"\n🔍 Checking server status...")

if args.dry_run:
    print(f"⚠️  DRY RUN MODE - Not connecting to server")
else:
    try:
        response = requests.get(STATUS_URL, timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server is online!")
            print(f"   Clients connected: {status.get('clients_connected', 'N/A')}")
            print(f"   Current round: {status.get('current_round', 'N/A')}")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {FL_SERVER_URL}")
        print(f"   Please check:")
        print(f"   1. Server is running")
        print(f"   2. IP address is correct in .env")
        print(f"   3. Port {FL_SERVER_PORT} is open")
        print(f"   4. Both devices are on same network")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"❌ Connection timeout")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

# ============================================
# PREPARE METADATA
# ============================================
# Get dataset info from training data if available
train_path = Path(os.getenv('LOCAL_DATASET_PATH', '../2nd_traning')) / 'train'
num_samples = 1000  # Default
if train_path.exists():
    try:
        # Count images in all subdirectories
        num_samples = sum(1 for _ in train_path.rglob('*.jpg')) + sum(1 for _ in train_path.rglob('*.png'))
    except:
        pass

file_size_mb = weights_path.stat().st_size / (1024 * 1024)

metadata = {
    'client_id': CLIENT_ID,
    'client_name': CLIENT_NAME,
    'model_name': MODEL_NAME,
    'upload_type': upload_type,
    'timestamp': datetime.now().isoformat(),
    'file_size_mb': round(file_size_mb, 2),
    'num_samples': num_samples
}

# Read training report if available
if report_exists:
    with open(report_file, 'r') as f:
        report_content = f.read()
        # Extract metrics from report
        if 'Accuracy:' in report_content:
            lines = report_content.split('\n')
            for line in lines:
                if 'Performance After Fine-tuning:' in line:
                    idx = lines.index(line)
                    metadata['metrics'] = '\n'.join(lines[idx:idx+5])
                    break

print(f"\n📋 Upload metadata:")
print(json.dumps(metadata, indent=2))

# ============================================
# UPLOAD TO SERVER
# ============================================
if args.dry_run:
    print(f"\n⚠️  DRY RUN - Skipping actual upload")
    print(f"   Would upload to: {UPLOAD_URL}")
    print(f"   File: {weights_path} ({file_size_mb:.2f} MB)")
else:
    print(f"\n🚀 Uploading to FL server...")
    
    try:
        with open(weights_path, 'rb') as f:
            files = {
                'weights': (weights_path.name, f, 'application/octet-stream')
            }
            data = {
                'metadata': json.dumps(metadata)
            }
            
            print(f"   Uploading {file_size_mb:.2f} MB...")
            response = requests.post(
                UPLOAD_URL, 
                files=files, 
                data=data,
                timeout=300  # 5 minutes timeout for large files
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Upload successful!")
                print(f"   Server response: {result.get('message', 'OK')}")
                print(f"   Upload ID: {result.get('upload_id', 'N/A')}")
                
                # Save upload receipt
                receipt_dir = Path('client_weights')
                receipt_dir.mkdir(exist_ok=True)
                receipt_file = receipt_dir / f"{MODEL_NAME}_upload_receipt.json"
                receipt = {
                    'upload_time': datetime.now().isoformat(),
                    'server_url': FL_SERVER_URL,
                    'upload_id': result.get('upload_id'),
                    'model_name': MODEL_NAME,
                    'client_id': CLIENT_ID,
                    'server_response': result
                }
                with open(receipt_file, 'w') as rf:
                    json.dump(receipt, rf, indent=2)
                print(f"   Receipt saved: {receipt_file}")
                
            else:
                print(f"\n❌ Upload failed!")
                print(f"   Status code: {response.status_code}")
                print(f"   Response: {response.text}")
                sys.exit(1)
                
    except requests.exceptions.Timeout:
        print(f"\n❌ Upload timeout (file too large or slow connection)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Upload error: {e}")
        sys.exit(1)

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("🎯 NEXT STEPS")
print("="*70)
print(f"1. Wait for server to aggregate weights from all clients")
print(f"2. Download updated global model from server")
print(f"3. Use updated model for next training round")
print(f"\n💡 To download global model:")
print(f"   python fl_client_download.py")
print("="*70)
