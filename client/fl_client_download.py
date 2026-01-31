import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION FROM .env
# ============================================
FL_SERVER_HOST = os.getenv('FL_SERVER_HOST', '192.168.1.100')
FL_SERVER_PORT = os.getenv('FL_SERVER_PORT', '5000')
CLIENT_ID = os.getenv('CLIENT_ID', 'hospital_m1_mac')
BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH', 'mobilenetv2_oral_cancer_best.h5')

# API endpoints
FL_DOWNLOAD_ENDPOINT = os.getenv('FL_DOWNLOAD_ENDPOINT', '/api/download_global_model')

# Construct full URL
FL_SERVER_URL = f"http://{FL_SERVER_HOST}:{FL_SERVER_PORT}"
DOWNLOAD_URL = f"{FL_SERVER_URL}{FL_DOWNLOAD_ENDPOINT}"

print("🌐 FEDERATED LEARNING CLIENT - DOWNLOAD GLOBAL MODEL")
print("="*70)
print(f"📡 Server: {FL_SERVER_URL}")
print(f"🏥 Client ID: {CLIENT_ID}")
print("="*70)

# ============================================
# CHECK SERVER STATUS
# ============================================
print(f"\n🔍 Checking for new global model...")

try:
    response = requests.get(
        DOWNLOAD_URL,
        params={'client_id': CLIENT_ID},
        timeout=30
    )
    
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        
        # Check if server returned file directly or JSON
        if 'application/json' in content_type:
            result = response.json()
            if not result.get('model_available', True):
                print(f"⚠️  No global model available yet")
                print(f"   {result.get('message', '')}")
                sys.exit(0)
        else:
            # Server returned the model file directly
            model_round = response.headers.get('X-Model-Round', 'unknown')
            print(f"✅ New global model available!")
            print(f"   Round: {model_round}")
            
            # Create models directory if it doesn't exist
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            # Save the updated global model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = models_dir / f"global_model_round_{model_round}_{timestamp}.h5"
            
            print(f"\n📥 Downloading global model...")
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            print(f"✅ Downloaded: {output_file}")
            print(f"   Size: {file_size_mb:.2f} MB")
            
            # Update base model (replace it)
            base_model_file = Path(BASE_MODEL_PATH)
            base_model_file.parent.mkdir(exist_ok=True)
            
            if base_model_file.exists():
                backup_base = base_model_file.parent / f"{base_model_file.stem}_backup_{timestamp}.h5"
                base_model_file.rename(backup_base)
                print(f"   Backed up old model: {backup_base}")
            
            # Copy new global model as base model
            import shutil
            shutil.copy(output_file, base_model_file)
            print(f"   ✅ Updated base model: {BASE_MODEL_PATH}")
            print(f"   Next training will use this updated model")
            
            # Save download info
            info_file = output_file.with_suffix('.json')
            info = {
                'download_time': datetime.now().isoformat(),
                'server_url': FL_SERVER_URL,
                'round': model_round,
                'client_id': CLIENT_ID,
                'file_size_mb': file_size_mb
            }
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"\n✅ Ready for next training round!")
            print(f"   Run: python fl_client_train_and_upload.py")
            sys.exit(0)
            
    else:
        print(f"❌ Server returned status {response.status_code}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to server at {FL_SERVER_URL}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
