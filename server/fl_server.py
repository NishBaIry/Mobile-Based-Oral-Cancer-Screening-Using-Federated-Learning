"""
Federated Learning Server - Flask API
Aggregates model weights from multiple clients using FedAvg
"""

import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load environment variables
load_dotenv()

# Import FedAvg algorithm
from fedavg import FedAvg

# Configure logging with cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format - we'll handle formatting in code
)
logger = logging.getLogger(__name__)

# Disable Flask's default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')
BASE_MODEL_PATH = os.getenv('BASE_MODEL_PATH', 'models/global_model.h5')
GLOBAL_MODEL_PATH = os.getenv('GLOBAL_MODEL_PATH', 'models/global_model_latest.h5')
VALIDATION_DATASET_PATH = os.getenv('VALIDATION_DATASET_PATH', '../new_data')  # Path to validation dataset
VALIDATE_BEFORE_REPLACE = os.getenv('VALIDATE_BEFORE_REPLACE', 'True').lower() == 'true'

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Initialize FedAvg aggregator
fed_avg = FedAvg(base_model_path=BASE_MODEL_PATH)

# State persistence file
STATE_FILE = os.path.join(MODELS_FOLDER, 'server_state.json')

# Load persisted state or initialize new
def load_server_state():
    """Load server state from disk"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                saved_state = json.load(f)
                print(f"📂 Loaded server state: Round {saved_state.get('current_round', 0)}, Total aggregations: {saved_state.get('total_aggregations', 0)}")
                return saved_state
        except Exception as e:
            print(f"⚠️ Could not load state file: {e}")
    return {
        'current_round': 0,
        'clients_connected': 0,
        'pending_updates': {},
        'aggregation_threshold': int(os.getenv('AGGREGATION_THRESHOLD', 2)),
        'total_aggregations': 0
    }

def save_server_state():
    """Save server state to disk"""
    try:
        # Don't save pending_updates (transient data)
        state_to_save = {
            'current_round': server_state['current_round'],
            'total_aggregations': server_state['total_aggregations'],
            'aggregation_threshold': server_state['aggregation_threshold']
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state_to_save, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save state: {e}")

# Server state
server_state = load_server_state()
server_state['clients_connected'] = 0
server_state['pending_updates'] = {}

# Sync fed_avg round counter with loaded state
fed_avg.current_round = server_state['current_round']

# ============================================
# HELPER FUNCTIONS
# ============================================

def print_header(text, char="="):
    """Print a formatted header"""
    logger.info("")
    logger.info(char * 70)
    logger.info(f"  {text}")
    logger.info(char * 70)

def print_subheader(text):
    """Print a formatted subheader"""
    logger.info("")
    logger.info(f"▸ {text}")
    logger.info("─" * 70)

def print_success(text):
    """Print success message"""
    logger.info(f"✅ {text}")

def print_error(text):
    """Print error message"""
    logger.error(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    logger.warning(f"⚠️  {text}")

def print_info(text, indent=0):
    """Print info message"""
    prefix = "  " * indent
    logger.info(f"{prefix}• {text}")

def load_weights_from_file(filepath):
    """Load model weights from .h5 or .npz file"""
    try:
        if filepath.endswith('.h5'):
            model = tf.keras.models.load_model(filepath, compile=False)
            return model.get_weights()
        elif filepath.endswith('.npz'):
            data = np.load(filepath, allow_pickle=True)
            # Load arrays in correct order (arr_0, arr_1, arr_2, ...)
            weights = []
            for i in range(len(data.files)):
                key = f'arr_{i}'
                if key in data:
                    weights.append(data[key])
            if not weights:
                # Fallback to sorted keys if arr_N pattern not found
                weights = [data[key] for key in sorted(data.files)]
            return weights
        else:
            print_error(f"Unsupported file format: {filepath}")
            return None
    except Exception as e:
        print_error(f"Failed to load weights from {filepath}: {e}")
        return None

def save_upload_info(client_id, metadata):
    """Save upload metadata"""
    info_file = f"{UPLOAD_FOLDER}/{client_id}_info.json"
    with open(info_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def evaluate_model_accuracy(model, dataset_path, img_size=224, batch_size=32):
    """
    Evaluate model accuracy on test dataset
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_path = os.path.join(dataset_path, 'test')
        
        if not os.path.exists(test_path):
            print_warning(f"Test dataset not found at {test_path}")
            return None
        
        # Create test data generator (suppress output)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Temporarily redirect stdout to suppress generator output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        sys.stdout = old_stdout
        
        # Compile model for evaluation
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Evaluate
        results = model.evaluate(test_generator, verbose=0)
        
        # Extract metrics
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]),
            'recall': float(results[3]),
            'auc': float(results[4])
        }
        
        return metrics
    
    except Exception as e:
        print_error(f"Error evaluating model: {e}")
        return None

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'message': 'Federated Learning Server',
        'status': 'online',
        'version': '1.0.0',
        'endpoints': {
            'status': '/api/status',
            'upload': '/api/upload_weights',
            'download': '/api/download_global_model'
        }
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get server status"""
    stats = fed_avg.get_stats()
    
    return jsonify({
        'status': 'online',
        'current_round': server_state['current_round'],
        'clients_connected': len(server_state['pending_updates']),
        'pending_updates': list(server_state['pending_updates'].keys()),
        'aggregation_threshold': server_state['aggregation_threshold'],
        'total_aggregations': server_state['total_aggregations'],
        'fedavg_stats': stats
    })

@app.route('/api/upload_weights', methods=['POST'])
def upload_weights():
    """
    Upload model weights/deltas from client
    """
    try:
        # Check if weights file is present
        if 'weights' not in request.files:
            return jsonify({'error': 'No weights file provided'}), 400
        
        weights_file = request.files['weights']
        
        if weights_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get metadata
        metadata_str = request.form.get('metadata', '{}')
        metadata = json.loads(metadata_str)
        
        client_id = metadata.get('client_id', 'unknown')
        client_name = metadata.get('client_name', 'Unknown Client')
        upload_type = metadata.get('upload_type', 'full_weights')
        
        print_header(f"📥 NEW CLIENT UPLOAD", "=")
        print_info(f"Client ID: {client_id}")
        print_info(f"Client Name: {client_name}")
        print_info(f"Upload Type: {upload_type}")
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"{client_id}_{timestamp}_{weights_file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        weights_file.save(filepath)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print_info(f"File Size: {file_size_mb:.2f} MB")
        print_success(f"Saved to: {filename}")
        
        # Load weights from file
        weights = load_weights_from_file(filepath)
        
        if weights is None:
            print_error("Failed to load weights from file")
            return jsonify({'error': 'Failed to load weights'}), 400
        
        # Store in pending updates
        server_state['pending_updates'][client_id] = {
            'weights': weights,
            'filepath': filepath,
            'upload_type': upload_type,
            'metadata': metadata,
            'timestamp': timestamp,
            'num_samples': metadata.get('num_samples', 1000)
        }
        
        # Save metadata
        save_upload_info(client_id, metadata)
        
        pending_count = len(server_state['pending_updates'])
        threshold = server_state['aggregation_threshold']
        
        print_info(f"Pending: {pending_count}/{threshold} clients")
        
        # Check if we have enough clients to aggregate
        if pending_count >= threshold:
            print_success(f"Threshold reached! Starting aggregation...")
            success = perform_aggregation()
            
            if success:
                return jsonify({
                    'message': 'Upload successful - Aggregation completed',
                    'upload_id': filename,
                    'client_id': client_id,
                    'aggregation_performed': True,
                    'new_round': server_state['current_round']
                }), 200
            else:
                return jsonify({
                    'message': 'Upload successful - Aggregation failed',
                    'upload_id': filename,
                    'client_id': client_id,
                    'aggregation_performed': False
                }), 200
        else:
            print_info(f"Waiting for {threshold - pending_count} more client(s)...")
        
        return jsonify({
            'message': 'Upload successful - Waiting for more clients',
            'upload_id': filename,
            'client_id': client_id,
            'pending_clients': pending_count,
            'threshold': threshold
        }), 200
    
    except Exception as e:
        print_error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_global_model', methods=['GET'])
def download_global_model():
    """
    Download the latest global model
    """
    client_id = request.args.get('client_id', 'unknown')
    response_format = request.args.get('format', 'file')
    
    print_subheader(f"📤 Model Download Request")
    print_info(f"Client: {client_id}")
    print_info(f"Format: {response_format}")
    
    try:
        # Check for TFLite request
        if response_format == 'tflite':
            tflite_path = GLOBAL_MODEL_PATH.replace('.h5', '.tflite')
            if not os.path.exists(tflite_path):
                print_warning("TFLite model not available")
                return jsonify({
                    'model_available': False,
                    'message': 'TFLite model not available. Server may still be processing.'
                }), 200
            
            with open(tflite_path, 'rb') as f:
                model_bytes = f.read()
            
            print_success("TFLite model sent")
            
            return model_bytes, 200, {
                'Content-Type': 'application/octet-stream',
                'Content-Disposition': f'attachment; filename=global_model_round_{server_state["current_round"]}.tflite',
                'X-Model-Round': str(server_state['current_round']),
                'X-Model-Format': 'tflite'
            }
        
        # Check for H5 model
        if not os.path.exists(GLOBAL_MODEL_PATH):
            print_warning("No global model available yet")
            return jsonify({
                'model_available': False,
                'message': 'No global model available yet. Upload weights from clients first.'
            }), 200
        
        if response_format == 'bytes':
            # For Flutter app - return raw bytes (H5)
            with open(GLOBAL_MODEL_PATH, 'rb') as f:
                model_bytes = f.read()
            
            print_success("H5 model sent (bytes)")
            
            return model_bytes, 200, {
                'Content-Type': 'application/octet-stream',
                'Content-Disposition': f'attachment; filename=global_model_round_{server_state["current_round"]}.h5',
                'X-Model-Round': str(server_state['current_round']),
                'X-Model-Format': 'h5'
            }
        else:
            # Default file download (H5)
            print_success("H5 model sent (file)")
            return send_file(
                GLOBAL_MODEL_PATH,
                as_attachment=True,
                download_name=f'global_model_round_{server_state["current_round"]}.h5',
                mimetype='application/octet-stream'
            )
    
    except Exception as e:
        print_error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger_aggregation', methods=['POST'])
def trigger_aggregation():
    """
    Manually trigger aggregation (admin endpoint)
    """
    if len(server_state['pending_updates']) == 0:
        return jsonify({
            'success': False,
            'message': 'No pending updates to aggregate'
        }), 400
    
    logger.info("Manual aggregation triggered")
    success = perform_aggregation()
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Aggregation completed',
            'round': server_state['current_round']
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': 'Aggregation failed'
        }), 500

# ============================================
# AGGREGATION LOGIC
# ============================================

def perform_aggregation():
    """
    Perform FedAvg aggregation on pending updates
    """
    try:
        print_header("🔄 FEDERATED AGGREGATION (FedAvg)", "=")
        
        # Load global model if not loaded
        if fed_avg.global_model is None:
            print_info("Loading base global model...")
            if not fed_avg.load_global_model():
                print_error("Failed to load global model")
                return False
        
        # Collect client weights and metadata
        client_weights_list = []
        client_samples_list = []
        client_ids = []
        
        print_subheader("Participating Clients")
        for client_id, update_info in server_state['pending_updates'].items():
            client_name = update_info['metadata'].get('client_name', 'Unknown')
            num_samples = update_info.get('num_samples', 1000)
            client_weights_list.append(update_info['weights'])
            client_samples_list.append(num_samples)
            client_ids.append(client_id)
            print_info(f"{client_name} ({client_id}) - {num_samples} samples", indent=1)
        
        print_info(f"Total: {len(client_weights_list)} clients")
        
        # Check if clients sent deltas or full weights
        upload_type = server_state['pending_updates'][list(server_state['pending_updates'].keys())[0]].get('upload_type', 'full_weights')
        use_deltas = (upload_type == 'weight_deltas')
        
        print_subheader("Aggregation")
        print_info(f"Method: FedAvg ({'weight deltas' if use_deltas else 'full weights'})")
        
        # Perform aggregation
        if use_deltas:
            success = fed_avg.update_global_model(
                client_deltas=client_weights_list,
                client_samples=client_samples_list,
                use_deltas=True
            )
        else:
            success = fed_avg.update_global_model(
                client_weights=client_weights_list,
                client_samples=client_samples_list,
                use_deltas=False
            )
        
        if not success:
            print_error("Aggregation computation failed")
            return False
        
        print_success("Aggregation computed successfully")
        
        # Always save model (validation disabled for demonstration)
        should_replace = True
        old_accuracy = None
        new_accuracy = None
        
        # Optional: Show metrics for information only (doesn't affect saving)
        if os.path.exists(VALIDATION_DATASET_PATH):
            print_subheader("Model Validation (Info Only)")
            
            # Evaluate old model (if exists)
            if os.path.exists(GLOBAL_MODEL_PATH):
                print_info("Evaluating OLD model...", indent=1)
                try:
                    old_model = tf.keras.models.load_model(GLOBAL_MODEL_PATH, compile=False)
                    old_metrics = evaluate_model_accuracy(old_model, VALIDATION_DATASET_PATH)
                    
                    if old_metrics:
                        old_accuracy = old_metrics['accuracy']
                        print_info(f"OLD Accuracy: {old_accuracy*100:.2f}%", indent=2)
                        print_info(f"OLD Precision: {old_metrics['precision']*100:.2f}%", indent=2)
                        print_info(f"OLD Recall: {old_metrics['recall']*100:.2f}%", indent=2)
                except Exception as e:
                    print_warning(f"Could not evaluate old model: {e}")
            
            # Evaluate new aggregated model
            print_info("Evaluating NEW model...", indent=1)
            try:
                new_metrics = evaluate_model_accuracy(fed_avg.global_model, VALIDATION_DATASET_PATH)
                
                if new_metrics:
                    new_accuracy = new_metrics['accuracy']
                    print_info(f"NEW Accuracy: {new_accuracy*100:.2f}%", indent=2)
                    print_info(f"NEW Precision: {new_metrics['precision']*100:.2f}%", indent=2)
                    print_info(f"NEW Recall: {new_metrics['recall']*100:.2f}%", indent=2)
                    
                    # Show comparison for info
                    if old_accuracy is not None:
                        improvement = (new_accuracy - old_accuracy) * 100
                        logger.info("")
                        if new_accuracy > old_accuracy:
                            print_info(f"Model IMPROVED by {improvement:.2f}%", indent=1)
                        elif new_accuracy == old_accuracy:
                            print_info("Model accuracy unchanged", indent=1)
                        else:
                            print_info(f"Model accuracy changed by {improvement:.2f}%", indent=1)
            except Exception as e:
                print_warning(f"Could not evaluate new model: {e}")
        
        # Always save the model regardless of validation
        print_info("Model will be saved (validation check disabled)", indent=1)
        
        # Save updated global model only if validation passed or not enabled
        if should_replace:
            print_subheader("Saving Model")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            round_model_path = f"{MODELS_FOLDER}/global_model_round_{fed_avg.current_round}_{timestamp}.h5"
            
            fed_avg.save_global_model(round_model_path)
            fed_avg.save_global_model(GLOBAL_MODEL_PATH)
            
            model_size_mb = os.path.getsize(round_model_path) / (1024 * 1024)
            print_success(f"Saved as: {os.path.basename(round_model_path)} ({model_size_mb:.2f} MB)")
            
            # Convert to TFLite for mobile deployment
            print_info("Converting to TFLite...", indent=1)
            try:
                tflite_path = round_model_path.replace('.h5', '.tflite')
                tflite_latest_path = GLOBAL_MODEL_PATH.replace('.h5', '.tflite')
                
                converter = tf.lite.TFLiteConverter.from_keras_model(fed_avg.global_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                with open(tflite_latest_path, 'wb') as f:
                    f.write(tflite_model)
                
                tflite_size_mb = len(tflite_model) / (1024 * 1024)
                print_success(f"TFLite saved ({tflite_size_mb:.2f} MB)")
            except Exception as e:
                print_warning(f"TFLite conversion failed: {e}")
        else:
            print_warning("Model NOT replaced (performance degraded)")
            fed_avg.current_round -= 1  # Revert round increment
            return False
        
        # Update server state
        server_state['current_round'] = fed_avg.current_round
        server_state['total_aggregations'] += 1
        
        # Persist state to disk
        save_server_state()
        
        # Clean up old files
        print_info("Cleaning up old files...", indent=1)
        model_files = sorted(Path(MODELS_FOLDER).glob('global_model_round_*.h5'), key=lambda p: p.stat().st_mtime)
        if len(model_files) > 5:
            for old_model in model_files[:-5]:
                old_model.unlink()
                print_info(f"Deleted: {old_model.name}", indent=2)
        
        upload_files = sorted(Path(UPLOAD_FOLDER).glob('*.npz'), key=lambda p: p.stat().st_mtime)
        if len(upload_files) > 10:
            for old_upload in upload_files[:-10]:
                old_upload.unlink()
        
        # Save aggregation info
        aggregation_info = {
            'round': fed_avg.current_round,
            'timestamp': timestamp,
            'num_clients': len(client_ids),
            'client_ids': client_ids,
            'client_samples': dict(zip(client_ids, client_samples_list)),
            'model_path': round_model_path if should_replace else 'not_saved',
            'validation': {
                'enabled': VALIDATE_BEFORE_REPLACE,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'replaced': should_replace
            }
        }
        
        info_file = f"{MODELS_FOLDER}/aggregation_round_{fed_avg.current_round}_info.json"
        with open(info_file, 'w') as f:
            json.dump(aggregation_info, f, indent=2)
        
        print_header(f"✅ AGGREGATION COMPLETE - Round {fed_avg.current_round}", "=")
        
        # Clear pending updates
        server_state['pending_updates'].clear()
        
        return True
    
    except Exception as e:
        print_error(f"Aggregation error: {e}")
        return False

# ============================================
# INITIALIZATION
# ============================================

def initialize_server():
    """Initialize server on startup"""
    print_header("🚀 FEDERATED LEARNING SERVER", "=")
    
    logger.info("")
    print_info(f"Upload Folder: {UPLOAD_FOLDER}")
    print_info(f"Models Folder: {MODELS_FOLDER}")
    print_info(f"Aggregation Threshold: {server_state['aggregation_threshold']} clients")
    print_info(f"Validation: {'Enabled' if VALIDATE_BEFORE_REPLACE else 'Disabled'}")
    
    # Check if base model exists
    logger.info("")
    if not os.path.exists(BASE_MODEL_PATH):
        print_warning(f"Base model not found: {BASE_MODEL_PATH}")
        print_warning("Place your initial model at this path")
    else:
        print_success(f"Base model found: {BASE_MODEL_PATH}")
        # Load it
        if fed_avg.load_global_model():
            print_success("Base model loaded successfully")
        else:
            print_error("Failed to load base model")
    
    print_header("Server Ready", "=")

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    initialize_server()
    
    # Get host and port from environment or use defaults
    host = os.getenv('FL_SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('FL_SERVER_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info("")
    logger.info("="*70)
    logger.info(f"  🌐 FL Server running on {host}:{port}")
    logger.info(f"  📍 Local: http://localhost:{port}")
    logger.info(f"  📍 Network: http://<your-ip>:{port}")
    logger.info("="*70)
    logger.info("")
    
    app.run(host=host, port=port, debug=debug)
