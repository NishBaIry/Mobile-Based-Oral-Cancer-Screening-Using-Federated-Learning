import os
from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================
# SETUP
# ============================================

print("="*60)
print("YOLO TRAINING - Oral Cancer Lesion Detection")
print("="*60)

# Check GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  No GPU detected - training will be VERY slow on CPU")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit()

# ============================================
# DATASET SETUP
# ============================================

# Dataset path - uses environment variable or relative path
DATASET_PATH = os.getenv('YOLO_DATASET_PATH', str(Path(__file__).parent.parent / 'yolo_dataset'))

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ Dataset not found at: {DATASET_PATH}")
    print("Please download your dataset from Roboflow first!")
    exit()

# Find data.yaml
data_yaml = None
for root, dirs, files in os.walk(DATASET_PATH):
    if 'data.yaml' in files:
        data_yaml = os.path.join(root, 'data.yaml')
        break

if data_yaml is None:
    print(f"\n❌ data.yaml not found in {DATASET_PATH}")
    print("Make sure you downloaded the dataset in YOLOv11 format from Roboflow")
    exit()

print(f"\n✅ Dataset found: {DATASET_PATH}")
print(f"✅ Config file: {data_yaml}")

# ============================================
# TRAINING CONFIGURATION
# ============================================

# Choose model size
MODEL_NAME = "yolo11n.pt"  # Nano - fastest, smallest
# MODEL_NAME = "yolo11s.pt"  # Small - better accuracy

print(f"\n🏗️ Model: {MODEL_NAME}")

CONFIG = {
    # Dataset
    'data': data_yaml,
    
    # Training
    'epochs': 100,
    'batch': 16,  # RTX 4060 can handle this
    'imgsz': 640,
    
    # Optimization
    'optimizer': 'auto',
    'lr0': 0.01,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Augmentation (YOLO does this automatically)
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,    # HSV-Saturation
    'hsv_v': 0.4,    # HSV-Value
    'degrees': 0.0,  # Rotation (disabled - Roboflow already did this)
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,  # No vertical flip
    'fliplr': 0.5,  # Horizontal flip
    'mosaic': 1.0,  # Mosaic augmentation
    'mixup': 0.0,   # Mixup augmentation
    'copy_paste': 0.0,
    
    # Regularization
    'patience': 20,  # Early stopping
    'close_mosaic': 10,  # Disable mosaic last 10 epochs
    
    # Device
    'device': 0,  # GPU 0
    'workers': 8,  # Data loading threads
    
    # Saving
    'project': 'runs/detect',
    'name': 'oral_cancer_yolo11n',
    'exist_ok': True,
    'save': True,
    'save_period': -1,  # Save checkpoint every N epochs (-1 = only save last and best)
    
    # Validation
    'val': True,
    
    # Other
    'pretrained': True,
    'verbose': True,
    'seed': 42,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': True,
    'amp': True,  # Automatic Mixed Precision
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'plots': True,
    'cache': False,  # Set to True if you have 32GB+ RAM
}

print("\n⚙️ Training Configuration:")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch']}")
print(f"  Image size: {CONFIG['imgsz']}")
print(f"  Device: GPU {CONFIG['device']}")
print(f"  Workers: {CONFIG['workers']}")
print(f"  Patience: {CONFIG['patience']} (early stopping)")

# ============================================
# LOAD MODEL
# ============================================

print(f"\n📥 Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

# ============================================
# TRAIN
# ============================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print("\n⏳ This will take approximately 2-3 hours on RTX 4060...")
print("💡 Tip: Training will auto-stop if no improvement for 20 epochs\n")

results = model.train(**CONFIG)

# ============================================
# EVALUATE ON VALIDATION SET
# ============================================

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)

metrics = model.val()

print(f"\n📊 Overall Performance:")
print(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
print(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
print(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")

print(f"\n📊 Per-Class Performance (mAP50):")
class_names = list(model.names.values())
for i, class_name in enumerate(class_names):
    if i < len(metrics.box.maps):
        print(f"  {class_name:20s}: {metrics.box.maps[i]:.4f} ({metrics.box.maps[i]*100:.2f}%)")

# ============================================
# TEST INFERENCE
# ============================================

print("\n" + "="*60)
print("TESTING INFERENCE")
print("="*60)

# Find a test image
test_dir = os.path.join(DATASET_PATH, 'test', 'images')
if os.path.exists(test_dir):
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if test_images:
        test_img = os.path.join(test_dir, test_images[0])
        print(f"\n🧪 Running inference on: {test_images[0]}")
        
        results = model.predict(
            source=test_img,
            save=True,
            conf=0.25,
            iou=0.45,
            show_labels=True,
            show_conf=True
        )
        print("✅ Inference complete! Check runs/detect/predict/ for results")

# ============================================
# EXPORT FOR MOBILE
# ============================================

print("\n" + "="*60)
print("EXPORTING FOR MOBILE DEPLOYMENT")
print("="*60)

best_model_path = f"runs/detect/oral_cancer_yolo11n/weights/best.pt"

if os.path.exists(best_model_path):
    print(f"\n📱 Exporting best model: {best_model_path}")
    
    # Load best model
    best_model = YOLO(best_model_path)
    
    # Export to TFLite
    print("\n  Exporting to TFLite...")
    try:
        best_model.export(format='tflite', imgsz=640, int8=False)
        print("  ✅ TFLite export complete")
    except Exception as e:
        print(f"  ⚠️  TFLite export failed: {e}")
    
    # Export to ONNX
    print("\n  Exporting to ONNX...")
    try:
        best_model.export(format='onnx', imgsz=640, simplify=True)
        print("  ✅ ONNX export complete")
    except Exception as e:
        print(f"  ⚠️  ONNX export failed: {e}")

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print(f"\n📂 Results Location:")
print(f"  Main folder: runs/detect/oral_cancer_yolo11n/")
print(f"  Best weights: runs/detect/oral_cancer_yolo11n/weights/best.pt")
print(f"  Last weights: runs/detect/oral_cancer_yolo11n/weights/last.pt")
print(f"  Training plots: runs/detect/oral_cancer_yolo11n/")

print(f"\n📊 Final Performance:")
print(f"  mAP50: {metrics.box.map50*100:.2f}%")

print(f"\n📋 Your Complete Hybrid System:")
print(f"  🔍 YOLO Detection:        mAP50 = {metrics.box.map50*100:.1f}%")
print(f"  🎯 MobileNetV2 Classification: Accuracy = 86.0%")

# ============================================
# PERFORMANCE EVALUATION
# ============================================

print("\n" + "="*60)
print("PERFORMANCE EVALUATION")
print("="*60)

if metrics.box.map50 >= 0.80:
    print("\n🏆 EXCELLENT! mAP50 ≥ 80%")
    print("   Your model is production-ready!")
    print("   ✅ Ready for Flutter deployment")
elif metrics.box.map50 >= 0.75:
    print("\n✅ VERY GOOD! mAP50 ≥ 75%")
    print("   Model performance is solid")
    print("   ✅ Ready for deployment")
elif metrics.box.map50 >= 0.70:
    print("\n👍 GOOD! mAP50 ≥ 70%")
    print("   Model is usable")
    print("   💡 Consider training YOLOv11s for better accuracy")
elif metrics.box.map50 >= 0.65:
    print("\n⚠️  ACCEPTABLE. mAP50 ≥ 65%")
    print("   Model needs improvement")
    print("   💡 Recommendation: Retrain with yolo11s.pt")
else:
    print("\n❌ LOW PERFORMANCE. mAP50 < 65%")
    print("   Recommendations:")
    print("   1. Retrain with yolo11s.pt for more capacity")
    print("   2. Check dataset quality and annotations")
    print("   3. Try more epochs (150-200)")

print("\n" + "="*60)
print("Next Steps:")
print("="*60)
print("1. Check training plots in: runs/detect/oral_cancer_yolo11n/")
print("2. Test the model on new images")
print("3. If mAP50 < 75%, retrain with yolo11s.pt")
print("4. Export to TFLite for Flutter app")
print("5. Integrate with MobileNetV2 for complete system")

print("\n✅ Training script complete!")