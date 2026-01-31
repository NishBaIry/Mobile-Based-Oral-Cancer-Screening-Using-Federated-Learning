"""
YOLOv11n TFLite Export Script
Fixes NumPy compatibility issue and exports best.pt to TFLite
"""

import subprocess
import sys

print("=" * 60)
print("YOLO TFLite EXPORT - NumPy Fix Applied")
print("=" * 60)

# Step 1: Fix NumPy version
print("\n[1/3] Fixing NumPy version...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "--quiet"])
    print("✅ NumPy downgraded successfully")
except Exception as e:
    print(f"❌ NumPy fix failed: {e}")
    sys.exit(1)

# Step 2: Import YOLO after NumPy fix
print("\n[2/3] Loading YOLO model...")
try:
    from ultralytics import YOLO
    print("✅ Ultralytics imported successfully")
except Exception as e:
    print(f"❌ Failed to import YOLO: {e}")
    sys.exit(1)

# Step 3: Export to TFLite
print("\n[3/3] Exporting to TFLite...")
try:
    # Load your trained model
    model = YOLO("runs/detect/oral_cancer_yolo11n/weights/best.pt")
    
    # Export to TFLite
    print("🔄 Starting export (this may take a few minutes)...")
    model.export(format="tflite", imgsz=640, int8=False)
    
    print("\n" + "=" * 60)
    print("✅ EXPORT SUCCESSFUL!")
    print("=" * 60)
    print(f"📁 TFLite model saved to: runs/detect/oral_cancer_yolo11n/weights/best_saved_model/")
    print(f"📄 Look for: best_float32.tflite or best_int8.tflite")
    
except Exception as e:
    print(f"\n❌ Export failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if best.pt exists at the specified path")
    print("2. Try running in a fresh virtual environment")
    print("3. Alternative: Use ONNX format (already exported successfully)")
    sys.exit(1)

print("\n🎉 You can now use best_float32.tflite in your Flutter app!")