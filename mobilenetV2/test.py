"""
HYBRID ORAL CANCER DETECTION SYSTEM - TESTING SCRIPT
Combines YOLOv11n (Detection) + MobileNetV2 (Classification)

Requirements:
- YOLOv11n model: best.pt or best.onnx
- MobileNetV2 model: mobilenetv2_oral_cancer.tflite
- Test images in a folder
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
YOLO_MODEL_PATH = "runs/detect/oral_cancer_yolo11n/weights/best.pt"  # or best.onnx
MOBILENET_MODEL_PATH = "mobilenetv2_oral_cancer.tflite"

# Test image path - uses environment variable or default relative path
DEFAULT_TEST_IMAGE = str(Path(__file__).parent.parent / 'yolo_dataset/test/images/Benign-lesion-2-_jpg.rf.96b7e9186931843aebace05079c2ec69.jpg')
TEST_IMAGE_PATH = os.getenv('TEST_IMAGE_PATH', DEFAULT_TEST_IMAGE)
# Detection threshold
CONFIDENCE_THRESHOLD = 0.25  # YOLO confidence threshold
IOU_THRESHOLD = 0.45  # Non-max suppression threshold

# Class names
YOLO_CLASSES = {0: 'Background', 1: 'Erythroplakia', 2: 'Leukoplakia'}
MOBILENET_CLASSES = {0: 'Benign', 1: 'Malignant'}

# Colors for visualization (BGR format)
COLORS = {
    'Background': (128, 128, 128),      # Gray
    'Erythroplakia': (0, 0, 255),       # Red
    'Leukoplakia': (255, 255, 255),     # White
    'Benign': (0, 255, 0),              # Green
    'Malignant': (0, 0, 255)            # Red
}

# ============================================================================
# STEP 1: LOAD MODELS
# ============================================================================

print("=" * 70)
print("HYBRID ORAL CANCER DETECTION SYSTEM - INFERENCE TEST")
print("=" * 70)

print("\n[1/5] Loading models...")

# Load YOLO detection model
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    exit(1)

# Load MobileNetV2 classification model (TFLite)
try:
    interpreter = tf.lite.Interpreter(model_path=MOBILENET_MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ MobileNetV2 model loaded: {MOBILENET_MODEL_PATH}")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Failed to load MobileNetV2 model: {e}")
    exit(1)

# ============================================================================
# STEP 2: LOAD TEST IMAGE
# ============================================================================

print(f"\n[2/5] Loading test image: {TEST_IMAGE_PATH}")

if not Path(TEST_IMAGE_PATH).exists():
    print(f"❌ Image not found: {TEST_IMAGE_PATH}")
    print("Please update TEST_IMAGE_PATH in the script!")
    exit(1)

original_image = cv2.imread(TEST_IMAGE_PATH)
if original_image is None:
    print(f"❌ Failed to load image")
    exit(1)

print(f"✅ Image loaded: {original_image.shape}")
annotated_image = original_image.copy()

# ============================================================================
# STEP 3: YOLO DETECTION (Find lesions)
# ============================================================================

print("\n[3/5] Running YOLO detection...")

start_time = time.time()
results = yolo_model.predict(
    source=original_image,
    conf=CONFIDENCE_THRESHOLD,
    iou=IOU_THRESHOLD,
    verbose=False
)
yolo_time = time.time() - start_time

print(f"✅ YOLO inference completed in {yolo_time*1000:.2f}ms")

# Extract detections
detections = []
result = results[0]

if len(result.boxes) == 0:
    print("⚠️  No lesions detected in the image")
else:
    print(f"✅ Detected {len(result.boxes)} lesion(s)")
    
    for box in result.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = YOLO_CLASSES.get(class_id, 'Unknown')
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'class_id': class_id,
            'class_name': class_name
        })
        
        print(f"   - {class_name}: {confidence:.2%} at [{x1}, {y1}, {x2}, {y2}]")

# ============================================================================
# STEP 4: MOBILENETV2 CLASSIFICATION (Classify each lesion)
# ============================================================================

print("\n[4/5] Running MobileNetV2 classification...")

def preprocess_for_mobilenet(image_crop):
    """Preprocess cropped lesion for MobileNetV2"""
    # Resize to 224x224 (MobileNetV2 input size)
    resized = cv2.resize(image_crop, (224, 224))
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    return batched

classification_results = []
total_classification_time = 0

for i, detection in enumerate(detections):
    x1, y1, x2, y2 = detection['bbox']
    
    # Crop the lesion region
    lesion_crop = original_image[y1:y2, x1:x2]
    
    if lesion_crop.size == 0:
        print(f"   ⚠️  Lesion {i+1}: Invalid crop, skipping")
        continue
    
    # Preprocess for MobileNetV2
    input_data = preprocess_for_mobilenet(lesion_crop)
    
    # Run inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    classification_time = time.time() - start_time
    total_classification_time += classification_time
    
    # Get prediction
    prediction = output_data[0]
    
    # Check if output is sigmoid (single value) or softmax (2 values)
    if len(prediction) == 1:
        # Sigmoid output: single probability value
        prob_malignant = float(prediction[0])
        predicted_class = 1 if prob_malignant > 0.5 else 0
        confidence = prob_malignant if predicted_class == 1 else (1 - prob_malignant)
    else:
        # Softmax output: two probability values
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
    
    class_name = MOBILENET_CLASSES[predicted_class]
    
    classification_results.append({
        'lesion_id': i + 1,
        'lesion_type': detection['class_name'],
        'classification': class_name,
        'confidence': confidence,
        'bbox': detection['bbox']
    })
    
    print(f"   Lesion {i+1} ({detection['class_name']}): {class_name} ({confidence:.2%}) [{classification_time*1000:.2f}ms]")

# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================

print("\n[5/5] Generating visualization...")

# Draw bounding boxes and labels
for i, result in enumerate(classification_results):
    x1, y1, x2, y2 = result['bbox']
    lesion_type = result['lesion_type']
    classification = result['classification']
    confidence = result['confidence']
    
    # Choose color based on classification
    color = COLORS.get(classification, (255, 255, 255))
    
    # Draw rectangle
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    
    # Create label
    label = f"{lesion_type} | {classification} {confidence:.1%}"
    
    # Draw label background
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
    
    # Draw label text
    cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image", fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("Hybrid Detection + Classification", fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig("hybrid_system_result.png", dpi=150, bbox_inches='tight')
print("✅ Visualization saved: hybrid_system_result.png")

plt.show()

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

total_time = yolo_time + total_classification_time
print(f"YOLO Detection Time:        {yolo_time*1000:.2f}ms")
print(f"MobileNetV2 Total Time:     {total_classification_time*1000:.2f}ms")
if len(classification_results) > 0:
    print(f"MobileNetV2 Avg Per Lesion: {(total_classification_time/len(classification_results))*1000:.2f}ms")
print(f"Total Inference Time:       {total_time*1000:.2f}ms")
print(f"\n🚀 System is {'REAL-TIME CAPABLE' if total_time < 0.25 else 'NOT REAL-TIME'} (<250ms target)")

# ============================================================================
# CLINICAL REPORT
# ============================================================================

print("\n" + "=" * 70)
print("CLINICAL REPORT")
print("=" * 70)

if len(classification_results) == 0:
    print("No lesions detected - Image appears normal")
else:
    malignant_count = sum(1 for r in classification_results if r['classification'] == 'Malignant')
    benign_count = len(classification_results) - malignant_count
    
    print(f"Total Lesions Detected: {len(classification_results)}")
    print(f"  - Benign: {benign_count}")
    print(f"  - Malignant: {malignant_count}")
    
    print("\nDetailed Findings:")
    for result in classification_results:
        print(f"  Lesion {result['lesion_id']}:")
        print(f"    Type: {result['lesion_type']}")
        print(f"    Classification: {result['classification']} ({result['confidence']:.1%} confidence)")
        print(f"    Location: {result['bbox']}")
    
    # Risk assessment
    if malignant_count > 0:
        print("\n⚠️  RECOMMENDATION: URGENT - Refer to oral pathologist immediately")
        print("    Malignant lesions detected requiring biopsy and specialist evaluation")
    elif benign_count > 0:
        print("\n✅ RECOMMENDATION: MONITOR - Schedule follow-up in 3-6 months")
        print("    Benign lesions detected - regular monitoring recommended")
    else:
        print("\n✅ No concerning lesions detected")

print("=" * 70)
