import os
import platform

# ============================================
# TENSORFLOW SETUP - CROSS-PLATFORM
# Must be set BEFORE importing TensorFlow
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Linux CUDA fix - set XLA flags to find libdevice
if platform.system() == 'Linux':
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'

# M1 Mac-specific fix (only apply on Apple Silicon)
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Now load dotenv for other settings
from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse

# ============================================
# COMMAND LINE ARGUMENTS
# ============================================
parser = argparse.ArgumentParser(description='Federated Learning Client - Fine-tuning Script')
parser.add_argument('--dataset-path', type=str, default=None,
                    help='Path to dataset folder (overrides .env value)')
parser.add_argument('--epochs', type=int, default=None,
                    help='Number of training epochs (overrides .env value)')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size for training (overrides .env value)')
parser.add_argument('--learning-rate', type=float, default=None,
                    help='Learning rate for training (overrides .env value)')
args = parser.parse_args()

# ============================================
# CONFIGURATION - FINE-TUNING FOR FEDERATED LEARNING
# Priority: Command-line args > .env > defaults
# ============================================
OLD_MODEL_PATH = os.getenv('BASE_MODEL_PATH', '../mobilenetv2_oral_cancer_best.h5')
DATASET_PATH = args.dataset_path or os.getenv('LOCAL_DATASET_PATH', '../2nd_traning')  # CLI overrides .env
BATCH_SIZE = args.batch_size or int(os.getenv('BATCH_SIZE', 8))  # Default 8 for 8GB GPU
IMG_SIZE = int(os.getenv('IMG_SIZE', 224))
EPOCHS = args.epochs or int(os.getenv('EPOCHS', 1))  # Fewer epochs for fine-tuning
LEARNING_RATE = args.learning_rate or float(os.getenv('LEARNING_RATE', 1e-5))  # Much lower LR for fine-tuning

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"mobilenetv2_finetuned_{TIMESTAMP}"

print("🏥 FEDERATED LEARNING - FINE-TUNING SCRIPT")
print("="*70)
print(f"📅 Training run: {TIMESTAMP}")
print(f"📂 Base model: {OLD_MODEL_PATH}")
print(f"📂 Local dataset: {DATASET_PATH}")
print(f"🎯 Strategy: Fine-tune pre-trained model on local hospital data")
print("="*70)

# ============================================
# GPU SETUP - CROSS-PLATFORM
# ============================================
# M1 Mac-specific optimization (skip on Linux/Windows)
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ GPU: {physical_devices[0]}")
else:
    print("ℹ️  Running on CPU")

# Cross-platform Adam optimizer helper
def get_adam_optimizer(learning_rate):
    """Returns Adam optimizer - uses legacy on M1 Mac, standard elsewhere"""
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

# ============================================
# DATA LOADING
# ============================================
print("\n📂 Loading local hospital dataset...")

# Light augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Increased from 10
    width_shift_range=0.2,  # Increased from 0.1
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,  # New: zoom augmentation
    brightness_range=[0.8, 1.2],  # New: brightness variation
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

valid_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'valid'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Training:   {train_generator.samples} images")
print(f"✅ Validation: {valid_generator.samples} images")
print(f"✅ Test:       {test_generator.samples} images")

# ============================================
# LOAD PRE-TRAINED MODEL
# ============================================
print(f"\n🔄 Loading pre-trained model: {OLD_MODEL_PATH}")

try:
    model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)
    print("✅ Pre-trained model loaded successfully!")
    
    # Evaluate old model on new dataset
    model.compile(
        optimizer=get_adam_optimizer(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\n📊 Pre-trained model performance on NEW dataset:")
    old_loss, old_acc, old_prec, old_recall, old_auc = model.evaluate(
        test_generator,
        verbose=0
    )
    print(f"   Accuracy:  {old_acc*100:.2f}%")
    print(f"   Precision: {old_prec*100:.2f}%")
    print(f"   Recall:    {old_recall*100:.2f}%")
    print(f"   AUC:       {old_auc:.4f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Training from scratch instead...")
    exit(1)

# ============================================
# ADD REGULARIZATION TO PREVENT OVERFITTING
# ============================================
print("\n🎯 Adding regularization to prevent overfitting...")

# Add dropout after the last dense layer
x = model.layers[-2].output  # Before final dense (which is the last layer)
x = tf.keras.layers.Dropout(0.5, name='dropout_extra')(x)  # Add dropout with unique name
predictions = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_final')(x)
model = tf.keras.Model(inputs=model.input, outputs=predictions)

# Compile again after modification
model.compile(
    optimizer=get_adam_optimizer(LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("✅ Added dropout regularization")

# Freeze base layers to prevent overfitting (only train top layers)
print("\n🎯 Freezing base layers for partial fine-tuning...")
for layer in model.layers[:-4]:  # Freeze all but last 4 layers
    layer.trainable = False

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✅ Trainable parameters after freezing: {trainable:,}")

# ============================================
# FINE-TUNING STRATEGY
# ============================================
print("\n🎯 Fine-tuning strategy:")
print("   - Low learning rate (1e-5) to preserve learned features")
print("   - Partial fine-tuning (base layers frozen)")
print("   - Enhanced data augmentation")
print("   - Added dropout regularization")
print("   - Aggressive early stopping to prevent overfitting")
print("   - Dropout layers already in model help regularization")

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n✅ Trainable parameters: {trainable:,}")

# Model already compiled above
model.summary()

# ============================================
# CALLBACKS - AGGRESSIVE EARLY STOPPING
# ============================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reduced from 7 to stop earlier
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,  # Reduced from 3
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================
# FINE-TUNE
# ============================================
print("\n" + "="*70)
print("🚀 FINE-TUNING ON LOCAL HOSPITAL DATA")
print("="*70)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "="*70)
print("📊 EVALUATION AFTER FINE-TUNING")
print("="*70)

test_loss, test_acc, test_prec, test_recall, test_auc = model.evaluate(
    test_generator,
    verbose=1
)

print(f"\n✅ Fine-tuned Model Performance:")
print(f"   Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision: {test_prec*100:.2f}%")
print(f"   Recall:    {test_recall*100:.2f}%")
print(f"   AUC:       {test_auc:.4f}")

# ============================================
# PERFORMANCE COMPARISON
# ============================================
print("\n" + "="*70)
print("📊 PERFORMANCE COMPARISON (on new dataset)")
print("="*70)
print(f"{'Metric':<15} {'Before':<15} {'After':<15} {'Change':<15}")
print("-" * 70)

metrics = [
    ('Accuracy', old_acc, test_acc),
    ('Precision', old_prec, test_prec),
    ('Recall', old_recall, test_recall),
    ('AUC', old_auc, test_auc)
]

for metric_name, before, after in metrics:
    change = after - before
    change_str = f"{change:+.4f} ({change*100:+.2f}%)"
    if change > 0:
        change_str = f"✅ {change_str}"
    elif change < 0:
        change_str = f"⚠️  {change_str}"
    else:
        change_str = f"➖ {change_str}"
    
    print(f"{metric_name:<15} {before*100:>6.2f}%       {after*100:>6.2f}%       {change_str}")

print("="*70)

# Check for overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = (train_acc - val_acc) * 100

print(f"\n🔍 Overfitting Check:")
print(f"   Final training accuracy:   {train_acc*100:.2f}%")
print(f"   Final validation accuracy: {val_acc*100:.2f}%")
print(f"   Gap: {gap:.2f}%")

if gap < 5:
    print(f"   ✅ Good generalization (gap < 5%)")
elif gap < 10:
    print(f"   ⚠️  Slight overfitting (gap < 10%)")
else:
    print(f"   ❌ Overfitting detected (gap > 10%)")

# ============================================
# SAVE MODELS
# ============================================
print("\n💾 Saving models for federated learning...")

# Create client_weights directory
os.makedirs('client_weights', exist_ok=True)

# Clean up old training files (keep only last 3 runs)
print("\n🗑️  Cleaning up old training files...")
existing_files = sorted(Path('client_weights').glob('mobilenetv2_finetuned_*'), key=lambda p: p.stat().st_mtime)
if len(existing_files) > 9:  # Keep last 3 runs * 3 files each = 9 files (weights, deltas, report)
    files_to_delete = existing_files[:-9]
    for old_file in files_to_delete:
        old_file.unlink()
        print(f"   Deleted: {old_file.name}")

# Save weights only (for FedAvg)
weights_path = f'client_weights/{MODEL_NAME}_weights.h5'
model.save_weights(weights_path)
print(f"✅ Weights only: {weights_path}")

# Get weight updates (delta from original)
print("\n📊 Computing weight deltas for FedAvg...")
old_model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)
old_weights = old_model.get_weights()
new_weights = model.get_weights()

# Save deltas
deltas = [new_w - old_w for new_w, old_w in zip(new_weights, old_weights)]
np.savez_compressed(f'client_weights/{MODEL_NAME}_deltas.npz', *deltas)
print(f"✅ Weight deltas: client_weights/{MODEL_NAME}_deltas.npz (for efficient FL)")

# ============================================
# TRAINING REPORT
# ============================================
report_path = f'client_weights/{MODEL_NAME}_report.txt'
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("FEDERATED LEARNING - FINE-TUNING REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: M1 Mac (Hospital Client)\n")
    f.write(f"Base Model: {OLD_MODEL_PATH}\n")
    f.write(f"Local Dataset: {DATASET_PATH}\n")
    f.write(f"Strategy: Fine-tuning pre-trained model\n\n")
    
    f.write("Dataset Size:\n")
    f.write(f"  Training:   {train_generator.samples} images\n")
    f.write(f"  Validation: {valid_generator.samples} images\n")
    f.write(f"  Test:       {test_generator.samples} images\n\n")
    
    f.write("Performance Before Fine-tuning (on new data):\n")
    f.write(f"  Accuracy:  {old_acc*100:.2f}%\n")
    f.write(f"  Precision: {old_prec*100:.2f}%\n")
    f.write(f"  Recall:    {old_recall*100:.2f}%\n")
    f.write(f"  AUC:       {old_auc:.4f}\n\n")
    
    f.write("Performance After Fine-tuning:\n")
    f.write(f"  Accuracy:  {test_acc*100:.2f}%\n")
    f.write(f"  Precision: {test_prec*100:.2f}%\n")
    f.write(f"  Recall:    {test_recall*100:.2f}%\n")
    f.write(f"  AUC:       {test_auc:.4f}\n\n")
    
    f.write("Improvement:\n")
    for metric_name, before, after in metrics:
        change = after - before
        f.write(f"  {metric_name}: {change:+.4f} ({change*100:+.2f}%)\n")
    
    f.write(f"\nOverfitting Check:\n")
    f.write(f"  Train-Val gap: {gap:.2f}%\n")
    
    f.write("\nFiles for Federated Learning:\n")
    f.write(f"  - {weights_path} (full weights)\n")
    f.write(f"  - client_weights/{MODEL_NAME}_deltas.npz (weight updates only)\n\n")
    
    f.write("Next Steps:\n")
    f.write("  1. Upload weight deltas to FL server\n")
    f.write("  2. Server aggregates with FedAvg\n")
    f.write("  3. Download updated global model\n")
    f.write("  4. Repeat training cycle\n")

print(f"✅ Report: {report_path}")

# ============================================
# PLOT TRAINING HISTORY
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#4A90E2')
axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2, color='#7B68EE')
axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='#4A90E2')
axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2, color='#7B68EE')
axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[0, 2].plot(history.history['precision'], label='Train', linewidth=2, color='#4A90E2')
axes[0, 2].plot(history.history['val_precision'], label='Val', linewidth=2, color='#7B68EE')
axes[0, 2].set_title('Precision', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Recall
axes[1, 0].plot(history.history['recall'], label='Train', linewidth=2, color='#4A90E2')
axes[1, 0].plot(history.history['val_recall'], label='Val', linewidth=2, color='#7B68EE')
axes[1, 0].set_title('Recall', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# AUC
axes[1, 1].plot(history.history['auc'], label='Train', linewidth=2, color='#4A90E2')
axes[1, 1].plot(history.history['val_auc'], label='Val', linewidth=2, color='#7B68EE')
axes[1, 1].set_title('AUC', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Learning Rate
axes[1, 2].plot(history.history['lr'], linewidth=2, color='#E74C3C')
axes[1, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('LR')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f'Fine-tuning History - {MODEL_NAME}', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'client_weights/{MODEL_NAME}_history.png', dpi=300, bbox_inches='tight')
print(f"✅ Training plots: client_weights/{MODEL_NAME}_history.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ FINE-TUNING COMPLETE!")
print("="*70)
print(f"\n📦 Files Generated:")
print(f"   1. {weights_path} (full weights)")
print(f"   2. client_weights/{MODEL_NAME}_deltas.npz (weight updates - RECOMMENDED)")
print(f"   3. {report_path} (training report)")
print(f"   4. client_weights/{MODEL_NAME}_history.png (plots)")
print(f"\n🔄 Next Steps:")
print(f"   1. Use 'fl_client_send.py' to upload weights to FL server")
print(f"   2. Configure server IP in .env file")
print(f"   3. Server will aggregate and send back global model")
print(f"\n💡 Run: python fl_client_send.py --model {MODEL_NAME}")

print("\n✅ Ready for federated learning! 🚀")
