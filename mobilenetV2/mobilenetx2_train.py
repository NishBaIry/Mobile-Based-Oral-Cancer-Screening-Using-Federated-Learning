import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = ''  # UPDATE THIS!
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 50

# ============================================
# GPU SETUP
# ============================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ GPU: {physical_devices[0]}")

# ============================================
# DATA LOADING - NO extra augmentation (already 3x in Roboflow)
# ============================================

print("\n📂 Loading 3x augmented dataset...")

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

valid_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'valid'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Training: {train_generator.samples}")
print(f"✅ Validation: {valid_generator.samples}")
print(f"✅ Test: {test_generator.samples}")

# ============================================
# BUILD MODEL - MobileNetV2
# ============================================

print("\n🏗️ Building MobileNetV2 model...")

model = Sequential([
    MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=1.0  # Full model (use 0.75 for even lighter version)
    ),
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# CRITICAL: Unfreeze ALL layers immediately (learned from EfficientNet mistake)
model.layers[0].trainable = True

print(f"✅ Total parameters: {model.count_params():,}")
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"✅ Trainable parameters: {trainable:,}")

# ============================================
# COMPILE
# ============================================

# Use lower learning rate since we're fine-tuning pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# ============================================
# CALLBACKS
# ============================================

callbacks = [
    ModelCheckpoint(
        'mobilenetv2_oral_cancer_best.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================
# TRAIN
# ============================================

print("\n" + "="*60)
print("TRAINING MobileNetV2")
print("="*60)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# EVALUATE ON TEST SET
# ============================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

test_loss, test_acc, test_prec, test_recall, test_auc = model.evaluate(
    test_generator,
    verbose=1
)

print(f"\n📊 Final Test Results:")
print(f"   Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision: {test_prec*100:.2f}%")
print(f"   Recall:    {test_recall*100:.2f}%")
print(f"   AUC:       {test_auc:.4f}")

# ============================================
# SAVE FINAL MODEL
# ============================================

model.save('mobilenetv2_oral_cancer_final.h5')
print("\n💾 Final model saved!")

# ============================================
# EXPORT FOR MOBILE
# ============================================

print("\n📱 Exporting for mobile deployment...")

# TensorFlow Lite (Android)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('mobilenetv2_oral_cancer.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ TFLite model saved: mobilenetv2_oral_cancer.tflite")

# ============================================
# PLOT TRAINING HISTORY
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val', linewidth=2)
axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val', linewidth=2)
axes[1, 1].set_title('Recall (Sensitivity)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mobilenetv2_training_history.png', dpi=300)
print("📈 Training history saved!")

print("\n✅ ALL DONE!")
print("\n📋 Your Hybrid System:")
print("   🔍 YOLO (Detection):      87.5% mAP - Pre-trained")
print(f"   🎯 MobileNetV2 (Class):   {test_acc*100:.1f}% accuracy")
print("\n🚀 Ready for Flutter deployment!")
