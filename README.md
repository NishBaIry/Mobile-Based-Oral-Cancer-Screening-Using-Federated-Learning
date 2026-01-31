# Mobile-Based Oral Cancer Screening Using Federated Learning

A hybrid deep learning mobile app with **YOLOv11n** for lesion detection and **MobileNetV2** for malignancy classification, using federated learning for privacy-preserving collaborative training across multiple clients.

## Quick Start

### 1. Use Pre-trained Models (Recommended)

Pre-trained models are already included in `models/` directory. No training required!

**For Mobile App:**
```bash
# Copy TFLite models to Flutter assets
cp models/*.tflite my_app/android/app/src/main/assets/
cd my_app && flutter run
```

**For FL Server:**
```bash
cp models/mobilenetv2_oral_cancer_best.h5 server/models/global_model.h5
```

### 2. Train Your Own Models (Optional)

If you want to train models for a different dataset:

```bash
# Step 1: Train YOLO detection model
cd Yolo/
python yolo_train.py
# Output: best.pt → export to .tflite

# Step 2: Train MobileNetV2 classification model
cd ../mobilenetV2/
python mobilenetx2_train.py
# Output: mobilenetv2_*.h5
```

### 3. Run Federated Learning

**Server:**
```bash
cd server/
cp .env.example .env
# Edit .env with your settings
mkdir -p models && cp ../mobilenetv2_oral_cancer_best.h5 models/global_model.h5
python fl_server.py
```

**Client:**
```bash
cd client/
cp .env.example .env
# Edit .env - set FL_SERVER_HOST to server IP
python dataset_gui.py  # GUI mode
# OR
python fl_client_train_and_upload.py --client_id client_1 --dataset_path /path/to/data
```

### 3. Mobile App (Inference Only)

```bash
cd my_app/
flutter pub get
flutter run
```
Place your `.tflite` models in `my_app/android/app/src/main/assets/`

## Project Structure

```
├── models/                  # Pre-trained base models
│   ├── yolo_oral_cancer.tflite        # YOLOv11n detection model
│   └── mobilenetv2_oral_cancer.tflite # MobileNetV2 classification model
├── Yolo/                    # YOLO training scripts
├── mobilenetV2/             # MobileNetV2 training scripts
├── server/                  # FL aggregation server
├── client/                  # FL client training
├── my_app/                  # Flutter mobile app
└── FL_ARCHITECTURE.md       # Detailed FL documentation
```

## Pre-trained Models

Base models are included in `models/` directory:

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOv11n | `yolo_oral_cancer.tflite` | ~10 MB | Mobile app - lesion detection |
| YOLOv11n | `yolo_weights.pt` | ~5.5 MB | Training weights |
| MobileNetV2 | `mobilenetv2_oral_cancer.tflite` | ~2.5 MB | Mobile app - classification |
| MobileNetV2 | `mobilenetv2_oral_cancer_best.h5` | ~29 MB | FL server base model |

## How It Works

```
[Image] → [YOLO Detection] → [Crop Region] → [MobileNetV2 Classification] → [Result]
```

- **YOLO**: Detects and localizes objects (640x640 input)
- **MobileNetV2**: Classifies detected regions (224x224 input)
- **FL Server**: Aggregates weights from multiple clients using FedAvg
- **Mobile App**: Runs inference on-device for privacy

## Requirements

- Python 3.8+
- TensorFlow 2.x
- CUDA (optional, for GPU training)
- Flutter 3.x (for mobile app)

```bash
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

## Configuration

Copy `.env.example` to `.env` in both `server/` and `client/` directories and edit as needed.

| Variable | Description |
|----------|-------------|
| `FL_SERVER_HOST` | Server IP (use `0.0.0.0` for server) |
| `FL_SERVER_PORT` | Port (default: 5000) |
| `AGGREGATION_THRESHOLD` | Min clients before aggregation |

## Documentation

- [FL_ARCHITECTURE.md](FL_ARCHITECTURE.md) - Detailed federated learning architecture
- [server/README.md](server/README.md) - Server setup guide
