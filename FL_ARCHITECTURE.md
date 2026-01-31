# Federated Learning System

## Overview

This project implements a **Federated Learning (FL)** system using MobileNetV2 for image classification. Multiple clients can collaboratively train a shared model without sharing their private data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEDERATED LEARNING ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   Client A   │      │   Client B   │      │   Client C   │
    │              │      │              │      │              │
    │ Local Data   │      │ Local Data   │      │ Local Data   │
    │ (Private)    │      │ (Private)    │      │ (Private)    │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           │  ┌──────────────────┼──────────────────┐  │
           │  │                  │                  │  │
           ▼  ▼                  ▼                  ▼  ▼
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │ Train Local │       │ Train Local │       │ Train Local │
    │   Model     │       │   Model     │       │   Model     │
    └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
           │                     │                     │
           │    Upload Weights   │    Upload Weights   │
           │    (NOT data)       │    (NOT data)       │
           ▼                     ▼                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │                    FL SERVER                            │
    │                                                         │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │              FedAvg Aggregation                 │   │
    │   │                                                 │   │
    │   │   W_global = (W_a + W_b + W_c) / 3             │   │
    │   │                                                 │   │
    │   │   - Receives weights from all clients          │   │
    │   │   - Averages them (weighted by samples)        │   │
    │   │   - Validates new model accuracy               │   │
    │   │   - Updates global model if improved           │   │
    │   └─────────────────────────────────────────────────┘   │
    │                                                         │
    └────────────────────────┬────────────────────────────────┘
                             │
                             │ Download Updated
                             │ Global Model
                             ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Client A   │  │   Client B   │  │   Client C   │
    │              │  │              │  │              │
    │ Uses updated │  │ Uses updated │  │ Uses updated │
    │ global model │  │ global model │  │ global model │
    │ for next     │  │ for next     │  │ for next     │
    │ round        │  │ round        │  │ round        │
    └──────────────┘  └──────────────┘  └──────────────┘

                    ↺ Repeat for N rounds
```

## Key Benefits

| Traditional ML | Federated Learning |
|----------------|-------------------|
| Data sent to central server | Data stays local (privacy) |
| Single point of failure | Distributed training |
| Privacy concerns | Data never leaves client |
| Limited data access | Learn from multiple sources |

## How It Works

### Round Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE FL ROUND                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CLIENT DOWNLOADS GLOBAL MODEL                           │
│     └─► GET /api/download_global_model                      │
│                                                             │
│  2. CLIENT TRAINS LOCALLY                                   │
│     └─► Fine-tune MobileNetV2 on local dataset              │
│     └─► Epochs: 3-5 (configurable)                          │
│     └─► Data never leaves the client                        │
│                                                             │
│  3. CLIENT UPLOADS WEIGHTS                                  │
│     └─► POST /api/upload_weights                            │
│     └─► Only model weights sent (not training data)         │
│                                                             │
│  4. SERVER AGGREGATES (when threshold reached)              │
│     └─► FedAvg: Average weights from all clients            │
│     └─► Validate on test set                                │
│     └─► Update global model if accuracy improved            │
│                                                             │
│  5. REPEAT                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
DTL/
├── server/                          # FL Server
│   ├── fl_server.py                 # Flask API server (main)
│   ├── fedavg.py                    # FedAvg aggregation algorithm
│   ├── models/                      # Global model storage
│   └── uploads/                     # Temporary client uploads
│
├── client/                          # FL Client
│   ├── dataset_gui.py               # GUI for training configuration
│   ├── mobilenetv2_finetune_federated.py  # Local training script
│   ├── fl_client_train_and_upload.py      # Orchestrates workflow
│   ├── fl_client_send.py            # Upload weights to server
│   └── fl_client_download.py        # Download global model
│
├── mobilenetx2_train.py             # Base model training (non-FL)
└── my_app/                          # Flutter mobile app (inference)
```

## Core Components

### 1. FL Server (`server/fl_server.py`)

```python
# Key API Endpoints:
POST /api/upload_weights      # Client uploads trained weights
GET  /api/download_global_model  # Client downloads latest model
GET  /api/status              # Check server status & round info
POST /api/trigger_aggregation # Manual aggregation trigger
```

### 2. FedAvg Algorithm (`server/fedavg.py`)

```python
# Simplified FedAvg:
def federated_averaging(client_weights, client_samples):
    """
    Average weights from multiple clients, weighted by sample count.

    W_global = Σ(n_k * W_k) / Σ(n_k)

    Where:
    - W_k = weights from client k
    - n_k = number of samples client k trained on
    """
    total_samples = sum(client_samples)
    averaged_weights = []

    for layer_idx in range(num_layers):
        weighted_sum = sum(
            w[layer_idx] * n / total_samples
            for w, n in zip(client_weights, client_samples)
        )
        averaged_weights.append(weighted_sum)

    return averaged_weights
```

### 3. Client Training (`client/mobilenetv2_finetune_federated.py`)

- Loads global model from server
- Fine-tunes MobileNetV2 on local dataset
- Binary classification
- Saves only the model weights (not full model)

### 4. Model Architecture

```
MobileNetV2 (ImageNet pretrained)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(128, ReLU)
    │
    ▼
Dropout(0.2)
    │
    ▼
Dense(1, Sigmoid)  →  Output: Classification probability [0-1]
```

## Quick Start

### Server Setup

```bash
cd server/
pip install -r requirements.txt

# Place initial model
mkdir -p models
cp ../mobilenetv2_oral_cancer_best.h5 models/global_model.h5

# Start server
python fl_server.py
```

### Client Setup

```bash
cd client/

# Configure server address in .env
FL_SERVER_HOST=<server-ip>
FL_SERVER_PORT=5000

# Launch GUI
python dataset_gui.py

# Or run directly
python fl_client_train_and_upload.py \
    --client_id client_1 \
    --dataset_path /path/to/data \
    --epochs 3
```

## Communication Protocol

```
CLIENT                                    SERVER
   │                                         │
   │  GET /api/download_global_model         │
   │────────────────────────────────────────>│
   │                                         │
   │         200 OK + model.h5               │
   │<────────────────────────────────────────│
   │                                         │
   │         [Local Training]                │
   │                                         │
   │  POST /api/upload_weights               │
   │  {client_id, weights.h5, samples}       │
   │────────────────────────────────────────>│
   │                                         │
   │         200 OK                          │
   │<────────────────────────────────────────│
   │                                         │
   │                    [If threshold reached]
   │                    [Server aggregates]   │
   │                    [Updates global model]│
   │                                         │
```

## Configuration

### Server (`server/.env`)
```env
FL_SERVER_HOST=0.0.0.0          # Listen on all interfaces
FL_SERVER_PORT=5000
AGGREGATION_THRESHOLD=3          # Min clients before aggregation
BASE_MODEL_PATH=models/global_model.h5
```

### Client (`client/.env`)
```env
FL_SERVER_HOST=10.148.244.68    # Server IP address
FL_SERVER_PORT=5000
```

## Privacy Guarantees

1. **Data Locality**: Training data never leaves the client
2. **Weight Sharing**: Only model parameters are transmitted
3. **Aggregation**: Individual contributions are averaged (hard to reverse)
4. **Minimal Metadata**: Only client ID and sample counts are shared

## Technologies Used

- **Backend**: Python, Flask, TensorFlow/Keras
- **Model**: MobileNetV2 (transfer learning)
- **Aggregation**: Federated Averaging (FedAvg)
- **Mobile App**: Flutter + TFLite (inference only)
- **Communication**: REST API over HTTP

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
