# FEDERATED LEARNING SERVER - SETUP GUIDE

## 🖥️ Server Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/shashankbairy/DTL/server

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Server

```bash
# Copy example config
cp .env.example .env

# Edit if needed (default is fine for most cases)
# nano .env
```

### 3. Place Base Global Model

Copy your initial trained model to the server:

```bash
# Create models directory
mkdir -p models

# Copy your base model
cp ../mobilenetv2_oral_cancer_best.h5 models/global_model.h5
```

### 4. Start Server

```bash
python fl_server.py
```

You should see:
```
🚀 Starting FL Server on 0.0.0.0:5000
   Access locally at: http://localhost:5000
   Access from network at: http://<your-ip>:5000
```

### 5. Find Your Server IP

**On macOS:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**On Linux:**
```bash
hostname -I
```

**On Windows:**
```cmd
ipconfig
```

Share this IP with your clients!

---

## 📁 Folder Structure

After setup, you'll have:

```
server/
├── fl_server.py          # Main Flask server
├── fedavg.py             # FedAvg aggregation algorithm
├── requirements.txt      # Python dependencies
├── .env                  # Configuration
├── models/               # Global models
│   ├── global_model.h5   # Initial/current global model
│   └── global_model_round_N_TIMESTAMP.h5  # Saved rounds
├── uploads/              # Client uploads (temporary)
│   └── client_weights/
└── README.md             # This file
```

---

## 🔄 How It Works

### Client Upload Flow:
1. Client trains locally and uploads weights
2. Server stores weights in `uploads/`
3. When enough clients upload (threshold reached):
   - Server performs FedAvg aggregation
   - **NEW: Validates new model on test dataset**
   - **NEW: Only replaces global model if accuracy improves**
   - Creates new global model
   - Saves to `models/global_model_latest.h5`

### Client Download Flow:
1. Client requests global model
2. Server sends latest `global_model_latest.h5`
3. Client uses it for next training round

---

## 🔧 API Endpoints

### 1. Check Status
```bash
curl http://localhost:5000/api/status
```

### 2. Upload Weights (Client)
```bash
# Done automatically by client script
python ../client/fl_client_send.py --model <model_name>
```

### 3. Download Global Model (Client)
```bash
# Done automatically by client script
python ../client/fl_client_download.py
```

### 4. Manual Aggregation (Admin)
```bash
curl -X POST http://localhost:5000/api/trigger_aggregation
```

---

## ⚙️ Configuration

Edit `.env` to change:

- `AGGREGATION_THRESHOLD` - Min clients before aggregation (default: 2)
- `FL_SERVER_PORT` - Server port (default: 5000)
- `FLASK_DEBUG` - Debug mode (default: False)
- **`VALIDATION_DATASET_PATH`** - Path to test dataset for validation (default: ../new_data)
- **`VALIDATE_BEFORE_REPLACE`** - Enable accuracy validation before model replacement (default: True)

---

## 🐛 Troubleshooting

### Server won't start:
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
FL_SERVER_PORT=5001 python fl_server.py
```

### "Base model not found":
```bash
# Make sure you copied the model
ls -lh models/global_model.h5

# If not there, copy it:
cp ../mobilenetv2_oral_cancer_best.h5 models/global_model.h5
```

### Can't connect from other devices:
```bash
# Check firewall allows port 5000
# Make sure FL_SERVER_HOST=0.0.0.0 in .env
# Both devices must be on same network
```

### View server logs:
```bash
# Server logs show all activity:
# - Client uploads
# - Aggregation operations
# - Model saves
# Just run the server and watch output
```

---

## 🧪 Testing

### Test locally (same machine):
```bash
# Terminal 1: Start server
cd server
python fl_server.py

# Terminal 2: Upload from client
cd ../client
python fl_client_send.py --model <model_name> --dry-run
```

### Test from network:
```bash
# On client machine, edit client/.env:
FL_SERVER_HOST=192.168.1.100  # Your server IP

# Run client upload
python fl_client_send.py --model <model_name>
```

---

## 📊 Monitoring

View aggregation history:
```bash
# Check saved rounds
ls -lh models/

# View aggregation info
cat models/aggregation_round_1_info.json
```

---

## 🚀 Production Tips

1. **Use HTTPS** for production (not included in this basic setup)
2. **Add authentication** for client uploads
3. **Set proper AGGREGATION_THRESHOLD** based on your needs
4. **Backup models/** folder regularly
5. **Monitor disk space** in uploads/ folder

---

## 🔐 Security Notes

- This is a **basic implementation** for learning/testing
- **Do not expose to public internet** without proper security
- Add authentication/authorization for production use
- Consider using HTTPS/TLS encryption
- Validate all client uploads

---

## 📞 Need Help?

Check:
1. Server logs (terminal output)
2. Client logs (terminal output)
3. Network connectivity (ping server IP)
4. Firewall settings (allow port 5000)
5. File permissions (models/ and uploads/ folders)
