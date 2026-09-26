# 🔍 AI-Powered Missing Person Recovery System (PathBack)

An end-to-end intelligent recovery platform designed to assist law enforcement agencies, investigative officers, and citizens in identifying, locating, and tracking missing persons using state-of-the-art computer vision, FAISS vector indexing, and deep metric face recognition.

---

## 🚀 Key Features

### 🖥️ Desktop / Investigation Portal (`Home.py`)
- **🔐 Secure Authentication:** Role-based access control for administrative personnel and field officers.
- **📝 Case Registration:** Register missing individuals with personal details, age at disappearance, demographic attributes, last known GPS coordinates, high-resolution photographs, and automatic 128-D SFace feature extraction.
- **📂 Case Management:** Real-time search, filter, status tracking (Active / Solved / Under Review), and case history logs.
- **🎯 FAISS AI Vector Matching Engine:** High-throughput sub-millisecond similarity lookups comparing missing person database records against public sightings using L2-normalized deep facial embeddings (FAISS `IndexFlatIP` & `IndexHNSWFlat` with graceful NumPy fallback).
- **📹 Real-Time CCTV & Video Surveillance Scanner:** High-throughput surveillance pipeline featuring:
  - **MOG2 Motion-Detection Gating:** Bypasses neural inference on static scenes for 3–5x FPS acceleration.
  - **IoU Multi-Face Tracking (SORT/ByteTrack):** Persistent track IDs across frames.
  - **Alert Deduplication:** Eliminates duplicate alert spam for the same target in a video feed.
  - **Live Telemetry & Alert Feed:** Real-time FPS, frame latency (ms), motion area, and timestamped thumbnail alert feed.

### 📱 Mobile / Public Sighting Portal (`mobile_app.py`)
- **📸 Public Sighting Uploads:** Allows citizens, volunteers, and patrol officers to upload photos or snap live camera pictures of potential sightings.
- **📍 Geolocation & Metadata:** Records sighting timestamp, GPS coordinates, location description, finder's contact details, and device metadata.
- **⚡ Automated Feature Extraction:** Instant background facial landmark detection and strictly L2-normalized feature vector generation upon upload.

---

## 🧠 AI Models & Architecture

| Component | Technology / Model | Purpose |
| :--- | :--- | :--- |
| **Face Detection** | YuNet (`yunet.onnx`) | High-speed, robust face bounding box & landmark detection with CUDA/CPU caching |
| **Face Recognition** | SFace (`sface.onnx`) | Deep metric 128-D L2-normalized cosine similarity face embedding extraction |
| **Vector Index Engine** | FAISS CPU (`IndexFlatIP` / `IndexHNSWFlat`) | Sub-millisecond maximum inner product similarity lookups across cases |
| **Index Fallback** | NumPy Vectorized Matrix Search | Zero-downtime cosine similarity fallback for $< 10$ records or systems without FAISS |
| **Motion Gating** | OpenCV MOG2 Subtractor | Frame-level motion detection to bypass deep inference on static video scenes |
| **Face Tracking** | IoU / SORT Tracker | Multi-object tracking and alert deduplication across surveillance video frames |
| **Geospatial Engine** | Haversine Formula | Great-circle distance calculations & radius-based candidate filtering |
| **Database** | SQLModel / SQLite (`sqlite_database.db`) | Relational persistence with automated schema migrations |

---

## 📁 Project Structure

```plaintext
PathBack/
├── Home.py                    # Main Admin & Desktop Investigation Portal
├── mobile_app.py              # Public / Field Citizen Sighting Web App
├── login_config.yml           # Authentication credentials & session settings
├── sqlite_database.db         # SQLite database storing cases, sightings & vectors
├── requirements.txt           # Python dependencies list
├── Instructions.txt           # Step-by-step setup & execution notes
├── run_desktop_app.ps1        # PowerShell launcher for Desktop Portal (:8501)
├── run_mobile_app.ps1         # PowerShell launcher for Mobile Portal (:8502)
├── models/                    # ONNX Deep Learning weight files & indices
│   ├── yunet.onnx             # YuNet Face Detector
│   ├── sface.onnx             # SFace Deep Metric Recognizer
│   ├── biometric_index.faiss  # Persisted FAISS vector index binary
│   └── biometric_index_meta.pkl # Vector index metadata mapping
└── pages/                     # Streamlit multi-page module hierarchy
    ├── 1_Register_New_Case.py # New case registration & facial scanning
    ├── 2_All_Cases.py         # Case management & status resolution
    ├── 3_Match_Cases.py       # AI deep face matching portal
    ├── 4_Help.py              # User guide & documentation
    ├── 5_CCTV_Video_Scanner.py # Real-time CCTV surveillance scanner
    └── helper/                # Core AI, DB queries, utilities & training scripts
        ├── data_models.py     # SQLModel entities, schemas & confidence tiers
        ├── db_queries.py      # Database operations & Haversine geospatial queries
        ├── match_algo.py      # FAISS similarity search & multi-tier matching
        ├── model_cache.py     # Streamlit session resource caching & CUDA probe
        ├── train_model.py     # FAISS vector index training & persistence
        └── utils.py           # Face extraction, MOG2 gating & IoU tracking
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python **3.10** or higher
- PowerShell / Terminal

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: If running PowerShell scripts is restricted, run `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser`)*

---

## 💻 Running the Applications

### 1. Launch the Desktop / Admin Investigation Portal
```bash
python -m streamlit run .\Home.py
```
*Or execute:* `.\run_desktop_app.ps1`  
*Access URL:* **[http://localhost:8501](http://localhost:8501)**

#### 🔑 Default Admin Credentials:
- **Username:** `admin`
- **Password:** `admin123`

---

### 2. Launch the Mobile / Citizen Sighting Portal
```bash
python -m streamlit run .\mobile_app.py --server.port 8502
```
*Or execute:* `.\run_mobile_app.ps1`  
*Access URL:* **[http://localhost:8502](http://localhost:8502)**

---

## 🔄 Refreshing the Biometric Vector Index

When new missing person profiles are registered, the FAISS vector index automatically updates. To manually re-index or build the offline index:
```bash
python pages/helper/train_model.py
```

---

## 📄 License & Attribution
Developed for rapid missing person tracking and humanitarian recovery using Computer Vision and Machine Learning.
