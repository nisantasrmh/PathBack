# 🔍 AI-Powered Missing Person Recovery System (PathBack)

An end-to-end intelligent recovery platform designed to assist law enforcement agencies, investigative officers, and citizens in identifying, locating, and tracking missing persons using state-of-the-art computer vision and deep metric face recognition.

---

## 🚀 Key Features

### 🖥️ Desktop / Investigation Portal (`Home.py`)
- **🔐 Secure Authentication:** Role-based access control for administrative personnel and field officers.
- **📝 Case Registration:** Register missing individuals with personal details, last known coordinates, high-resolution photographs, and automatic AI facial feature extraction.
- **📂 Case Management:** Real-time search, filter, status tracking (Active / Solved), and case history logs.
- **🎯 AI Matching Engine:** Compares missing person database records against public sightings using deep facial feature vectors and 468-point mesh geometry.
- **🎥 CCTV & Video Footage Scanner:** Upload surveillance recordings and CCTV footage to automatically detect, crop, extract, and match faces frame-by-frame against active cases with timestamped match alerts.

### 📱 Mobile / Public Sighting Portal (`mobile_app.py`)
- **📸 Public Sighting Uploads:** Allows citizens, volunteers, and patrol officers to upload photos of potential sightings.
- **📍 Geolocation & Metadata:** Records sighting timestamp, location description, finder's contact details, and optional notes.
- **⚡ Automated Feature Extraction:** Instant background facial landmark detection and feature vector generation upon upload.

---

## 🧠 AI Models & Architecture

| Component | Technology / Model | Purpose |
| :--- | :--- | :--- |
| **Face Detection** | YuNet (`face_detection_yunet_2023mar.onnx`) | High-speed, robust face bounding box & landmark detection |
| **Face Recognition** | SFace (`sface.onnx`) | Deep metric 128-D cosine similarity face embedding extraction |
| **Facial Geometry** | MediaPipe FaceMesh | 468 3D facial landmark mesh construction |
| **Case Classification** | Scikit-learn KNN / BallTree | Fast nearest-neighbor case matching over indexed feature vectors |
| **Database** | SQLModel / SQLite (`sqlite_database.db`) | Relational persistence of cases, sightings, and embeddings |

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
├── models/                    # ONNX Deep Learning weight files
│   ├── face_detection_yunet_2023mar.onnx
│   └── sface.onnx
└── pages/                     # Streamlit multi-page module hierarchy
    ├── 1_Register_New_Case.py
    ├── 2_All_Cases.py
    ├── 3_Match_Cases.py
    ├── 4_Help.py
    ├── 5_CCTV_Video_Scanner.py
    └── helper/                # Core AI, DB queries, utilities & training scripts
        ├── data_models.py
        ├── db_queries.py
        ├── match_algo.py
        ├── model_cache.py
        ├── train_model.py
        └── utils.py
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

## 🔄 Refreshing the AI Classifier

When new missing person profiles are registered, the KNN classifier can be manually refreshed at any time:
```bash
python pages/helper/train_model.py
```

---

## 📄 License & Attribution
Developed for rapid missing person tracking and humanitarian recovery using Computer Vision and Machine Learning.
