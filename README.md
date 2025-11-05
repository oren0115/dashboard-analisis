# 📊 Sentiment Analysis Project - Analisis Opini Publik Menggunakan Machine Learning

Project ini adalah implementasi lengkap untuk analisis sentimen teks menggunakan machine learning. Project ini mencakup seluruh pipeline dari data understanding hingga deployment dengan API dan dashboard interaktif.

## 📋 Fitur

- ✅ **Data Understanding**: Eksplorasi data dengan visualisasi lengkap
- ✅ **Data Preprocessing**: Pembersihan teks, tokenization, stopword removal, dan stemming
- ✅ **Feature Extraction**: TF-IDF Vectorization
- ✅ **Model Training**: Multiple models (Logistic Regression, SVM, Random Forest)
- ✅ **Model Evaluation**: Metrik lengkap (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- ✅ **API Deployment**: Flask API untuk prediksi sentimen
- ✅ **Dashboard Interaktif**: Streamlit dashboard dengan visualisasi

## 🚀 Instalasi

### 1. Clone atau Download Project

```bash
# Pastikan Anda berada di directory project
cd "path/to/project"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pastikan Dataset Ada

Pastikan file berikut ada di directory project:

- `Sentiment1.csv`
- `Train1.csv`

## 📝 Cara Menggunakan

### Step 1: Data Understanding

Jalankan script untuk analisis data awal:

```bash
python data_understanding.py
```

Script ini akan:

- Menggabungkan dataset
- Menganalisis distribusi sentimen
- Membuat visualisasi (disimpan di folder `output/`)
- Menyimpan dataset gabungan sebagai `dataset_combined.csv`

### Step 2: Data Preprocessing

Jalankan script untuk preprocessing data:

```bash
python preprocessing.py
```

Script ini akan:

- Membersihkan teks (remove URL, mention, hashtag, dll)
- Case folding
- Remove stopwords (menggunakan Sastrawi)
- Stemming (menggunakan Sastrawi)
- Encoding labels (Positive=1, Neutral=0, Negative=-1)
- Menyimpan dataset preprocessed sebagai `dataset_preprocessed.csv`
- Menyimpan preprocessor sebagai `preprocessor.pkl`

### Step 3: Model Training

Jalankan script untuk training model:

```bash
python train_model.py
```

Script ini akan:

- Split data menjadi train dan test
- Extract features menggunakan TF-IDF
- Train 3 model: Logistic Regression, SVM, Random Forest
- Evaluasi semua model
- Membuat visualisasi perbandingan model
- Menyimpan model terbaik sebagai `best_model.pkl`
- Menyimpan semua model di folder `models/`

### Step 4: API Server (Optional)

Jalankan API server:

```bash
python app.py
```

API akan berjalan di `http://localhost:5000`

**Contoh penggunaan API:**

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya sangat senang hari ini!"}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Teks 1", "Teks 2", "Teks 3"]}'
```

**Endpoint yang tersedia:**

- `GET /` - Home
- `GET /health` - Health check
- `POST /predict` - Prediksi sentimen dari satu teks
- `POST /predict/batch` - Prediksi sentimen dari multiple teks

### Step 5: Dashboard (Optional)

Jalankan dashboard interaktif:

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka di browser dengan URL `http://localhost:8501`

**Fitur Dashboard:**

- 📈 Analisis Data: Statistik dan visualisasi dataset
- 🤖 Prediksi Sentimen: Prediksi real-time dari teks baru
- 📊 Visualisasi: Word cloud dan grafik interaktif

## 📁 Struktur Project

```
project/
│
├── Sentiment1.csv              # Dataset 1
├── Train1.csv                   # Dataset 2
│
├── data_understanding.py        # Script untuk EDA
├── preprocessing.py             # Script untuk preprocessing
├── train_model.py              # Script untuk training model
├── app.py                      # Flask API server
├── dashboard.py                # Streamlit dashboard
│
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
│
├── dataset_combined.csv        # Dataset gabungan (dihasilkan)
├── dataset_preprocessed.csv    # Dataset setelah preprocessing (dihasilkan)
│
├── preprocessor.pkl            # Preprocessor (dihasilkan)
├── best_model.pkl             # Model terbaik (dihasilkan)
│
├── models/                     # Folder untuk semua model
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── random_forest.pkl
│
└── output/                     # Folder untuk visualisasi
    ├── sentiment_distribution.png
    ├── text_length_analysis.png
    ├── temporal_trends.png
    ├── wordclouds.png
    ├── confusion_matrix_*.png
    └── model_comparison.png
```

## 🔧 Model yang Digunakan

1. **Logistic Regression**: Model linear yang cepat dan interpretable
2. **SVM (Support Vector Machine)**: Bagus untuk data dengan banyak fitur
3. **Random Forest**: Ensemble method yang robust

Model terbaik dipilih berdasarkan F1-Score tertinggi.

## 📊 Metrik Evaluasi

Setiap model dievaluasi menggunakan:

- **Accuracy**: Akurasi keseluruhan
- **Precision**: Presisi per kelas (weighted average)
- **Recall**: Recall per kelas (weighted average)
- **F1-Score**: F1-score per kelas (weighted average)
- **Confusion Matrix**: Matrix untuk melihat prediksi per kelas

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Matplotlib & Seaborn**: Visualization
- **WordCloud**: Word cloud generation
- **Sastrawi**: Indonesian language processing
- **Flask**: API framework
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

## 📝 Catatan

- Dataset yang digunakan adalah teks bahasa Indonesia dari media sosial
- Preprocessing menggunakan library Sastrawi untuk bahasa Indonesia
- Model menggunakan TF-IDF untuk feature extraction
- Pastikan semua file model dan preprocessor sudah ada sebelum menjalankan API atau dashboard

## 🤝 Kontribusi

Silakan fork dan buat pull request untuk perbaikan atau fitur baru!

## 📄 Lisensi

Project ini untuk keperluan edukasi dan pembelajaran.

---

**Dibuat dengan ❤️ untuk analisis sentimen teks bahasa Indonesia**
