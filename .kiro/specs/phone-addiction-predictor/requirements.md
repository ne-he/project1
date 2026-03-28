# Requirements Document

## Introduction

Fitur ini mendeploy model Machine Learning CatBoost (RMSE test 0.362, R² test 0.947) yang telah dilatih pada dataset `Phone_Addiction.csv` ke dalam sebuah aplikasi web interaktif menggunakan Streamlit. Aplikasi menerima input perilaku penggunaan smartphone dari pengguna, menjalankan pipeline preprocessing yang identik dengan notebook pelatihan, lalu menampilkan prediksi tingkat kecanduan smartphone pada skala 1–10 beserta interpretasinya.

## Glossary

- **App**: Aplikasi Streamlit (`app.py`) yang menjadi antarmuka pengguna.
- **Preprocessor**: Modul `src/preprocessing.py` yang menjalankan seluruh langkah transformasi data.
- **Model**: Modul `src/model.py` yang memuat model CatBoost dan menjalankan inferensi.
- **Pipeline**: Urutan transformasi data yang harus identik antara training dan inferensi: cleaning → imputation → OHE → feature engineering → log transform → scaling.
- **Artifact**: File tersimpan hasil training: `models/catboost_model.cbm`, `models/scaler.pkl`, `models/encoders.pkl`.
- **Addiction_Level**: Target prediksi, nilai kontinu pada skala 1.0–10.0.
- **OHE**: OneHotEncoder untuk kolom `Gender` dan `Phone_Usage_Purpose`.
- **Scaler**: StandardScaler yang di-fit pada data training.
- **train_and_save.py**: Script Python untuk melatih ulang model dan menyimpan semua artifact.

---

## Requirements

### Requirement 1: Setup Struktur Proyek

**User Story:** Sebagai developer, saya ingin struktur proyek yang terorganisir, sehingga kode mudah dipelihara dan di-deploy.

#### Acceptance Criteria

1. THE App SHALL memiliki struktur direktori: `phone-addiction-predictor/` dengan subfolder `src/` dan `models/`.
2. THE App SHALL menyertakan file `app.py`, `src/preprocessing.py`, `src/model.py`, `requirements.txt`, `README.md`, dan `.gitignore` pada direktori root proyek.
3. THE App SHALL menyertakan file `train_and_save.py` pada direktori root untuk keperluan training ulang dan penyimpanan artifact.
4. WHEN folder `models/` tidak berisi artifact, THEN THE App SHALL menampilkan pesan error yang informatif kepada pengguna.

---

### Requirement 2: Preprocessing Pipeline yang Identik dengan Notebook

**User Story:** Sebagai data scientist, saya ingin pipeline preprocessing di aplikasi identik dengan notebook pelatihan, sehingga prediksi model valid dan konsisten.

#### Acceptance Criteria

1. THE Preprocessor SHALL membersihkan kolom `Sleep_Hours` dengan cara: konversi ke string, strip karakter kutip (`"`), lalu konversi ke float.
2. THE Preprocessor SHALL menormalisasi kolom `Gender` dengan cara: strip whitespace, lowercase, capitalize, dan mengganti nilai `"femle"` menjadi `"Female"`.
3. THE Preprocessor SHALL mengganti nilai `"Unknown"` pada kolom `Phone_Usage_Purpose` dengan `NaN` sebelum imputation.
4. THE Preprocessor SHALL melakukan imputation nilai numerik yang hilang menggunakan median dari data training.
5. THE Preprocessor SHALL melakukan imputation nilai kategorikal yang hilang menggunakan modus dari data training.
6. THE Preprocessor SHALL menerapkan OneHotEncoder dengan parameter `drop=["Other", "Other"]`, `sparse_output=False`, `handle_unknown="ignore"` pada kolom `["Gender", "Phone_Usage_Purpose"]`.
7. THE Preprocessor SHALL membuat 11 fitur turunan (engineered features) sesuai formula notebook:
   - `usage_zero_flag`, `checks_per_hour`, `apps_per_hour`, `screen_before_bed_ratio`
   - `usage_to_sleep_ratio`, `late_screen_ratio`, `social_to_solo_ratio`
   - `resilience_gap`, `high_gaming_x_sleep`, `social_media_x_anxiety`
8. THE Preprocessor SHALL menerapkan transformasi `np.log1p` pada kolom skewed: `["Age", "checks_per_hour", "apps_per_hour", "screen_before_bed_ratio", "usage_to_sleep_ratio", "social_to_solo_ratio", "social_media_x_anxiety"]`.
9. THE Preprocessor SHALL menerapkan StandardScaler yang telah di-fit pada data training untuk mentransformasi input inferensi.
10. WHEN input inferensi diterima, THE Preprocessor SHALL menjalankan seluruh langkah 1–9 secara berurutan dalam satu fungsi `preprocess_pipeline()`.
11. FOR ALL input valid yang diproses oleh Preprocessor, urutan kolom output SHALL identik dengan urutan kolom saat training.

---

### Requirement 3: Penyimpanan dan Pemuatan Artifact Model

**User Story:** Sebagai developer, saya ingin artifact model disimpan dan dimuat dengan benar, sehingga aplikasi dapat berjalan tanpa perlu melatih ulang setiap kali dijalankan.

#### Acceptance Criteria

1. THE train_and_save.py SHALL melatih model CatBoost menggunakan dataset `Phone_Addiction.csv` dengan parameter terbaik dari notebook.
2. THE train_and_save.py SHALL menyimpan model CatBoost ke `models/catboost_model.cbm` menggunakan metode native CatBoost.
3. THE train_and_save.py SHALL menyimpan objek `StandardScaler` yang telah di-fit ke `models/scaler.pkl` menggunakan `joblib`.
4. THE train_and_save.py SHALL menyimpan objek `OneHotEncoder` yang telah di-fit ke `models/encoders.pkl` menggunakan `joblib`.
5. THE Model SHALL memuat `catboost_model.cbm`, `scaler.pkl`, dan `encoders.pkl` dari folder `models/` saat aplikasi diinisialisasi.
6. WHEN file artifact tidak ditemukan, THEN THE Model SHALL melempar `FileNotFoundError` dengan pesan yang menyebutkan nama file yang hilang.
7. THE Model SHALL menggunakan `@st.cache_resource` untuk meng-cache pemuatan artifact agar tidak dimuat ulang setiap prediksi.

---

### Requirement 4: Antarmuka Input Pengguna

**User Story:** Sebagai pengguna, saya ingin mengisi form input yang jelas dan terstruktur, sehingga saya dapat memasukkan data perilaku smartphone saya dengan mudah.

#### Acceptance Criteria

1. THE App SHALL menampilkan judul aplikasi dan deskripsi singkat pada halaman utama.
2. THE App SHALL menyediakan input untuk 19 fitur berikut:
   - `Age` (number input, min 1, max 100)
   - `Gender` (selectbox: Male, Female, Other)
   - `Daily_Usage_Hours` (number input, min 0.0, max 24.0)
   - `Sleep_Hours` (number input, min 0.0, max 24.0)
   - `Interllectual_Performance` (number input, min 0, max 100)
   - `Social_Interactions` (number input, min 0, max 20)
   - `Exercise_Hours` (number input, min 0.0, max 24.0)
   - `Screen_Time_Before_Bed` (number input, min 0.0, max 24.0)
   - `Phone_Checks_Per_Day` (number input, min 0, max 500)
   - `Anxiety_Level` (number input, min 0, max 10)
   - `Depression_Level` (number input, min 0, max 10)
   - `Self_Esteem` (number input, min 0, max 10)
   - `Apps_Used_Daily` (number input, min 0, max 100)
   - `Time_on_Social_Media` (number input, min 0.0, max 24.0)
   - `Time_on_Gaming` (number input, min 0.0, max 24.0)
   - `Time_on_Education` (number input, min 0.0, max 24.0)
   - `Phone_Usage_Purpose` (selectbox: Browsing, Education, Gaming, Social Media, Other)
   - `Family_Communication` (number input, min 0, max 20)
   - `Weekend_Usage_Hours` (number input, min 0.0, max 24.0)
3. THE App SHALL mengelompokkan input ke dalam beberapa seksi yang logis (misalnya: Informasi Dasar, Penggunaan Smartphone, Kesehatan Mental).
4. THE App SHALL menyediakan tombol "Prediksi" untuk memicu proses prediksi.
5. WHEN pengguna mengklik tombol "Prediksi", THE App SHALL menjalankan preprocessing dan inferensi model.

---

### Requirement 5: Prediksi dan Tampilan Hasil

**User Story:** Sebagai pengguna, saya ingin melihat hasil prediksi tingkat kecanduan beserta interpretasinya, sehingga saya dapat memahami kondisi saya.

#### Acceptance Criteria

1. THE Model SHALL menerima DataFrame satu baris hasil preprocessing dan mengembalikan nilai prediksi `Addiction_Level` bertipe float.
2. THE App SHALL menampilkan nilai prediksi dibulatkan ke dua desimal pada skala 1.0–10.0.
3. THE App SHALL menampilkan interpretasi kategorikal berdasarkan nilai prediksi:
   - Nilai < 4.0: "Rendah – Penggunaan smartphone Anda tergolong sehat."
   - Nilai 4.0–6.9: "Sedang – Perhatikan pola penggunaan smartphone Anda."
   - Nilai ≥ 7.0: "Tinggi – Disarankan untuk mengurangi penggunaan smartphone."
4. THE App SHALL menampilkan indikator visual (misalnya progress bar atau warna) yang mencerminkan tingkat kecanduan.
5. IF terjadi error saat preprocessing atau inferensi, THEN THE App SHALL menampilkan pesan error yang informatif tanpa crash.

---

### Requirement 6: Dependencies dan Konfigurasi Deployment

**User Story:** Sebagai developer, saya ingin semua dependency terdokumentasi dengan versi yang tepat, sehingga aplikasi dapat direproduksi di lingkungan lain.

#### Acceptance Criteria

1. THE App SHALL menyertakan `requirements.txt` yang mencantumkan semua dependency dengan versi yang kompatibel: `streamlit`, `catboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`.
2. THE App SHALL menyertakan `.gitignore` yang mengecualikan: folder `__pycache__/`, file `*.pyc`, folder `.venv/`, dan folder `models/*.cbm` (opsional, jika model tidak di-commit).
3. THE App SHALL menyertakan `README.md` dengan instruksi: instalasi dependency, cara menjalankan `train_and_save.py`, dan cara menjalankan aplikasi Streamlit.
4. WHEN `requirements.txt` diinstal pada Python 3.9+, THE App SHALL dapat dijalankan tanpa error dependency.
