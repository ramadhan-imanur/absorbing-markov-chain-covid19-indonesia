# Pemodelan Rantai Markov Waktu Diskret untuk Dinamika Penularan COVID-19 di Indonesia

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Report: PDF](https://img.shields.io/badge/Report-PDF-red.svg)](report/laporan.pdf)

Studi kasus data science ini memodelkan dinamika penularan COVID-19 di Indonesia menggunakan pendekatan stokastik Rantai Markov Waktu Diskret berkeadaan penyerap (*Absorbing Markov Chain*). Parameter peluang transisi diestimasi langsung dari deret waktu empiris 929 hari observasi nasional (2 Maret 2020 hingga 16 September 2022) untuk menghitung rata-rata durasi sakit dan peluang kesembuhan akhir.

---

## Ringkasan Eksekutif & Temuan Kunci

Dengan membagi populasi ke dalam empat kompartemen (Rentan, Terinfeksi, Sembuh, Meninggal), model menghasilkan metrik epidemiologi berikut:

| Parameter Evaluasi | Nilai Model | Rujukan / Data Lapangan | Keterangan |
| :--- | :---: | :---: | :--- |
| **Rata-rata Durasi Sakit ($v_I$)** | **15,45 hari** | 10 s.d. 14 hari | Konsisten dengan durasi isolasi mandiri standar Kemenkes RI dan PDPI. |
| **Peluang Sembuh Pasien Aktif ($u_I^{(R)}$)** | **96,81%** | 96,7% s.d. 97,5% | Menggambarkan tingkat kesembuhan agregat nasional selama periode pandemi. |
| **Peluang Mortalitas Pasien Aktif ($u_I^{(D)}$)** | **3,19%** | 2,5% s.d. 3,3% | Sejalan dengan rentang Case Fatality Rate (CFR) kumulatif di Indonesia. |
| **Waktu Tunggu Penularan Rentan ($v_S$)** | **38.477 hari** | Laju per kapita $P_{SI} \approx 2{,}6 \times 10^{-5}$ | Rata-rata waktu transmisi individu rentan pada skala populasi 265 juta jiwa. |

Semua hasil perhitungan diselesaikan melalui dua metode independen: penurunan analitis *First Step Analysis* (FSA) dan dekomposisi matriks fundamental Kemeny-Snell ($N = (I - Q)^{-1}$). Kedua metode memberikan hasil identik hingga digit presisi terakhir.

---

## Visualisasi Data & Hasil Pemodelan

| Dinamika Deret Waktu SIRD (2020-2022) | Hasil Analisis Langkah Pertama ($u_i, v_i$) |
| :---: | :---: |
| ![Dinamika SIRD](report/figures/dinamika_sird_indonesia.png) | ![Hasil FSA](report/figures/hasil_fsa_sird.png) |

---

## Metodologi

Alur kerja analisis data terbagi ke dalam empat tahap:

1. **Pengumpulan & Pembersihan Data**: Mengolah 929 data harian nasional bersumber dari Satuan Tugas Penanganan COVID-19 dan Kementerian Kesehatan RI. Data mencakup kasus aktif, kesembuhan harian, dan kematian harian.
2. **Rekonstruksi Kompartemen SIRD**: Menentukan jumlah individu pada status Susceptible ($S$), Infected ($I$), Recovered ($R$), dan Death ($D$) dengan basis populasi $N = 265.185.520$ jiwa.
3. **Estimasi Matriks Peluang Transisi ($P$)**: Mengestimasi parameter probabilitas transisi satu langkah antar status dari data empiris:
   $$P = \begin{pmatrix} 0{,}999974 & 0{,}000026 & 0 & 0 \\ 0 & 0{,}935293 & 0{,}062643 & 0{,}002064 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
   Status $R$ dan $D$ berperan sebagai keadaan penyerap (*absorbing states*) karena diasumsikan proses berhenti ketika individu masuk ke salah satu status tersebut.
4. **First Step Analysis & Komputasi Simbolik**: Mempartisi matriks ke bentuk kanonik untuk memperoleh submatriks transien $Q$ dan submatriks serapan $R$. Matriks fundamental $N$ dan matriks peluang serapan $B = NR$ dihitung secara simbolik menggunakan SymPy dan numerik menggunakan NumPy/SciPy.

---

## Evaluasi & Catatan Kritis Asumsi Markov

Pendekatan rantai Markov memberikan estimasi yang cepat dan transparan, namun memiliki sejumlah batasan yang perlu dicatat:

- **Sifat nir-memori (*memoryless*)**: Model mengasumsikan peluang sembuh pada hari ke-14 sama dengan hari ke-1, mengabaikan efek akumulasi respon imun dan penurunan beban virus (*viral load*).
- **Homogenitas waktu**: Nilai parameter diasumsikan konstan sepanjang 929 hari, sedangkan di lapangan laju penularan berubah drastis saat varian Delta dan Omicron masuk, serta saat cakupan vaksinasi meningkat.
- **Rekomendasi pengembangan**: Pemodelan lanjutan dapat menerapkan Rantai Semi-Markov dengan waktu tunggu berdistribusi Weibull atau Gamma untuk menangkap durasi sakit yang lebih heterogen.

---

## Struktur Repositori

```text
absorbing-markov-chain-covid19-indonesia/
├── .gitignore                      # Filter cache Python, Jupyter, dan berkas pembantu LaTeX
├── LICENSE                         # Lisensi sumber terbuka MIT
├── README.md                       # Dokumentasi proyek dan laporan ringkas data science
├── code/
│   ├── requirements.txt            # Dependensi Python
│   ├── data/
│   │   ├── raw/                    # Dataset mentah 929 hari COVID-19 Indonesia
│   │   │   └── covid_19_indonesia_time_series_all.csv
│   │   └── processed/              # Data terstandardisasi dan berkas CSV matriks transisi
│   │       ├── data_sird_harian.csv
│   │       ├── matriks_P_sird.csv
│   │       ├── matriks_Q_sird.csv
│   │       ├── matriks_R_sird.csv
│   │       ├── matriks_N_sird.csv
│   │       └── matriks_B_sird.csv
│   ├── notebooks/
│   │   ├── 01_fenomena_dan_data.ipynb      # Pembersihan data & estimasi parameter transisi
│   │   └── 02_first_step_analysis.ipynb    # Komputasi numerik & simbolik FSA (SymPy)
│   ├── scripts/
│   │   ├── fsa_solver.py           # Engine analitis matriks fundamental
│   │   └── verify_sird_calculation.py # Skrip verifikasi aljabar independen
│   └── figures/                    # Visualisasi hasil analisis
│       ├── dinamika_sird_indonesia.png
│       └── hasil_fsa_sird.png
└── report/
    ├── logo_uns.png                # Aset logo universitas
    ├── references.bib              # Berkas bibliografi BibTeX
    ├── laporan.tex                 # Sumber naskah ilmiah format LaTeX
    ├── laporan.pdf                 # Laporan lengkap naskah ilmiah
    └── Kelompok 1.pdf              # Salinan deliverable tugas
```

---

## Cara Menjalankan Proyek

### 1. Kloning Repositori
```bash
git clone https://github.com/ramadhan-imanur/absorbing-markov-chain-covid19-indonesia.git
cd absorbing-markov-chain-covid19-indonesia
```

### 2. Instalasi Dependensi
Pastikan menggunakan Python 3.10 atau versi yang lebih baru:
```bash
pip install -r code/requirements.txt
```

### 3. Eksekusi Verifikasi Perhitungan
Jalankan skrip verifikasi untuk memeriksa matriks fundamental dan hasil FSA secara langsung di terminal:
```bash
python3 code/scripts/verify_sird_calculation.py
```

### 4. Menjalankan Jupyter Notebook
Untuk menelusuri proses eksplorasi data dan komputasi tahap demi tahap:
```bash
jupyter notebook code/notebooks/
```

---

## Tech Stack & Pustaka

- **Bahasa**: Python 3.10+
- **Manipulasi Data**: Pandas, NumPy
- **Komputasi Simbolik & Numerik**: SymPy, SciPy
- **Visualisasi**: Matplotlib, Seaborn
- **Laporan Ilmiah**: LaTeX (TeX Live)

---

## Tim Pengembang & Konteks

Proyek ini disusun sebagai bagian dari studi proses stokastik terapan pada Program Studi S1 Matematika, Fakultas MIPA, Universitas Sebelas Maret (UNS).

**Kelompok 1:**
- Fadhila Hardi Ningrum (M0124004)
- Karunia Febyayu Puspitaningtyas (M0124010)
- Ramadhan Imanur Rochim (M0124015)
- Achika Vigo Azhyra (M0125001)
- Fawwaz Absyar Rifai (M0125044)

**Dosen Pengampu:** Ade Susanti, S.Si., M.Si.
