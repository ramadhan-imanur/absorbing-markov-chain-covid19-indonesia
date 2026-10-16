"""
Unit Tests untuk Pipeline Analisis Penyerapan Stokastik SIRD COVID-19 Indonesia
Memverifikasi integritas data, aksioma stokastik, matriks fundamental, dan metrik klinis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Menentukan path dinamis
TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
DATA_RAW = CODE_DIR / "data" / "raw"
DATA_PROCESSED = CODE_DIR / "data" / "processed"


def test_raw_dataset_exists_and_integrity():
    """Memverifikasi berkas data mentah tersedia dan memiliki observasi lengkap."""
    raw_path = DATA_RAW / "covid_19_indonesia_time_series_all.csv"
    assert raw_path.exists(), f"Berkas data mentah tidak ditemukan di {raw_path}"
    
    df = pd.read_csv(raw_path)
    assert not df.empty, "Dataset mentah kosong"
    assert "Location Level" in df.columns
    assert "Date" in df.columns
    
    df_nat = df[df["Location Level"] == "Country"]
    assert len(df_nat) >= 900, f"Jumlah observasi nasional ({len(df_nat)}) kurang dari target 900 hari"


def test_processed_sird_dataset():
    """Memverifikasi data olahan SIRD harian memiliki kolom valid dan urutan waktu benar."""
    proc_path = DATA_PROCESSED / "data_sird_harian.csv"
    assert proc_path.exists(), "Berkas data_sird_harian.csv belum dibuat"
    
    df = pd.read_csv(proc_path)
    expected_cols = ["Date", "S", "I", "R", "D", "p_SI", "p_IR", "p_ID"]
    for col in expected_cols:
        assert col in df.columns, f"Kolom {col} hilang dari data_sird_harian.csv"
    
    assert len(df) == 929, f"Jumlah hari observasi olahan {len(df)} tidak tepat 929 hari"
    assert (df["S"] > 0).all(), "Populasi rentan S harus selalu positif"
    assert (df["I"] >= 0).all(), "Populasi terinfeksi I tidak boleh negatif"


def test_stochastic_matrix_axioms():
    """Memverifikasi aksioma non-negativitas dan penjumlahan baris = 1 pada matriks transisi P."""
    p_path = DATA_PROCESSED / "matriks_P_sird.csv"
    assert p_path.exists(), "matriks_P_sird.csv tidak ditemukan"
    
    P = np.loadtxt(p_path, delimiter=",")
    assert P.shape == (4, 4), f"Dimensi matriks transisi harus 4x4, ditemukan {P.shape}"
    
    # Aksioma 1: Non-negatif
    assert np.all(P >= 0.0), "Terdapat entri probabilitas bernilai negatif"
    
    # Aksioma 2: Jumlah baris = 1
    row_sums = np.sum(P, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg="Jumlah probabilitas tiap baris harus sama dengan 1")


def test_absorbing_states_classification():
    """Memverifikasi bahwa status R (indeks 2) dan D (indeks 3) adalah absorbing states sejati."""
    p_path = DATA_PROCESSED / "matriks_P_sird.csv"
    P = np.loadtxt(p_path, delimiter=",")
    
    # State R: P[2, 2] == 1 dan sum != 2 adalah 0
    assert P[2, 2] == 1.0, "State Recovered (R) harus bernilai 1 di diagonal utama"
    assert np.isclose(np.sum(P[2, :]) - P[2, 2], 0.0), "State Recovered (R) tidak boleh memiliki transisi keluar"
    
    # State D: P[3, 3] == 1 dan sum != 3 adalah 0
    assert P[3, 3] == 1.0, "State Death (D) harus bernilai 1 di diagonal utama"
    assert np.isclose(np.sum(P[3, :]) - P[3, 3], 0.0), "State Death (D) tidak boleh memiliki transisi keluar"


def test_fundamental_matrix_and_absorption_metrics():
    """Memverifikasi Teori Kemeny-Snell N = (I - Q)^(-1) dan B = N * R."""
    q_path = DATA_PROCESSED / "matriks_Q_sird.csv"
    r_path = DATA_PROCESSED / "matriks_R_sird.csv"
    n_path = DATA_PROCESSED / "matriks_N_sird.csv"
    b_path = DATA_PROCESSED / "matriks_B_sird.csv"
    
    Q = np.loadtxt(q_path, delimiter=",")
    R = np.loadtxt(r_path, delimiter=",")
    N = np.loadtxt(n_path, delimiter=",")
    B = np.loadtxt(b_path, delimiter=",")
    
    I2 = np.eye(2)
    N_calc = np.linalg.inv(I2 - Q)
    np.testing.assert_allclose(N, N_calc, atol=1e-3, err_msg="Matriks N tersimpan berbeda dari (I - Q)^(-1)")
    
    B_calc = N @ R
    np.testing.assert_allclose(B, B_calc, atol=1e-3, err_msg="Matriks B tersimpan berbeda dari N * R")
    
    # Baris B harus berjumlah 1 (total probability of absorption)
    np.testing.assert_allclose(np.sum(B, axis=1), 1.0, atol=1e-4)


def test_clinical_epidemiological_sanity():
    """Memverifikasi parameter output memenuhi rentang kewajaran klinis COVID-19 Indonesia."""
    n_path = DATA_PROCESSED / "matriks_N_sird.csv"
    b_path = DATA_PROCESSED / "matriks_B_sird.csv"
    N = np.loadtxt(n_path, delimiter=",")
    B = np.loadtxt(b_path, delimiter=",")
    
    # v_I = N[1, 1] (durasi sakit terinfeksi)
    v_I = N[1, 1]
    assert 12.0 <= v_I <= 18.0, f"Rata-rata durasi sakit v_I={v_I:.2f} hari berada di luar batas wajar klinis (12-18 hari)"
    
    # u_I^(R) = B[1, 0] (peluang sembuh)
    u_I_R = B[1, 0]
    assert 0.95 <= u_I_R <= 0.98, f"Peluang kesembuhan u_I(R)={u_I_R*100:.2f}% berada di luar batas acuan Satgas (95-98%)"
    
    # u_I^(D) = B[1, 1] (peluang kematian CFR)
    u_I_D = B[1, 1]
    assert 0.02 <= u_I_D <= 0.04, f"Peluang kematian u_I(D)={u_I_D*100:.2f}% berada di luar batas CFR empiris (2-4%)"
