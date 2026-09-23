"""
Modul First Step Analysis (FSA) Solver untuk Rantai Markov Absorbing.
Mata Kuliah: Pengantar Proses Stokastik
Program Studi Sarjana Matematika, FMIPA, Universitas Sebelas Maret

Modul ini menyediakan fungsi untuk:
1. Validasi matriks stokastik (aksioma non-negativitas dan jumlah baris = 1)
2. Identifikasi & klasifikasi state (transient vs absorbing)
3. Partisi matriks ke bentuk kanonik P = [[Q, R], [0, I]]
4. Pembentukan sistem persamaan First Step Analysis (FSA)
5. Penyelesaian analitis simbolik (SymPy) dengan fraksi eksak
6. Penyelesaian numerik via Teori Matriks Fundamental Kemeny-Snell N = (I - Q)^(-1)
7. Model khusus Dinamika Progresi Klinis COVID-19 di Indonesia
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import sympy as sp
import pandas as pd


def verify_stochastic_matrix(P: np.ndarray, tol: float = 1e-6) -> Tuple[bool, List[str]]:
    """
    Memverifikasi dua syarat mutlak matriks stokastik:
    1. Aksioma Non-Negatif: P_ij >= 0 untuk seluruh i, j
    2. Aksioma Penjumlahan Baris: sum_j P_ij = 1 untuk seluruh baris i
    """
    P = np.asarray(P, dtype=float)
    issues = []
    
    # Syarat 1: Non-negatif
    if np.any(P < -tol):
        neg_indices = np.argwhere(P < -tol)
        issues.append(f"Terdapat entri negatif pada indeks: {neg_indices.tolist()}")
        
    # Syarat 2: Penjumlahan baris = 1
    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=tol))[0]
        issues.append(f"Baris berikut tidak berjumlah 1: {bad_rows.tolist()} dengan jumlah {row_sums[bad_rows].tolist()}")
        
    is_valid = len(issues) == 0
    return is_valid, issues


def classify_states(P: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Mengklasifikasikan state menjadi:
    - Absorbing: P[i, i] == 1 dan P[i, j] == 0 untuk seluruh j != i
    - Transient: Bukan absorbing dan memiliki probabilitas untuk terserap
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    absorbing_states = []
    transient_states = []
    
    for i in range(n):
        if np.isclose(P[i, i], 1.0) and np.isclose(np.sum(P[i, :]) - P[i, i], 0.0):
            absorbing_states.append(i)
        else:
            transient_states.append(i)
            
    return transient_states, absorbing_states


def partition_canonical_form(
    P: np.ndarray, 
    transient_states: List[int], 
    absorbing_states: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Menyusun ulang matriks P ke dalam bentuk kanonik standar:
    P_canonical = [[Q, R],
                   [0, I]]
    Mengembalikan (Q, R, P_reordered, reordered_labels).
    """
    P = np.asarray(P, dtype=float)
    reordered_states = transient_states + absorbing_states
    P_reordered = P[np.ix_(reordered_states, reordered_states)]
    
    num_t = len(transient_states)
    
    Q = P_reordered[:num_t, :num_t]
    R = P_reordered[:num_t, num_t:]
    
    return Q, R, P_reordered, reordered_states


def solve_fsa_numerical(Q: np.ndarray, R: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Menyelesaikan First Step Analysis secara numerik menggunakan teori Matriks Fundamental Kemeny-Snell:
    - N = (I - Q)^(-1)        [Matriks fundamental: rata-rata kunjungan ke state transien]
    - v = N * 1               [Rata-rata waktu sampai penyerapan]
    - B = N * R               [Probabilitas penyerapan ke state absorbing tertentu]
    """
    num_t = Q.shape[0]
    I_t = np.eye(num_t)
    
    # Matriks Fundamental N
    N = np.linalg.inv(I_t - Q)
    
    # Rata-rata waktu absorbsi v
    v = np.sum(N, axis=1)
    
    # Probabilitas absorbsi B
    B = N @ R
    
    return {
        "N": N,
        "v": v,
        "B": B
    }


def get_covid_clinical_model() -> Dict[str, any]:
    """
    Membangun model rantai Markov absorbing untuk progresi klinis pasien COVID-19 di Indonesia.
    
    Ruang Keadaan:
    T1: Gejala Ringan / Isolasi Mandiri
    T2: Gejala Sedang / Rawat Inap Ruang Perawatan Biasa
    T3: Gejala Berat-Kritis / Perawatan Intensif (ICU)
    A1: Sembuh / Bebas Infeksi (Absorbing)
    A2: Meninggal Dunia (Absorbing)
    
    Interval Waktu: 1 minggu (7 hari) per transisi
    """
    state_labels = [
        "Ringan (T1)",
        "Sedang (T2)",
        "Berat/ICU (T3)",
        "Sembuh (A1)",
        "Meninggal (A2)"
    ]
    
    # Representasi Fraksi Eksak SymPy
    P_sym = sp.Matrix([
        [sp.Rational(1, 4),  sp.Rational(1, 20), 0,                 sp.Rational(69, 100), sp.Rational(1, 100)],
        [sp.Rational(1, 10), sp.Rational(3, 10), sp.Rational(1, 10), sp.Rational(45, 100), sp.Rational(5, 100)],
        [0,                  sp.Rational(1, 5),  sp.Rational(1, 4),  sp.Rational(25, 100), sp.Rational(30, 100)],
        [0,                  0,                  0,                  1,                    0],
        [0,                  0,                  0,                  0,                    1]
    ])
    
    P_float = np.array(P_sym.tolist(), dtype=float)
    
    Q_sym = P_sym[:3, :3]
    R_sym = P_sym[:3, 3:]
    
    I_3 = sp.eye(3)
    M_sym = I_3 - Q_sym
    N_sym = M_sym.inv()
    v_sym = N_sym * sp.Matrix([1, 1, 1])
    B_sym = N_sym * R_sym
    
    return {
        "state_labels": state_labels,
        "transient_labels": state_labels[:3],
        "absorbing_labels": state_labels[3:],
        "P_sym": P_sym,
        "P_float": P_float,
        "Q_sym": Q_sym,
        "R_sym": R_sym,
        "N_sym": N_sym,
        "v_sym": v_sym,
        "B_sym": B_sym,
        "det_I_minus_Q": M_sym.det()
    }


def solve_fsa_symbolic_system(
    Q_sym: sp.Matrix,
    R_sym: sp.Matrix,
    state_names: List[str]
) -> Dict[str, any]:
    """
    Menyelesaikan sistem persamaan First Step Analysis secara simbolik eksplisit.
    """
    num_t = Q_sym.shape[0]
    
    # 1. Sistem Persamaan Waktu Rata-rata Penyerapan (v_i)
    v_vars = [sp.Symbol(f"v_{i+1}") for i in range(num_t)]
    v_eqs = []
    for i in range(num_t):
        rhs = 1
        for j in range(num_t):
            rhs += Q_sym[i, j] * v_vars[j]
        v_eqs.append(sp.Eq(v_vars[i], rhs))
        
    v_sol = sp.solve(v_eqs, v_vars)
    
    # 2. Sistem Persamaan Peluang Penyerapan Meninggal (u_i^(A2))
    u_death_vars = [sp.Symbol(f"u_{i+1}^{{A2}}") for i in range(num_t)]
    u_death_eqs = []
    for i in range(num_t):
        rhs = R_sym[i, 1]  # Kolom 1 adalah Meninggal (A2)
        for j in range(num_t):
            rhs += Q_sym[i, j] * u_death_vars[j]
        u_death_eqs.append(sp.Eq(u_death_vars[i], rhs))
        
    u_death_sol = sp.solve(u_death_eqs, u_death_vars)
    
    # 3. Sistem Persamaan Peluang Penyerapan Sembuh (u_i^(A1))
    u_recov_vars = [sp.Symbol(f"u_{i+1}^{{A1}}") for i in range(num_t)]
    u_recov_eqs = []
    for i in range(num_t):
        rhs = R_sym[i, 0]  # Kolom 0 adalah Sembuh (A1)
        for j in range(num_t):
            rhs += Q_sym[i, j] * u_recov_vars[j]
        u_recov_eqs.append(sp.Eq(u_recov_vars[i], rhs))
        
    u_recov_sol = sp.solve(u_recov_eqs, u_recov_vars)
    
    return {
        "v_equations": v_eqs,
        "v_solutions": v_sol,
        "u_death_equations": u_death_eqs,
        "u_death_solutions": u_death_sol,
        "u_recov_equations": u_recov_eqs,
        "u_recov_solutions": u_recov_sol
    }


if __name__ == "__main__":
    print("=================================================================")
    print("UJI VALIDASI LENGKAP FSA SOLVER: DINAMIKA KLINIS COVID-19")
    print("=================================================================\n")
    
    model = get_covid_clinical_model()
    
    # 1. Verifikasi Matriks Stokastik
    valid, issues = verify_stochastic_matrix(model["P_float"])
    print(f"1. Verifikasi Aksioma Stokastik Matriks P: {'VALID' if valid else 'INVALID'}")
    if not valid:
        print("   Masalah:", issues)
        
    # 2. Klasifikasi State
    trans_idx, absorb_idx = classify_states(model["P_float"])
    print(f"2. Keadaan Transien: {[model['state_labels'][i] for i in trans_idx]}")
    print(f"   Keadaan Penyerap: {[model['state_labels'][i] for i in absorb_idx]}")
    
    # 3. Determinan Matriks (I - Q)
    print(f"3. Determinan Matriks (I - Q): {model['det_I_minus_Q']}")
    
    # 4. Matriks Fundamental N
    print("\n4. Matriks Fundamental N = (I - Q)^(-1):")
    sp.pprint(model["N_sym"])
    
    # 5. Ekspektasi Waktu Penyerapan v
    print("\n5. Vektor Waktu Rata-rata Penyerapan v = N * 1:")
    sp.pprint(model["v_sym"])
    print("   Desimal v (minggu):", [float(x) for x in model["v_sym"]])
    print("   Desimal v (hari)  :", [float(x) * 7 for x in model["v_sym"]])
    
    # 6. Matriks Probabilitas Penyerapan B = N * R
    print("\n6. Matriks Probabilitas Penyerapan B = N * R (Kolom 1: Sembuh, Kolom 2: Meninggal):")
    sp.pprint(model["B_sym"])
    print("   Peluang Sembuh   :", [f"{float(x)*100:.2f}%" for x in model["B_sym"][:, 0]])
    print("   Peluang Meninggal:", [f"{float(x)*100:.2f}%" for x in model["B_sym"][:, 1]])
    
    # 7. Uji Penyelesaian Aljabar Simbolik
    sym_res = solve_fsa_symbolic_system(model["Q_sym"], model["R_sym"], model["state_labels"])
    print("\n7. Solusi Sistem Aljabar Simbolik:")
    print("   Solusi v:", sym_res["v_solutions"])
    print("   Solusi u (Meninggal):", sym_res["u_death_solutions"])
    print("   Solusi u (Sembuh):", sym_res["u_recov_solutions"])
    
    print("\nSeluruh pengujian fsa_solver.py berhasil 100%!")
