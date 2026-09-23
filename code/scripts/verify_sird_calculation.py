"""
Verifikasi dan Komputasi Ulang Model Rantai Markov SIRD COVID-19 Indonesia
Kelompok 1 - Pengantar Proses Stokastik, Matematika FMIPA UNS (2026)
"""

import os
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Pastikan headless matplotlib
import matplotlib
matplotlib.use('Agg')

def verify_data_processing():
    raw_path = "Tugas 3/covid_19_indonesia_time_series_all.csv"
    if not os.path.exists(raw_path):
        raw_path = "Tugas 3/code/data/raw/covid_19_indonesia_time_series_all.csv"
    
    df = pd.read_csv(raw_path)
    df_nat = df[df['Location Level'] == 'Country'].copy()
    df_nat['Date'] = pd.to_datetime(df_nat['Date'])
    
    # Filter 2 Maret 2020 s.d. 16 September 2022
    df_filtered = df_nat[(df_nat['Date'] >= '2020-03-02') & (df_nat['Date'] <= '2022-09-16')].sort_values('Date').reset_index(drop=True)
    n_days = len(df_filtered)
    
    pop_N = 265185520
    df_filtered['S'] = pop_N - df_filtered['Total Cases']
    df_filtered['I'] = df_filtered['Total Active Cases']
    df_filtered['R'] = df_filtered['Total Recovered']
    df_filtered['D'] = df_filtered['Total Deaths']
    
    # Rasio transisi harian
    df_filtered['p_SI'] = df_filtered['New Cases'] / df_filtered['S']
    df_filtered['p_IR'] = df_filtered['New Recovered'] / df_filtered['I']
    df_filtered['p_ID'] = df_filtered['New Deaths'] / df_filtered['I']
    
    sum_p_SI = df_filtered['p_SI'].sum()
    sum_p_IR = df_filtered['p_IR'].sum()
    sum_p_ID = df_filtered['p_ID'].sum()
    
    mean_p_SI = sum_p_SI / n_days
    mean_p_IR = sum_p_IR / n_days
    mean_p_ID = sum_p_ID / n_days
    
    # Simpan data olahan
    proc_dir = "Tugas 3/code/data/processed"
    os.makedirs(proc_dir, exist_ok=True)
    df_filtered[['Date', 'S', 'I', 'R', 'D', 'p_SI', 'p_IR', 'p_ID']].to_csv(
        os.path.join(proc_dir, "data_sird_harian.csv"), index=False
    )
    
    print(f"=== HASIL VERIFIKASI DATA (Total Hari: {n_days}) ===")
    print(f"Populasi N: {pop_N:,}")
    print(f"Sum P_SI: {sum_p_SI:.9f} | Rata-rata P_SI: {mean_p_SI:.9f} (Draf: 0.000026)")
    print(f"Sum P_IR: {sum_p_IR:.9f} | Rata-rata P_IR: {mean_p_IR:.9f} (Draf: 0.062643)")
    print(f"Sum P_ID: {sum_p_ID:.9f} | Rata-rata P_ID: {mean_p_ID:.9f} (Draf: 0.002064)")
    
    return {
        'n_days': n_days,
        'sum_p_SI': sum_p_SI, 'mean_p_SI': mean_p_SI,
        'sum_p_IR': sum_p_IR, 'mean_p_IR': mean_p_IR,
        'sum_p_ID': sum_p_ID, 'mean_p_ID': mean_p_ID,
        'df': df_filtered
    }

def solve_fsa_exact_and_rounded():
    # Menggunakan nilai draf teman (dibulatkan)
    p_SS_d = 0.999974
    p_SI_d = 0.000026
    p_II_d = 0.935293
    p_IR_d = 0.062643
    p_ID_d = 0.002064
    
    print("\n=== PENYELESAIAN FSA MENGGUNAKAN NILAI DRAF TEMAN ===")
    # 1. Peluang Penyerapan u_i menuju Sembuh (R)
    # u_I = P_II * u_I + P_IR (1) + P_ID (0)
    # (1 - P_II) * u_I = P_IR => u_I = P_IR / (1 - P_II) = P_IR / (P_IR + P_ID)
    denom_I = p_IR_d + p_ID_d
    u_I_R_d = p_IR_d / denom_I
    u_I_D_d = p_ID_d / denom_I
    
    # u_S = P_SS * u_S + P_SI * u_I
    # (1 - P_SS) * u_S = P_SI * u_I => P_SI * u_S = P_SI * u_I => u_S = u_I
    u_S_R_d = u_I_R_d
    u_S_D_d = u_I_D_d
    
    print(f"Peluang Sembuh u_I(R): {u_I_R_d:.6f} ({u_I_R_d*100:.2f}%)")
    print(f"Peluang Sembuh u_S(R): {u_S_R_d:.6f} ({u_S_R_d*100:.2f}%)")
    print(f"Peluang Meninggal u_I(D): {u_I_D_d:.6f} ({u_I_D_d*100:.2f}%)")
    print(f"Peluang Meninggal u_S(D): {u_S_D_d:.6f} ({u_S_D_d*100:.2f}%)")
    
    # 2. Waktu Rata-rata Penyerapan v_i (dalam hari)
    # v_I = 1 + P_II * v_I => (1 - P_II) * v_I = 1 => v_I = 1 / (1 - P_II)
    v_I_d = 1.0 / denom_I
    # v_S = 1 + P_SS * v_S + P_SI * v_I => (1 - P_SS) * v_S = 1 + P_SI * v_I => P_SI * v_S = 1 + P_SI * v_I
    # v_S = (1 / P_SI) + v_I
    v_S_d = (1.0 / p_SI_d) + v_I_d
    
    print(f"\nWaktu Rata-rata Menuju Absorpsi (Hari):")
    print(f"v_I: {v_I_d:.4f} hari ({v_I_d/7.0:.2f} minggu)")
    print(f"v_S: {v_S_d:.2f} hari ({v_S_d/365.25:.2f} tahun)")
    
    # 3. Teori Matriks Fundamental Kemeny-Snell
    # Q = [[P_SS, P_SI], [0, P_II]]
    # R = [[0, 0], [P_IR, P_ID]]
    Q = np.array([[p_SS_d, p_SI_d], [0.0, p_II_d]])
    R = np.array([[0.0, 0.0], [p_IR_d, p_ID_d]])
    I_mat = np.eye(2)
    
    I_minus_Q = I_mat - Q
    det_val = np.linalg.det(I_minus_Q)
    N_mat = np.linalg.inv(I_minus_Q)
    v_vec = N_mat.dot(np.ones(2))
    B_mat = N_mat.dot(R)
    
    print(f"\nMatriks I - Q:\n{I_minus_Q}")
    print(f"Determinan det(I - Q): {det_val:.10e}")
    print(f"Matriks Fundamental N = (I - Q)^(-1):\n{N_mat}")
    print(f"Vektor v = N * 1:\n{v_vec}")
    print(f"Matriks Absorpsi B = N * R:\n{B_mat}")
    
    # Simpan matriks ke processed CSV
    proc_dir = "Tugas 3/code/data/processed"
    np.savetxt(os.path.join(proc_dir, "matriks_P_sird.csv"), 
               np.array([[p_SS_d, p_SI_d, 0, 0],
                         [0, p_II_d, p_IR_d, p_ID_d],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]), delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(proc_dir, "matriks_Q_sird.csv"), Q, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(proc_dir, "matriks_R_sird.csv"), R, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(proc_dir, "matriks_N_sird.csv"), N_mat, delimiter=",", fmt="%.4f")
    np.savetxt(os.path.join(proc_dir, "matriks_B_sird.csv"), B_mat, delimiter=",", fmt="%.6f")
    
    return {
        'u_I_R': u_I_R_d, 'u_S_R': u_S_R_d,
        'u_I_D': u_I_D_d, 'u_S_D': u_S_D_d,
        'v_I': v_I_d, 'v_S': v_S_d,
        'Q': Q, 'R': R, 'N': N_mat, 'B': B_mat, 'v': v_vec
    }

def generate_visualizations(df, fsa_results):
    fig_dir = "Tugas 3/code/figures"
    rep_fig_dir = "Tugas 3/report/figures"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rep_fig_dir, exist_ok=True)
    
    # 1. Grafik Dinamika Kompartemen SIRD di Indonesia
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    
    ax1.plot(df['Date'], df['S'] / 1e6, label='Susceptible (S)', color='#1f77b4', linewidth=1.8)
    ax1.set_xlabel('Tanggal', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Populasi Rentan S (Juta Jiwa)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['I'] / 1e3, label='Infected / Kasus Aktif (I)', color='#d62728', linewidth=1.5)
    ax2.plot(df['Date'], df['R'] / 1e6, label='Recovered (R)', color='#2ca02c', linewidth=1.5, linestyle='--')
    ax2.plot(df['Date'], df['D'] / 1e3, label='Death (D)', color='#7f7f7f', linewidth=1.5, linestyle=':')
    ax2.set_ylabel('I (Ribu Jiwa) & R (Juta Jiwa) & D (Ribu Jiwa)', color='#333333', fontsize=11, fontweight='bold')
    
    plt.title('Dinamika Kompartemen Epidemiologi SIRD COVID-19 di Indonesia (2020–2022)', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    
    p1 = os.path.join(fig_dir, "dinamika_sird_indonesia.png")
    fig.savefig(p1)
    fig.savefig(os.path.join(rep_fig_dir, "dinamika_sird_indonesia.png"))
    plt.close(fig)
    print(f"Tersimpan: {p1}")
    
    # 2. Grafik Hasil First Step Analysis (Peluang Absorpsi & Waktu Absorpsi)
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    
    # Bar Peluang Absorpsi
    states = ['Susceptible (S)', 'Infected (I)']
    p_sembuh = [fsa_results['u_S_R'] * 100, fsa_results['u_I_R'] * 100]
    p_meninggal = [fsa_results['u_S_D'] * 100, fsa_results['u_I_D'] * 100]
    
    x = np.arange(len(states))
    width = 0.35
    
    rects1 = ax_u.bar(x - width/2, p_sembuh, width, label='Sembuh (R)', color='#2ca02c')
    rects2 = ax_u.bar(x + width/2, p_meninggal, width, label='Meninggal (D)', color='#d62728')
    
    ax_u.set_ylabel('Peluang Absorpsi (%)', fontsize=11, fontweight='bold')
    ax_u.set_title('Peluang Penyerapan Akhir ($u_i$)', fontsize=11, fontweight='bold')
    ax_u.set_xticks(x)
    ax_u.set_xticklabels(states, fontsize=10, fontweight='bold')
    ax_u.set_ylim(0, 115)
    ax_u.legend(loc='upper right')
    
    for rect in rects1:
        h = rect.get_height()
        ax_u.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax_u.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bar Waktu Absorpsi
    v_vals = [fsa_results['v_S'], fsa_results['v_I']]
    colors = ['#1f77b4', '#ff7f0e']
    rects_v = ax_v.bar(states, v_vals, color=colors, width=0.45)
    ax_v.set_ylabel('Ekspektasi Hari Menuju Absorpsi (Skala Log)', fontsize=11, fontweight='bold')
    ax_v.set_yscale('log')
    ax_v.set_title('Rata-rata Waktu Menuju Absorpsi ($v_i$)', fontsize=11, fontweight='bold')
    ax_v.set_xticklabels(states, fontsize=10, fontweight='bold')
    
    for i, rect in enumerate(rects_v):
        h = rect.get_height()
        ket = f'{h:.1f} hari' if i == 1 else f'{h:.0f} hari\n(~{h/365.25:.1f} thn)'
        ax_v.annotate(ket, xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    fig.tight_layout()
    p2 = os.path.join(fig_dir, "hasil_fsa_sird.png")
    fig.savefig(p2)
    fig.savefig(os.path.join(rep_fig_dir, "hasil_fsa_sird.png"))
    plt.close(fig)
    print(f"Tersimpan: {p2}")

if __name__ == "__main__":
    data_res = verify_data_processing()
    fsa_res = solve_fsa_exact_and_rounded()
    generate_visualizations(data_res['df'], fsa_res)
    print("\n[SUKSES] Seluruh verifikasi, komputasi, dan visualisasi model SIRD 4x4 selesai sempurna.")
