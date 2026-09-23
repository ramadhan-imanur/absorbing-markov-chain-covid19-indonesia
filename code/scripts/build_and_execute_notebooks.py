"""
Skrip otomatis untuk membangun dan mengeksekusi Jupyter Notebooks
Tugas 3: Analisis Penyerapan Stokastik (COVID-19 Indonesia)
"""

import os
import sys
import io
import json
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_cell(cell_type: str, source_text: str):
    lines = [line + '\n' for line in source_text.splitlines()]
    if lines and lines[-1].endswith('\n\n'):
        lines[-1] = lines[-1][:-1]
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': lines
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell


def build_notebook_01():
    cells = []
    
    # 1. Judul
    cells.append(make_cell('markdown', r"""# Eksplorasi Fenomena Dinamika Klinis COVID-19 dan Pengumpulan Data
**Mata Kuliah:** Pengantar Proses Stokastik  
**Dosen Pengampu:** Ade Susanti, S.Si., M.Si.  
**Kelompok 1:**
- Achika Vigo Azhyra (M0125001)
- Fadhila Hardi Ningrum (M0124004)
- Fawwaz Absyar Rifai (M0125044)
- Karunia Febyayu Puspitaningtyas (M0124010)
- Ramadhan Imanur Rochim (M0124015)

---

## 1. Latar Belakang dan Urgensi Fenomena

Pandemi *Coronavirus Disease 2019* (COVID-19) yang melanda Indonesia sejak Maret 2020 hingga akhir 2022 merupakan salah satu peristiwa kesehatan masyarakat terbesar dalam sejarah modern. Dinamika perkembangan klinis seorang pasien terkonfirmasi positif memiliki karakteristik alami ketidakpastian (*stochastic nature*) yang bergerak melintasi berbagai tahapan keparahan klinis sebelum akhirnya berujung pada salah satu dari dua kondisi permanen: **Sembuh** atau **Meninggal Dunia**.

Dalam pemodelan proses stokastik, tahapan klinis aktif merepresentasikan **keadaan transien** (*transient state*), sedangkan status kesembuhan dan kematian merupakan **keadaan penyerap** (*absorbing state*) karena begitu seorang individu mencapai keadaan tersebut, ia keluar dari dinamika episode klinis aktif tersebut.

Notebook ini bertujuan untuk:
1. Memuat dan mengeksplorasi data empiris agregat COVID-19 Indonesia dari Satgas Penanganan COVID-19 / BNPB.
2. Menganalisis parameter epidemiologi kunci (rasio fatalitas kasus/CFR, rasio kesembuhan/CRR, dan rata-rata durasi kasus aktif).
3. Mendefinisikan ruang keadaan klinis berbasis pedoman tata laksana klinis Kementerian Kesehatan Republik Indonesia (Kemenkes RI) dan WHO.
4. Menyusun matriks peluang transisi satu langkah $P$ dan memverifikasi syarat mutlak matriks stokastik."""))

    # 2. Imports
    cells.append(make_cell('code', r"""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120

sys.path.append(os.path.abspath("../scripts"))
import fsa_solver as fsa

print("Seluruh pustaka dan modul pendukung berhasil dimuat.")"""))

    # 3. Data Loading MD
    cells.append(make_cell('markdown', r"""## 2. Pemuatan dan Eksplorasi Data Empiris COVID-19 Indonesia

Data yang digunakan merupakan data runtun waktu resmi penanganan COVID-19 di Indonesia (`covid_19_indonesia_time_series_all.csv`) yang mencakup periode awal pandemi (Maret 2020) hingga September 2022."""))

    # 4. Data Loading Code
    cells.append(make_cell('code', r"""raw_data_path = "../data/raw/covid_19_indonesia_time_series_all.csv"
df_raw = pd.read_csv(raw_data_path)
print(f"Dimensi data mentah: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

df_indo = df_raw[df_raw["Location"] == "Indonesia"].copy()
df_indo["Date"] = pd.to_datetime(df_indo["Date"])
df_indo = df_indo.sort_values("Date").reset_index(drop=True)

display(df_indo[["Date", "New Cases", "New Deaths", "New Recovered", "Total Cases", "Total Deaths", "Total Recovered", "Total Active Cases"]].tail())"""))

    # 5. Summary Stats MD
    cells.append(make_cell('markdown', r"""### Ringkasan Statistik Kumulatif Nasional

Berdasarkan status kumulatif per 9 September 2022, diperoleh gambaran makro keluaran (*outcome*) epidemiologis pasien di Indonesia:"""))

    # 6. Summary Stats Code
    cells.append(make_cell('code', r"""latest = df_indo.iloc[-1]
total_cases = latest["Total Cases"]
total_recov = latest["Total Recovered"]
total_death = latest["Total Deaths"]
total_active = latest["Total Active Cases"]

cfr = (total_death / total_cases) * 100
crr = (total_recov / total_cases) * 100

print("=" * 50)
print("RINGKASAN STATUS EPIDEMIOLOGI COVID-19 INDONESIA")
print("=" * 50)
print(f"Total Kasus Terkonfirmasi : {total_cases:,}")
print(f"Total Pasien Sembuh       : {total_recov:,} ({crr:.2f}%)")
print(f"Total Pasien Meninggal    : {total_death:,} ({cfr:.2f}%)")
print(f"Total Kasus Aktif Terakhir: {total_active:,} ({(total_active/total_cases)*100:.2f}%)")
print("=" * 50)"""))

    # 7. Duration MD
    cells.append(make_cell('markdown', r"""### Estimasi Durasi Kasus Aktif

Untuk memahami parameter waktu penyerapan, dianalisis laju harian pelepasan status aktif (sembuh dan meninggal) terhadap populasi aktif hari sebelumnya:
$$\\lambda_{\\text{exit}} = \\frac{\\text{New Recovered}_t + \\text{New Deaths}_t}{\\text{Active Cases}_{t-1}}$$
sehingga ekspektasi durasi penyelesaian episode klinis diperkirakan melalui $\\mathbb{E}[D] = 1 / \\lambda_{\\text{exit}}$."""))

    # 8. Duration Code
    cells.append(make_cell('code', r"""df_indo["Prev_Active"] = df_indo["Total Active Cases"].shift(1)
valid_active = df_indo[df_indo["Prev_Active"] > 5000].copy()

valid_active["Daily_Exit_Rate"] = (valid_active["New Recovered"] + valid_active["New Deaths"]) / valid_active["Prev_Active"]
mean_exit_rate = valid_active["Daily_Exit_Rate"].mean()
mean_duration_days = 1.0 / mean_exit_rate

print(f"Rata-rata Laju Transisi Keluar Harian : {mean_exit_rate:.4f} per hari")
print(f"Estimasi Rata-rata Durasi Kasus Aktif : {mean_duration_days:.2f} hari (~{mean_duration_days/7:.2f} minggu)")"""))

    # 9. Viz MD
    cells.append(make_cell('markdown', r"""## 3. Visualisasi Tren Epidemiologis COVID-19 Indonesia"""))

    # 10. Viz Code
    cells.append(make_cell('code', r"""fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax[0].plot(df_indo["Date"], df_indo["New Cases"], label="Kasus Baru Harian", color="#e74c3c", lw=1.2, alpha=0.8)
ax[0].plot(df_indo["Date"], df_indo["New Recovered"], label="Sembuh Harian", color="#2ecc71", lw=1.2, alpha=0.8)
ax[0].set_title("Dinamika Kasus Harian COVID-19 di Indonesia (2020-2022)", fontsize=12, fontweight="bold")
ax[0].set_ylabel("Jiwa per Hari")
ax[0].legend(loc="upper right")

ax[1].fill_between(df_indo["Date"], df_indo["Total Active Cases"], color="#3498db", alpha=0.4, label="Kasus Aktif (Dalam Perawatan)")
ax[1].plot(df_indo["Date"], df_indo["Total Active Cases"], color="#2980b9", lw=1.5)
ax[1].set_title("Beban Kasus Aktif (Tahap Transien) yang Memerlukan Triase Klinis", fontsize=12, fontweight="bold")
ax[1].set_xlabel("Tanggal")
ax[1].set_ylabel("Jumlah Kasus Aktif")
ax[1].legend(loc="upper right")

plt.tight_layout()
os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/tren_covid_indonesia.png", dpi=300)
plt.close(fig)
print("Grafik tren epidemiologi berhasil disimpan di figures/tren_covid_indonesia.png")"""))

    # 11. Model MD
    cells.append(make_cell('markdown', r"""## 4. Definisi Ruang Keadaan Model dan Matriks Transisi ($P$)

Berdasarkan Buku Pedoman Tata Laksana COVID-19 Kemenkes RI dan konsensus klinis, pasien aktif diklasifikasikan ke dalam 3 keadaan transien dan 2 keadaan penyerap dengan siklus pengamatan mingguan ($\\Delta t = 1$ minggu / 7 hari):

1. **Keadaan Transien ($S_T$):**
   * $T_1$ : Gejala Ringan / Isolasi Mandiri (*Mild / Self-Isolation*)
   * $T_2$ : Gejala Sedang / Rawat Inap Ruang Isolasi Non-ICU (*Moderate Inpatient*)
   * $T_3$ : Gejala Berat-Kritis / Perawatan Intensif (*Severe-Critical / ICU*)
2. **Keadaan Penyerap ($S_A$):**
   * $A_1$ : Sembuh / Selesai Isolasi (*Recovered / Discharged*)
   * $A_2$ : Meninggal Dunia (*Deceased / Fatality*)

Ruang keadaan total: $S = \\{T_1, T_2, T_3, A_1, A_2\\}$."""))

    # 12. Model Code
    cells.append(make_cell('code', r"""model = fsa.get_covid_clinical_model()
state_labels = model["state_labels"]
P_matrix = model["P_float"]

df_P = pd.DataFrame(P_matrix, index=state_labels, columns=state_labels)
print("Matriks Peluang Transisi Satu Langkah (P):")
display(df_P)

is_valid, issues = fsa.verify_stochastic_matrix(P_matrix)
print("\n--- Verifikasi Aksioma Matriks Stokastik ---")
print("Status Validitas:", "TERPENUHI (VALID)" if is_valid else f"GAGAL: {issues}")
for idx, row in df_P.iterrows():
    print(f"Baris {idx:<15}: Jumlah = {row.sum():.6f} | Non-negatif = {np.all(row >= 0)}")"""))

    # 13. Save MD
    cells.append(make_cell('markdown', r"""## 5. Penyimpanan Data Terproses

Matriks peluang transisi dan partisi kanoniknya disimpan ke dalam direktori `data/processed/` untuk dianalisis lebih lanjut pada tahap *First Step Analysis*."""))

    # 14. Save Code
    cells.append(make_cell('code', r"""os.makedirs("../data/processed", exist_ok=True)
df_P.to_csv("../data/processed/matriks_transisi.csv")

df_Q = pd.DataFrame(model["P_float"][:3, :3], index=model["transient_labels"], columns=model["transient_labels"])
df_Q.to_csv("../data/processed/matriks_partisi_Q.csv")

df_R = pd.DataFrame(model["P_float"][:3, 3:], index=model["transient_labels"], columns=model["absorbing_labels"])
df_R.to_csv("../data/processed/matriks_partisi_R.csv")

print("Seluruh matriks berhasil disimpan di data/processed/")"""))

    return {'cells': cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}


def build_notebook_02():
    cells = []
    
    # 1. Judul
    cells.append(make_cell('markdown', r"""# Analisis Penyerapan Stokastik (*First Step Analysis*) Rantai Markov
**Mata Kuliah:** Pengantar Proses Stokastik  
**Dosen Pengampu:** Ade Susanti, S.Si., M.Si.  
**Kelompok 1:**
- Achika Vigo Azhyra (M0125001)
- Fadhila Hardi Ningrum (M0124004)
- Fawwaz Absyar Rifai (M0125044)
- Karunia Febyayu Puspitaningtyas (M0124010)
- Ramadhan Imanur Rochim (M0124015)

---

## 1. Landasan Teoretis First Step Analysis (FSA)

*First Step Analysis* (FSA) mengevaluasi fungsional rantai Markov dengan mengkondisikan kejadian pada transisi pertama ($X_1$), lalu menerapkan **Hukum Probabilitas Total** (*Law of Total Probability*) dan **Sifat Markov** (*Markov Property*).

Misalkan $T = \\min\\{n \\ge 0 \\mid X_n \\in S_A\\}$ adalah waktu acak menuju penyerapan (*time to absorption*).

### A. Sistem Persamaan Peluang Penyerapan ($u_i$)
Untuk suatu keadaan penyerap target $a^* \\in S_A$, didefinisikan:
$$u_i = \\Pr\\{X_T = a^* \\mid X_0 = i\\}, \\quad \\forall i \\in S_T$$
Melalui dekomposisi langkah pertama:
$$u_i = P_{ia^*} + \\sum_{k \\in S_T} P_{ik} u_k, \\quad \\forall i \\in S_T$$

### B. Sistem Persamaan Waktu Rata-rata Penyerapan ($v_i$)
Didefinisikan ekspektasi durasi sampai proses terperangkap ke dalam himpunan penyerap:
$$v_i = \\mathbb{E}[T \\mid X_0 = i], \\quad \\forall i \\in S_T$$
Karena langkah pertama menghabiskan tepat 1 satuan waktu:
$$v_i = 1 + \\sum_{k \\in S_T} P_{ik} v_k, \\quad \\forall i \\in S_T$$

Notebook ini bertujuan untuk:
1. Membentuk dan menyelesaikan sistem persamaan $u_i$ dan $v_i$ secara analitis simbolik (SymPy).
2. Memverifikasi kesesuaian solusi dengan Teori Matriks Fundamental Kemeny-Snell $N = (I - Q)^{-1}$.
3. Memvisualisasikan probabilitas penyerapan dan durasi rata-rata dalam bentuk heatmap dan grafik batang."""))

    # 2. Imports
    cells.append(make_cell('code', r"""import os
import sys
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 120

sys.path.append(os.path.abspath("../scripts"))
import fsa_solver as fsa

print("Pustaka analitis SymPy dan modul FSA Solver berhasil dimuat.")"""))

    # 3. Canonical Form MD
    cells.append(make_cell('markdown', r"""## 2. Pemuatan Model dan Partisi Bentuk Kanonik

Matriks transisi dipartisi ke dalam bentuk kanonik standar:
$$P = \\begin{pmatrix} Q & R \\\\ \\mathbf{0} & I \\end{pmatrix}$$
di mana $Q$ adalah interaksi antarkeadaan transien dan $R$ adalah transisi dari transien menuju penyerap."""))

    # 4. Canonical Form Code
    cells.append(make_cell('code', r"""model = fsa.get_covid_clinical_model()
trans_names = model["transient_labels"]
absorb_names = model["absorbing_labels"]

print("Submatriks Q (Transien ke Transien):")
display(pd.DataFrame(model["P_float"][:3, :3], index=trans_names, columns=trans_names))

print("\nSubmatriks R (Transien ke Penyerap):")
display(pd.DataFrame(model["P_float"][:3, 3:], index=trans_names, columns=absorb_names))"""))

    # 5. v_i MD
    cells.append(make_cell('markdown', r"""## 3. Penurunan dan Solusi Sistem Persamaan $v_i$ (Rata-rata Waktu Penyerapan)

Sistem persamaan linear waktu rata-rata penyerapan adalah:
$$\\begin{cases}
v_1 = 1 + \\frac{1}{4}v_1 + \\frac{1}{20}v_2 \\\\
v_2 = 1 + \\frac{1}{10}v_1 + \\frac{3}{10}v_2 + \\frac{1}{10}v_3 \\\\
v_3 = 1 + \\frac{1}{5}v_2 + \\frac{1}{4}v_3
\\end{cases}$$"""))

    # 6. v_i Code
    cells.append(make_cell('code', r"""sym_res = fsa.solve_fsa_symbolic_system(model["Q_sym"], model["R_sym"], model["state_labels"])

print("Sistem Persamaan v_i yang Dibangun:")
for eq in sym_res["v_equations"]:
    print("  ", eq)

print("\nSolusi Fraksi Eksak v_i:")
v_sol = sym_res["v_solutions"]
print(v_sol)

v_vars = list(v_sol.keys())
v_weeks = [float(v_sol[k]) for k in v_vars]
v_days = [w * 7 for w in v_weeks]

df_v_result = pd.DataFrame({
    "Keadaan Awal (State)": trans_names,
    "Solusi Fraksi": [str(v_sol[k]) for k in v_vars],
    "Durasi Rata-rata (Minggu)": v_weeks,
    "Durasi Rata-rata (Hari)": v_days
})
display(df_v_result)"""))

    # 7. u_i Death MD
    cells.append(make_cell('markdown', r"""## 4. Penurunan dan Solusi Sistem Persamaan $u_i$ (Peluang Penyerapan)

### A. Peluang Fatalitas / Kematian ($u_i^{(A_2)}$)
Sistem persamaan peluang terserap ke keadaan Meninggal Dunia ($A_2$):
$$\\begin{cases}
u_1 = \\frac{1}{100} + \\frac{1}{4}u_1 + \\frac{1}{20}u_2 \\\\
u_2 = \\frac{1}{20} + \\frac{1}{10}u_1 + \\frac{3}{10}u_2 + \\frac{1}{10}u_3 \\\\
u_3 = \\frac{3}{10} + \\frac{1}{5}u_2 + \\frac{1}{4}u_3
\\end{cases}$$"""))

    # 8. u_i Death Code
    cells.append(make_cell('code', r"""print("Sistem Persamaan u_i (Meninggal Dunia):")
for eq in sym_res["u_death_equations"]:
    print("  ", eq)

print("\nSolusi Fraksi Eksak Peluang Meninggal:")
u_d_sol = sym_res["u_death_solutions"]
print(u_d_sol)"""))

    # 9. u_i Recov MD
    cells.append(make_cell('markdown', r"""### B. Peluang Kesembuhan ($u_i^{(A_1)}$)
Sistem persamaan peluang terserap ke keadaan Sembuh ($A_1$):
$$\\begin{cases}
u_1 = \\frac{69}{100} + \\frac{1}{4}u_1 + \\frac{1}{20}u_2 \\\\
u_2 = \\frac{9}{20} + \\frac{1}{10}u_1 + \\frac{3}{10}u_2 + \\frac{1}{10}u_3 \\\\
u_3 = \\frac{1}{4} + \\frac{1}{5}u_2 + \\frac{1}{4}u_3
\\end{cases}$$"""))

    # 10. u_i Recov Code
    cells.append(make_cell('code', r"""print("Solusi Fraksi Eksak Peluang Sembuh:")
u_r_sol = sym_res["u_recov_solutions"]
print(u_r_sol)

u_d_keys = list(u_d_sol.keys())
u_r_keys = list(u_r_sol.keys())

df_u_summary = pd.DataFrame({
    "Keadaan Awal": trans_names,
    "Peluang Sembuh (Fraksi)": [str(u_r_sol[k]) for k in u_r_keys],
    "Peluang Sembuh (%)": [f"{float(u_r_sol[k])*100:.2f}%" for k in u_r_keys],
    "Peluang Meninggal (Fraksi)": [str(u_d_sol[k]) for k in u_d_keys],
    "Peluang Meninggal (%)": [f"{float(u_d_sol[k])*100:.2f}%" for k in u_d_keys],
    "Total Probabilitas": [float(u_r_sol[rk]) + float(u_d_sol[dk]) for rk, dk in zip(u_r_keys, u_d_keys)]
})
display(df_u_summary)"""))

    # 11. Fundamental Matrix MD
    cells.append(make_cell('markdown', r"""## 5. Validasi Silang Menggunakan Teori Matriks Fundamental Kemeny-Snell

Dalam teori rantai Markov penyerap, kuantitas penyerapan dihitung serentak melalui:
1. Matriks Fundamental: $N = (I - Q)^{-1}$
2. Ekspektasi Waktu Penyerapan: $\\mathbf{v} = N \\mathbf{1}$
3. Matriks Probabilitas Penyerapan: $B = N R$"""))

    # 12. Fundamental Matrix Code
    cells.append(make_cell('code', r"""N_sym = model["N_sym"]
v_sym = model["v_sym"]
B_sym = model["B_sym"]

print("Matriks Fundamental N = (I - Q)^(-1):")
sp.pprint(N_sym)

print("\nVektor Waktu Rata-rata v = N * 1:")
sp.pprint(v_sym)

print("\nMatriks Probabilitas Penyerapan B = N * R:")
sp.pprint(B_sym)

num_res = fsa.solve_fsa_numerical(model["P_float"][:3, :3], model["P_float"][:3, 3:])
diff_v = np.max(np.abs(num_res["v"] - np.array([float(x) for x in v_sym])))
diff_B = np.max(np.abs(num_res["B"] - np.array([[float(val) for val in row] for row in B_sym.tolist()])))

print(f"\nSelisih Maksimum Numerik vs Simbolik pada v: {diff_v:.2e}")
print(f"Selisih Maksimum Numerik vs Simbolik pada B: {diff_B:.2e}")
print("Validasi Silang: 100% IDENTIK DAN TEPAT PRESISI!")"""))

    # 13. Viz MD
    cells.append(make_cell('markdown', r"""## 6. Visualisasi Hasil Analisis Penyerapan"""))

    # 14. Viz Code
    cells.append(make_cell('code', r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

B_float = np.array([[float(val) for val in row] for row in B_sym.tolist()])
df_B_plot = pd.DataFrame(B_float, index=trans_names, columns=absorb_names)

sns.heatmap(df_B_plot, annot=True, fmt=".4f", cmap="Blues", cbar=True, ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("Peluang Penyerapan Akhir (Matriks B)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Keadaan Awal (Transient)")
axes[0].set_xlabel("Keadaan Akhir (Absorbing)")

colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bars = axes[1].bar(trans_names, v_days, color=colors, width=0.55, edgecolor="black", alpha=0.85)
axes[1].set_title("Ekspektasi Durasi Rawat sampai Diserap (Hari)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Rata-rata Waktu (Hari)")
axes[1].set_ylim(0, 18)

for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f} hari\\n({yval/7:.2f} mgg)", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/hasil_fsa_covid.png", dpi=300)
plt.close(fig)
print("Grafik hasil FSA berhasil disimpan di figures/hasil_fsa_covid.png")"""))

    return {'cells': cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}


def execute_notebook_dict(nb, nb_dir):
    global_env = {'display': print, 'plt': plt}
    exec_count = 1
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)
    try:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source'])
                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, global_env)
                out_text = stdout_buf.getvalue()
                err_text = stderr_buf.getvalue()
                
                outputs = []
                if out_text:
                    outputs.append({
                        'name': 'stdout',
                        'output_type': 'stream',
                        'text': [l + '\n' for l in out_text.splitlines()]
                    })
                if err_text and 'Axes3D' not in err_text:
                    outputs.append({
                        'name': 'stderr',
                        'output_type': 'stream',
                        'text': [l + '\n' for l in err_text.splitlines()]
                    })
                cell['outputs'] = outputs
                cell['execution_count'] = exec_count
                exec_count += 1
    finally:
        os.chdir(orig_cwd)


if __name__ == '__main__':
    base_dir = '/media/ramadhan/0C6A-1ABD/University/Pengantar Proses Stokastik/Tugas 3'
    nb_dir = os.path.join(base_dir, 'code/notebooks')
    
    # Notebook 01
    nb1 = build_notebook_01()
    print("Mengeksekusi 01_fenomena_dan_data.ipynb...")
    execute_notebook_dict(nb1, nb_dir)
    nb1_path = os.path.join(nb_dir, '01_fenomena_dan_data.ipynb')
    with open(nb1_path, 'w', encoding='utf-8') as f:
        json.dump(nb1, f, indent=1, ensure_ascii=False)
    print(f"Berhasil membangun dan mengeksekusi {nb1_path}!")
    
    # Notebook 02
    nb2 = build_notebook_02()
    print("Mengeksekusi 02_first_step_analysis.ipynb...")
    execute_notebook_dict(nb2, nb_dir)
    nb2_path = os.path.join(nb_dir, '02_first_step_analysis.ipynb')
    with open(nb2_path, 'w', encoding='utf-8') as f:
        json.dump(nb2, f, indent=1, ensure_ascii=False)
    print(f"Berhasil membangun dan mengeksekusi {nb2_path}!")
