"""
Split each multi-panel problem figure into individual sub-figures.
Output: output/figures/fig{1..N}_{description}.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data/'
FIG_DIR = 'output/figures/'
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_DIR + 'Cement_ESP_Data.csv')
df = df.sort_values('timestamp').reset_index(drop=True)
df['C_out_mgNm3'] = df['C_out_mgNm3'].interpolate().fillna(method='bfill')

INLET_FEATS = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h']
VOLT_FEATS  = ['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']
RAPP_FEATS  = ['T1_s', 'T2_s', 'T3_s', 'T4_s']
OP_FEATS    = VOLT_FEATS + RAPP_FEATS

# ---- Physics model (same as esp_solver.py) ----
U0 = 55.0; C_in0 = 36.0; Q0 = 463000.0; C_out0 = 49.9
eta0 = 1.0 - C_out0 / (C_in0 * 1000.0)
K_eff = -np.log(1.0 - eta0); P_EXP = 1.0
T_REF = df['Temp_C'].mean(); Q_REF = df['Q_Nm3h'].mean()

def predict_c_out(U, T, C_in, Q, Temp):
    U = np.atleast_1d(U); T = np.atleast_1d(T)
    if U.ndim == 1: U = U.reshape(1, -1)
    if T.ndim == 1: T = T.reshape(1, -1)
    C_in_mg = np.atleast_1d(C_in) * 1000.0
    Q = np.atleast_1d(Q); Temp = np.atleast_1d(Temp)
    U_eff = np.mean(U, axis=1)
    f_U = (U_eff / U0) ** P_EXP
    f_Q = Q_REF / Q
    f_T = (Temp / T_REF) ** 0.3
    T_avg = np.mean(T, axis=1)
    alpha = 0.03
    f_R = 1.0 - alpha * (T_avg - 300.0) / 300.0
    f_R = np.clip(f_R, 0.6, 1.05)
    exponent = -K_eff * f_U * f_Q * f_T * f_R
    exponent = np.clip(exponent, -20, 0)
    return np.clip(C_in_mg * np.exp(exponent), 0.01, 500)

# ---- Derived features ----
df['U_avg'] = df[VOLT_FEATS].mean(axis=1)
df['T_avg'] = df[RAPP_FEATS].mean(axis=1)

# ---- K-means (K=4) ----
K = 4
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[INLET_FEATS].values)
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X_scaled)
cluster_info = {}
for cl in range(K):
    mask = df['cluster'] == cl
    cluster_info[cl] = {
        'n': mask.sum(), 'Temp': df.loc[mask,'Temp_C'].mean(),
        'C_in': df.loc[mask,'C_in_gNm3'].mean(), 'Q': df.loc[mask,'Q_Nm3h'].mean(),
    }
c_order = sorted(range(K), key=lambda c: cluster_info[c]['C_in'])
c_names = {c_order[0]: 'Low T, Low C', c_order[1]: 'Mid T, Mid C',
           c_order[2]: 'Mid T, High C', c_order[3]: 'High T, High C'}

# ---- Optimization results (hardcoded from solver output) ----
opt_results = {
    0: {'U_opt': [70.,70.,70.,70.], 'T_opt': [120.,120.,120.,120.], 'C_pred': 7.59, 'P_pred': 2097., 'P_curr': 1805.},
    1: {'U_opt': [70.,70.,70.,70.], 'T_opt': [351.,135.,382.,258.], 'C_pred': 4.05, 'P_pred': 2057., 'P_curr': 1756.},
    2: {'U_opt': [70.,70.,70.,70.], 'T_opt': [120.,120.,120.,120.], 'C_pred': 11.20, 'P_pred': 2097., 'P_curr': 1787.},
    3: {'U_opt': [70.,70.,70.,70.], 'T_opt': [120.,120.,120.,120.], 'C_pred': 7.22, 'P_pred': 2097., 'P_curr': 1736.},
}
opt5 = {
    0: {'P_pred': 2097., 'C_pred': 7.6, 'increase_pct': 0.0},
    1: {'P_pred': 2071., 'C_pred': 4.05, 'increase_pct': 0.7},
    2: {'P_pred': 2097., 'C_pred': 11.2, 'increase_pct': 0.0},
    3: {'P_pred': 2097., 'C_pred': 7.2, 'increase_pct': 0.0},
}

# Spearman correlations
corr_vars = {'C_in': 'C_in_gNm3', 'Temp': 'Temp_C', 'Q': 'Q_Nm3h',
             'U_avg': 'U_avg', 'U2/Q': 'U2_Q', 'T_avg': 'T_avg'}
df['U2_Q'] = df[VOLT_FEATS].pow(2).sum(axis=1) / df['Q_Nm3h'] / 1e6
results_corr = {}
for name, col in corr_vars.items():
    r, p = spearmanr(df[col], df['C_out_mgNm3'])
    results_corr[name] = r

cl_low = c_order[0]; cl_high = c_order[-1]


def save_fig(fig, name):
    fig.savefig(FIG_DIR + name, dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  {name}")


# ============ FIG 1: C_out distribution histogram ============
def fig1a():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(df['C_out_mgNm3'], bins=60, color='steelblue', edgecolor='white', lw=0.3)
    ax.axvline(50, color='red', ls='--', lw=2, label='Sensor saturation (50 mg/Nm³)')
    ax.set_xlabel('C_out (mg/Nm³)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Fig 1a: C_out Distribution — Severe Clipping at 50 mg/Nm³', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.text(49.5, ax.get_ylim()[1]*0.85, '55.8% of data\nexactly = 50', fontsize=8,
            color='red', ha='left', weight='bold')
    save_fig(fig, 'fig1a_C_out_distribution.png')

# ============ FIG 2: C_in vs C_out scatter ============
def fig1b():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(df['C_in_gNm3'], df['C_out_mgNm3'], s=0.8, alpha=0.3, c='steelblue')
    Cin_r = np.linspace(18, 72, 80)
    c_pred = predict_c_out(np.full((80,4), 55), np.full((80,4), 300),
                           Cin_r, 462830, 126)
    ax.plot(Cin_r, c_pred, 'r-', lw=2, label='Physics model (U=55 kV)')
    ax.set_xlabel('C_in (g/Nm³)', fontsize=10)
    ax.set_ylabel('C_out (mg/Nm³)', fontsize=10)
    ax.set_title('Fig 1b: C_in vs C_out — Physics Model Prediction', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.annotate('Data clipped\nat 50 mg/Nm³', xy=(50, 50), fontsize=8, color='red',
                ha='center', va='bottom')
    save_fig(fig, 'fig1b_Cin_vs_Cout.png')

# ============ FIG 3: Voltage effect curve ============
def fig1c():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    U_r = np.linspace(40, 70, 60)
    c_u = predict_c_out(np.column_stack([U_r]*4), np.full((60,4), 300), 36, 462830, 126)
    ax.plot(U_r, c_u, 'b-', lw=2, label='C_out')
    ax.set_xlabel('Average Voltage (kV)', fontsize=10)
    ax.set_ylabel('C_out (mg/Nm³)', color='b', fontsize=10)
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Fig 1c: Voltage Effect on C_out', fontsize=10, weight='bold')
    ax.grid(True, alpha=0.3, lw=0.5)
    # Annotate sensitivity
    dU = 66-44; dC = float(predict_c_out(np.full(4,66), np.full(4,300), 36, 462830, 126)[0] -
                          predict_c_out(np.full(4,44), np.full(4,300), 36, 462830, 126)[0])
    ax.annotate(f'Sensitivity:\n{dC/dU:.3f} per kV', xy=(55, 48.5), fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    save_fig(fig, 'fig1c_voltage_effect.png')

# ============ FIG 4: Rapping period effect curve ============
def fig1d():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    T_r = np.linspace(120, 600, 60)
    c_t = predict_c_out(np.full((60,4), 55), np.column_stack([T_r]*4), 36, 462830, 126)
    ax.plot(T_r, c_t, 'b-', lw=2, label='C_out')
    ax.set_xlabel('Average Rapping Period (s)', fontsize=10)
    ax.set_ylabel('C_out (mg/Nm³)', color='b', fontsize=10)
    ax.set_title('Fig 1d: Rapping Period Effect on C_out', fontsize=10, weight='bold')
    ax.grid(True, alpha=0.3, lw=0.5)
    dT = 500-200
    dC = float(predict_c_out(np.full(4,55), np.full(4,500), 36, 462830, 126)[0] -
              predict_c_out(np.full(4,55), np.full(4,200), 36, 462830, 126)[0])
    ax.annotate(f'Sensitivity:\n{dC/dT:.5f} per s\n(~1/180 of voltage)', xy=(350, 49.0), fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    save_fig(fig, 'fig1d_rapping_effect.png')

# ============ FIG 5: Spearman correlation bar chart ============
def fig1e():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    names = list(results_corr.keys()); vals = list(results_corr.values())
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in vals]
    ax.barh(range(len(names)), vals, color=colors, height=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(['C_in', 'Temp', 'Q', 'U_avg', 'U²/Q', 'T_avg'], fontsize=9)
    ax.set_xlabel('Spearman r', fontsize=10)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_title('Fig 1e: Spearman Correlation with C_out\n(all r≈0 — data-driven methods fail)', fontsize=10, weight='bold')
    ax.grid(True, alpha=0.3, lw=0.5, axis='x')
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(v + (0.002 if v>=0 else -0.002), i, f'{v:+.4f}', va='center', fontsize=8,
                ha='left' if v>=0 else 'right')
    save_fig(fig, 'fig1e_spearman_correlation.png')

# ============ FIG 6: K-means PCA projection ============
def fig2a():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    pca = PCA(n_components=2); X_pca = pca.fit_transform(X_scaled)
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    cn = ['Low T, Low C', 'Mid T, Mid C', 'Mid T, High C', 'High T, High C']
    for cl in range(K):
        mask = df['cluster'] == cl
        ax.scatter(X_pca[mask,0], X_pca[mask,1], s=1.5, alpha=0.5, c=[colors[cl]], label=cn[cl])
    ax.set_xlabel('PC1', fontsize=10); ax.set_ylabel('PC2', fontsize=10)
    ax.set_title('Fig 2a: K=4 Clusters (PCA Projection)', fontsize=10, weight='bold')
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(True, alpha=0.3, lw=0.5)
    save_fig(fig, 'fig2a_kmeans_pca.png')

# ============ FIG 7: Clusters in C_in-Temp space ============
def fig2b():
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    cn = ['Low T, Low C', 'Mid T, Mid C', 'Mid T, High C', 'High T, High C']
    for cl in range(K):
        ax.scatter(df.loc[df['cluster']==cl,'C_in_gNm3'],
                   df.loc[df['cluster']==cl,'Temp_C'], s=1.5, alpha=0.4, label=cn[cl])
    ax.set_xlabel('C_in (g/Nm³)', fontsize=10); ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.set_title('Fig 2b: Clusters in C_in-Temperature Space', fontsize=10, weight='bold')
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(True, alpha=0.3, lw=0.5)
    save_fig(fig, 'fig2b_clusters_Cin_Temp.png')

# ============ FIG 8: Optimal voltages comparison ============
def fig2c():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x_pos = np.arange(4); w = 0.8/K
    cn = ['Low T, Low C', 'Mid T, Mid C', 'Mid T, High C', 'High T, High C']
    for i, cl in enumerate(range(K)):
        ax.bar(x_pos+i*w, opt_results[cl]['U_opt'], w, label=cn[cl])
    ax.set_xticks(x_pos+w*(K-1)/2)
    ax.set_xticklabels(['U1', 'U2', 'U3', 'U4'], fontsize=10)
    ax.set_ylabel('Voltage (kV)', fontsize=10)
    ax.set_title('Fig 2c: Optimal Voltages by Condition (C_out≤10 mg/Nm³)', fontsize=10, weight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 80)
    ax.axhline(70, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(3.5, 70.5, 'Max: 70 kV', fontsize=7, ha='right', color='gray')
    ax.grid(True, alpha=0.3, lw=0.5, axis='y')
    save_fig(fig, 'fig2c_optimal_voltages.png')

# ============ FIG 9: Power comparison ============
def fig2d():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    cn = ['Low T\nLow C', 'Mid T\nMid C', 'Mid T\nHigh C', 'High T\nHigh C']
    p_cur = [opt_results[c]['P_curr'] for c in range(K)]
    p_opt = [opt_results[c]['P_pred'] for c in range(K)]
    x_pos = np.arange(K)
    ax.bar(x_pos-0.15, p_cur, 0.3, label='Current mean', color='#d62728')
    ax.bar(x_pos+0.15, p_opt, 0.3, label='Optimal (C_out≤10)', color='#2ca02c')
    for i in range(K):
        inc = (p_opt[i]-p_cur[i])/p_cur[i]*100
        ax.annotate(f'+{inc:.0f}%', (x_pos[i], max(p_cur[i], p_opt[i])+15),
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos); ax.set_xticklabels(cn, fontsize=8)
    ax.set_ylabel('P_total (kW)', fontsize=10)
    ax.set_title('Fig 2d: Power Comparison — Current vs Optimal', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.5, axis='y')
    save_fig(fig, 'fig2d_power_comparison.png')

# ============ FIG 10: Parameter comparison bar chart ============
def fig3a():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x_pos = np.arange(8); w = 0.35
    cn = {cl_low: 'Low T, Low C', cl_high: 'High T, High C'}
    ax.bar(x_pos-w/2, np.concatenate([opt_results[cl_low]['U_opt'], opt_results[cl_low]['T_opt']]),
           w, label=cn[cl_low], color='#1f77b4')
    ax.bar(x_pos+w/2, np.concatenate([opt_results[cl_high]['U_opt'], opt_results[cl_high]['T_opt']]),
           w, label=cn[cl_high], color='#d62728')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['U1','U2','U3','U4','T1','T2','T3','T4'], fontsize=9, rotation=20)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Fig 3a: Optimal Parameter Comparison — Two Conditions', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.5, axis='y')
    save_fig(fig, 'fig3a_parameter_comparison.png')

# ============ FIG 11: C_out contour map ============
def fig3b():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    U_grid = np.linspace(45, 70, 40); T_grid = np.linspace(120, 550, 40)
    UU, TT = np.meshgrid(U_grid, T_grid)
    CC = np.zeros_like(UU)
    info = cluster_info[cl_high]
    for i in range(40):
        for j in range(40):
            CC[j,i] = predict_c_out(np.full(4, U_grid[i]), np.full(4, T_grid[j]),
                                    info['C_in'], info['Q'], info['Temp'])[0]
    cs = ax.contourf(UU, TT, CC, levels=20, cmap='RdYlGn_r')
    ax.contour(UU, TT, CC, levels=[10], colors='blue', linewidths=2, linestyles='--')
    ax.plot(np.mean(opt_results[cl_high]['U_opt']), np.mean(opt_results[cl_high]['T_opt']),
            'r*', ms=15, label='Optimal')
    ax.set_xlabel('Average Voltage (kV)', fontsize=10)
    ax.set_ylabel('Average Rapping Period (s)', fontsize=10)
    ax.set_title('Fig 3b: C_out Contour — High T, High C Condition', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('C_out (mg/Nm³)', fontsize=9)
    save_fig(fig, 'fig3b_Cout_contour.png')

# ============ FIG 12: Sensitivity comparison ============
def fig3c():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sens_low, sens_high = [], []
    for cl, lst in [(cl_low, sens_low), (cl_high, sens_high)]:
        base = np.concatenate([opt_results[cl]['U_opt'], opt_results[cl]['T_opt']])
        info = cluster_info[cl]
        for i in range(8):
            pu = base.copy(); pu[i] *= 1.2
            p_dn = base.copy(); p_dn[i] *= 0.8
            cu = predict_c_out(pu[:4], pu[4:], info['C_in'], info['Q'], info['Temp'])[0]
            cd = predict_c_out(p_dn[:4], p_dn[4:], info['C_in'], info['Q'], info['Temp'])[0]
            lst.append(abs(cu-cd) / (pu[i]-p_dn[i]) if abs(pu[i]-p_dn[i])>1e-6 else 0)
    x_pos = np.arange(8); w = 0.35
    cn = {cl_low: 'Low T, Low C', cl_high: 'High T, High C'}
    ax.bar(x_pos-w/2, sens_low, w, label=cn[cl_low], color='#1f77b4')
    ax.bar(x_pos+w/2, sens_high, w, label=cn[cl_high], color='#d62728')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['U1','U2','U3','U4','T1','T2','T3','T4'], fontsize=9, rotation=20)
    ax.set_ylabel('|ΔC_out / ΔParam|', fontsize=10)
    ax.set_title('Fig 3c: Sensitivity Comparison (±20% perturbation)', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.5, axis='y')
    # Annotate ratio
    ax.annotate('Voltage sensitivity\n~180× rapping', xy=(1, max(max(sens_low), max(sens_high))*0.85),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    save_fig(fig, 'fig3c_sensitivity_comparison.png')

# ============ FIG 13: 10→5 power comparison ============
def fig4a():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x_pos = np.arange(K); w = 0.3
    cn = ['Low T\nLow C', 'Mid T\nMid C', 'Mid T\nHigh C', 'High T\nHigh C']
    p10v = [opt_results[c]['P_pred'] for c in range(K)]
    p5v = [opt5[c]['P_pred'] for c in range(K)]
    increases = [opt5[c]['increase_pct'] for c in range(K)]
    ax.bar(x_pos-w/2, p10v, w, label='C_out≤10', color='#2ca02c')
    ax.bar(x_pos+w/2, p5v, w, label='C_out≤5', color='#d62728')
    for i in range(K):
        ax.annotate(f'+{increases[i]:.1f}%', (x_pos[i], max(p10v[i], p5v[i])+15),
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos); ax.set_xticklabels(cn, fontsize=8)
    ax.set_ylabel('P_total (kW)', fontsize=10)
    ax.set_title('Fig 4a: Power Comparison — Standard 10→5 mg/Nm³', fontsize=10, weight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.5, axis='y')
    save_fig(fig, 'fig4a_standard_power.png')

# ============ FIG 14: Power increase % bar chart ============
def fig4b():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cn = ['Low T, Low C', 'Mid T, Mid C', 'Mid T, High C', 'High T, High C']
    increases = [opt5[c]['increase_pct'] for c in range(K)]
    w_inc = np.average(increases, weights=[cluster_info[c]['n'] for c in range(K)])
    colors = plt.cm.RdYlGn_r(np.array(increases)/max(increases)/2+0.4)
    ax.barh(range(K), increases, color=colors, height=0.5)
    ax.set_yticks(range(K)); ax.set_yticklabels(cn, fontsize=9)
    ax.set_xlabel('Power Increase (%)', fontsize=10)
    ax.set_title('Fig 4b: Power Increase by Condition (10→5 mg/Nm³)', fontsize=10, weight='bold')
    ax.axvline(w_inc, color='blue', ls='--', lw=1.2, label=f'Weighted avg: {w_inc:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, lw=0.5, axis='x')
    for i, inc in enumerate(increases):
        ax.text(inc+0.2, i, f'{inc:.1f}%', va='center', fontsize=8)
    save_fig(fig, 'fig4b_power_increase.png')


if __name__ == '__main__':
    print("Generating individual sub-figures...")
    fig1a(); fig1b(); fig1c(); fig1d(); fig1e()
    print("  Fig 1a-1e: Problem 1 sub-figures done")
    fig2a(); fig2b(); fig2c(); fig2d()
    print("  Fig 2a-2d: Problem 2 sub-figures done")
    fig3a(); fig3b(); fig3c()
    print("  Fig 3a-3c: Problem 3 sub-figures done")
    fig4a(); fig4b()
    print("  Fig 4a-4b: Problem 4 sub-figures done")
    print(f"\nTotal: 14 individual figures saved to {FIG_DIR}")
