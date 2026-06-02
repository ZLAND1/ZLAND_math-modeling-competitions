"""
Split all_results.png into 4 separate high-quality figures for the paper.
Each figure is formatted for direct insertion into the Word document.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output/'
FIG_DIR = 'output/figures/'
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Shared style
# ============================================================
def setup_ax(ax, title, xlabel='X 坐标 (m)', ylabel='Y 坐标 (m)'):
    ax.set_title(title, fontsize=11, weight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_aspect('equal')

def add_legend(ax, fontsize=7.5):
    ax.legend(fontsize=fontsize, loc='upper right',
              framealpha=0.85, edgecolor='#cccccc')


# ============================================================
# Figure 1: Problem 1 — Noiseless time alignment
# ============================================================
def fig_problem1():
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    d1 = pd.read_excel('data/附件1.xlsx', sheet_name='方式1(4Hz)')
    d2 = pd.read_excel('data/附件1.xlsx', sheet_name='方式2(5Hz)')
    traj = pd.read_csv('output/problem1_10hz_traj.csv')

    ax.plot(d1['X坐标(m)'], d1['Y坐标(m)'], 'b.', ms=1.0, alpha=0.4, label='Method-1 (4Hz)')
    ax.plot(d2['X坐标(m)'], d2['Y坐标(m)'], 'r.', ms=1.0, alpha=0.4, label='Method-2 (5Hz, raw)')
    ax.plot(traj['X坐标(m)'], traj['Y坐标(m)'], 'g-', lw=1.2, label='10Hz fused trajectory')

    setup_ax(ax, 'Figure 1: Noise-free Time Alignment Result\n'
             '(delta = 198.43 s, RMSE ~ 1e-8 m)')
    add_legend(ax)

    # Annotation
    ax.annotate('Two trajectories\nperfectly overlap\nafter alignment',
                xy=(0.5, 3.0), fontsize=7.5, color='#555555',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFF0',
                          edgecolor='#aaaaaa', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR + 'fig1_problem1_alignment.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 1 done")


# ============================================================
# Figure 2: Problem 2 — Noise + system bias (before/after)
# ============================================================
def fig_problem2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    d1 = pd.read_excel('data/附件2.xlsx', sheet_name='方式1(4Hz)')
    d2 = pd.read_excel('data/附件2.xlsx', sheet_name='方式2(5Hz)')
    traj = pd.read_csv('output/problem2_10hz_traj.csv')

    # Left: before correction
    ax1.plot(d1['X坐标(m)'], d1['Y坐标(m)'], 'b.', ms=0.6, alpha=0.3, label='Method-1')
    ax1.plot(d2['X坐标(m)'], d2['Y坐标(m)'], 'r.', ms=0.6, alpha=0.3, label='Method-2 (raw)')
    setup_ax(ax1, '(a) Before correction\n(system bias causes ~3.9m offset)')
    add_legend(ax1)

    # Draw bias arrow
    mid = len(d1)//2
    ax1.annotate('', xy=(d1['X坐标(m)'].iloc[mid]+3.47, d1['Y坐标(m)'].iloc[mid]-1.83),
                 xytext=(d1['X坐标(m)'].iloc[mid], d1['Y坐标(m)'].iloc[mid]),
                 arrowprops=dict(arrowstyle='->', color='#FF6600', lw=2.0))
    ax1.text(d1['X坐标(m)'].iloc[mid]+0.3, d1['Y坐标(m)'].iloc[mid]-0.8,
             'bias\n~3.9 m', fontsize=7.5, color='#FF6600', weight='bold')

    # Right: after correction
    ax2.plot(d1['X坐标(m)'], d1['Y坐标(m)'], 'b.', ms=0.6, alpha=0.3, label='Method-1')
    ax2.plot(traj['X坐标(m)'], traj['Y坐标(m)'], 'g-', lw=1.0, label='10Hz fused')
    setup_ax(ax2, '(b) After joint estimation & correction\n(delta=50.46s, b=(3.47,-1.83)m, RMSE=1.23m)')
    add_legend(ax2)

    fig.suptitle('Figure 2: Joint Estimation of Time Offset and System Bias', fontsize=12,
                 weight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR + 'fig2_problem2_joint_estimation.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 2 done")


# ============================================================
# Figure 3: Problem 3 — Real data fusion
# ============================================================
def fig_problem3():
    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    d1 = pd.read_excel('data/附件3.xlsx', sheet_name='方式1(4Hz)')
    d2 = pd.read_excel('data/附件3.xlsx', sheet_name='方式2(5Hz)')
    traj = pd.read_csv('output/problem3_10hz_traj.csv')

    ax.plot(d1['X坐标(m)'], d1['Y坐标(m)'], 'b.', ms=1.5, alpha=0.4, label='Method-1 (4Hz, reference)')
    ax.plot(d2['X坐标(m)'], d2['Y坐标(m)'], 'r.', ms=1.5, alpha=0.4, label='Method-2 (5Hz, raw)')
    ax.plot(traj['X坐标(m)'], traj['Y坐标(m)'], 'g-', lw=1.3, label='10Hz fused trajectory')

    setup_ax(ax, 'Figure 3: Real Measurement Data Fusion\n'
             '(delta=-367.92s, |bias|<0.2m, RMSE=4.53m)')
    add_legend(ax)

    # Annotation
    ax.annotate('t-test: p>0.01\nno significant bias\n(|b|<0.2 m)',
                xy=(2.0, 4.0), fontsize=7.5, color='#555555',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFF0',
                          edgecolor='#aaaaaa', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR + 'fig3_problem3_real_data.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 3 done")


# ============================================================
# Figure 4: Problem 4 — Task scheduling
# ============================================================
def fig_problem4():
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    traj = pd.read_csv('output/problem3_10hz_traj.csv')
    shoots = pd.read_excel('data/附件4.xlsx', sheet_name='射击目标')
    photos = pd.read_excel('data/附件4.xlsx', sheet_name='拍照目标')

    # Trajectory
    ax.plot(traj['X坐标(m)'], traj['Y坐标(m)'], 'gray', lw=0.5, alpha=0.5, label='Robot trajectory')

    # All targets
    ax.scatter(shoots['X坐标(m)'], shoots['Y坐标(m)'], c='#FF4444', marker='^',
               s=40, edgecolors='darkred', linewidths=0.5, zorder=3, label='Shoot targets (18)')
    ax.scatter(photos['X坐标(m)'], photos['Y坐标(m)'], c='#4488FF', marker='s',
               s=40, edgecolors='darkblue', linewidths=0.5, zorder=3, label='Photo targets (18)')

    # Scheduled tasks (from paper Table 4)
    scheduled = [
        ('P18','拍照', None), ('S11','射击', None), ('P11','拍照', None),
        ('P05','拍照', None), ('P05','拍照', None), ('P05','拍照', None),
        ('P16','拍照', None), ('P18','拍照', None), ('S12','射击', None),
        ('S11','射击', None), ('P13','拍照', None), ('P09','拍照', None),
        ('S15','射击', None)
    ]

    for tid, ttype, _ in scheduled:
        tgt = shoots if ttype == '射击' else photos
        row = tgt[tgt['编号'] == tid]
        if len(row) > 0:
            marker = 'D' if ttype == '射击' else 'o'
            color = '#FF8800' if ttype == '射击' else '#00CC44'
            ax.scatter(row.iloc[0]['X坐标(m)'], row.iloc[0]['Y坐标(m)'],
                       c=color, marker=marker, s=80, edgecolors='black',
                       linewidths=0.6, zorder=5)

    # Legend for scheduled
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF8800',
                          markersize=7, markeredgecolor='black', markeredgewidth=0.5))
    labels.append('Scheduled shoot (4)')
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CC44',
                          markersize=7, markeredgecolor='black', markeredgewidth=0.5))
    labels.append('Scheduled photo (9)')

    setup_ax(ax, 'Figure 4: Optimal Task Scheduling on Fusion Trajectory\n'
             '(13 tasks: 4 shoot + 9 photo, P05: 3 multi-angle shots)')
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              framealpha=0.85, edgecolor='#cccccc')

    # Annotate P05
    p05 = photos[photos['编号'] == 'P05']
    if len(p05) > 0:
        ax.annotate('P05: 3-angle\nphoto', xy=(p05.iloc[0]['X坐标(m)'], p05.iloc[0]['Y坐标(m)']),
                    fontsize=7, color='#006600', weight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                              edgecolor='#006600', alpha=0.85))

    plt.tight_layout()
    fig.savefig(FIG_DIR + 'fig4_problem4_scheduling.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Figure 4 done")


if __name__ == '__main__':
    fig_problem1()
    fig_problem2()
    fig_problem3()
    fig_problem4()
    print(f"\nAll 4 figures saved to {FIG_DIR}")
