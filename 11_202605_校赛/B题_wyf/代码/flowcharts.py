"""
Generate English flowcharts with comfortable single line spacing.
Output: wyf/code/output/flowchart_q1.png ~ q4.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BOX_STYLE = "round,pad=0.06"
PROCESS_COLOR = "#E8F0FE"
DECISION_COLOR = "#FFF3CD"
IO_COLOR = "#D4EDDA"
LOOP_COLOR = "#FFF8E1"
BOX_EDGE = "#333333"
TEXT_COLOR = "#1a1a1a"
ARROW_COLOR = "#555555"

LINE_SP = 0.026

def draw_box(ax, x, y, w, h, text, color=PROCESS_COLOR, fontsize=8.5,
             bold=False, edge_width=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=BOX_STYLE, facecolor=color,
                         edgecolor=BOX_EDGE, linewidth=edge_width, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    lines = text.split('\n')
    n = len(lines)
    fs = fontsize if n <= 2 else fontsize - 0.5
    for i, line in enumerate(lines):
        yy = y + (n/2 - i - 0.5) * (fontsize * LINE_SP)
        ax.text(x, yy, line, ha='center', va='center', fontsize=fs,
                weight=weight, color=TEXT_COLOR, zorder=3)

def arrow(ax, x1, y1, x2, y2, rad=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.3,
                               connectionstyle=f'arc3,rad={rad}'), zorder=1)

def note(ax, x, y, text, fs=7.5, c='#666666'):
    ax.text(x, y, text, fontsize=fs, color=c, ha='center', va='center', style='italic')


def draw_q1():
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_title('Problem 1: Time Alignment Without Noise', fontsize=11,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (5.5, 10.0, 7.0, 0.95, 'Input: Method-1 (4Hz) + Method-2 (5Hz)\nNo noise, clock offset only', IO_COLOR),
        (5.5, 8.4, 7.0, 0.95, 'Cubic spline on Method-1 data\n-> Continuous reference trajectory p1(t)', PROCESS_COLOR),
        (5.5, 6.8, 7.0, 1.0,  'Define cost function:\nJ(delta) = Sum||p1(t2j - delta) - p2_raw||^2', PROCESS_COLOR),
        (5.5, 5.2, 7.0, 0.95, 'Coarse search: delta in [-503, 1048]s\n400 evenly-spaced samples -> delta_coarse', PROCESS_COLOR),
        (5.5, 3.7, 7.0, 0.95, 'Fine optimization: bounded search\n+-30s around delta_coarse -> delta*', PROCESS_COLOR),
        (5.5, 2.1, 7.0, 1.0,  "Time alignment: t2' = t2 - 198.43s\nMerge + cubic spline at 0.1s intervals", PROCESS_COLOR),
        (5.5, 0.7, 7.0, 0.95, 'Output: 10Hz fused trajectory\nRMSE ~ 0, verified noise-free assumption', IO_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)
    for i in range(len(nodes) - 1):
        arrow(ax, 5.5, nodes[i][1] - nodes[i][3]/2, 5.5, nodes[i+1][1] + nodes[i+1][3]/2)

    note(ax, 9.4, 5.2, 'RMSE\n~1e-8 m')
    plt.tight_layout()
    fig.savefig('output/flowchart_q1.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q1 done")


def draw_q2():
    fig, ax = plt.subplots(figsize=(6.5, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_title('Problem 2: Joint Estimation of Time Offset & System Bias', fontsize=11,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (4.2, 11.7, 6.5, 0.95, 'Input: Annex-2 with random noise\n+ fixed system bias b=(bx,by)', IO_COLOR),
        (4.2, 10.1, 6.5, 0.95, 'Savitzky-Golay filter\n(window=21, poly order=3)', PROCESS_COLOR),
        (4.2, 8.5, 6.5, 0.95, 'Smooth spline p1(t) on Method-1\nJoint cost J(delta, b)', PROCESS_COLOR),
        (4.2, 7.0, 6.5, 0.95, 'Initialize: MSE coarse search -> delta(0)\nb(0) = (0, 0)', PROCESS_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)
    for i in range(len(nodes) - 1):
        arrow(ax, 4.2, nodes[i][1] - nodes[i][3]/2, 4.2, nodes[i+1][1] + nodes[i+1][3]/2)

    y_loop = 4.5
    draw_box(ax, 4.2, y_loop, 8.2, 3.2,
             'Alternating optimization (iterate till |db|<1e-8):\n'
             '  Step 1: fix b, bounded scalar optimize delta\n'
             '  Step 2: fix delta, solve b analytically:\n'
             '       bx = mean(p2x_raw - p1x)\n'
             '       by = mean(p2y_raw - p1y)',
             color=LOOP_COLOR, fontsize=7.8)
    arrow(ax, 4.2, 7.0 - 0.47, 4.2, y_loop + 1.6)

    ax.annotate('', xy=(8.25, 6.1), xytext=(8.25, 4.8),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.2,
                               connectionstyle='arc3,rad=-0.5'), zorder=1)
    ax.text(9.7, 5.45, 'update\ndelta, b', fontsize=7.2, color='#666666', ha='center', va='center')

    draw_box(ax, 4.2, 2.2, 6.5, 0.95,
             'Output: delta*=50.46s, b*=(3.47, -1.83)m\nRMSE (after correction) = 1.23m', PROCESS_COLOR)
    arrow(ax, 4.2, y_loop - 1.6, 4.2, 2.2 + 0.47)

    draw_box(ax, 4.2, 0.8, 6.5, 0.95,
             'Final output: 10Hz fused trajectory\n(bias corrected + time aligned)', IO_COLOR)
    arrow(ax, 4.2, 2.2 - 0.47, 4.2, 0.8 + 0.47)

    note(ax, 11.6, 7.2, 'b is linear\n-> closed-form\ndimensionality\nreduction')
    plt.tight_layout()
    fig.savefig('output/flowchart_q2.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q2 done")


def draw_q3():
    fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Problem 3: Bias Detection & Fusion with Real Measurements', fontsize=11,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (7.5, 12.8, 9.2, 0.95, 'Input: Annex-3 real measurement data\nMethod-1 (1381 pts) + Method-2 (1621 pts)', IO_COLOR),
        (7.5, 11.1, 9.2, 0.95, 'Step 1: Assume no bias, time alignment via MSE\n-> delta0 = -367.93s', PROCESS_COLOR),
        (7.5, 9.5, 9.2, 0.95, 'Step 2: Compute position differences in overlap\nDx = x1 - x2_raw,  Dy = y1 - y2_raw', PROCESS_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)
    for i in range(len(nodes) - 1):
        arrow(ax, 7.5, nodes[i][1] - nodes[i][3]/2, 7.5, nodes[i+1][1] + nodes[i+1][3]/2)

    dy = 7.7
    draw_box(ax, 7.5, dy, 9.2, 1.1,
             'One-sample t-test (alpha=0.01)\nH0: mu_x = mu_y = 0   vs   H1: mu != 0',
             color=DECISION_COLOR, fontsize=8.5, bold=True)
    arrow(ax, 7.5, 9.5 - 0.47, 7.5, dy + 0.55)

    ax.text(7.5, 6.5, 'Result:  X: p=0.138  |  Y: p=0.037\nBoth p > 0.01, cannot reject H0',
            fontsize=7.8, color='#444444', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA',
                      edgecolor='#CCCCCC', linewidth=0.7))

    arrow(ax, 7.5, dy - 0.55, 3.0, 5.5)
    arrow(ax, 7.5, dy - 0.55, 12.0, 5.5)

    ax.text(1.5, 6.3, 'p < 0.01\nBias detected', fontsize=7.5, color='#CC0000',
            ha='center', va='center', weight='bold')
    draw_box(ax, 3.0, 5.2, 5.3, 0.8, 'Significant system bias\n-> Use Problem-2 joint estimation',
             color='#FCE4E4', fontsize=8, edge_width=1.2)
    arrow(ax, 3.0, 5.2 - 0.4, 3.0, 3.8 + 0.4)
    draw_box(ax, 3.0, 3.8, 5.3, 0.8, 'Alternating optimization -> delta*, b*\nCorrect & fuse -> 10Hz trajectory',
             PROCESS_COLOR, fontsize=8)

    ax.text(13.5, 6.3, 'p > 0.01\nNo bias', fontsize=7.5, color='#006600',
            ha='center', va='center', weight='bold')
    draw_box(ax, 12.0, 5.2, 5.3, 0.8, 'No significant system bias\n(|b| < 0.2m, negligible)',
             color='#E8F5E9', fontsize=8, edge_width=1.2)
    arrow(ax, 12.0, 5.2 - 0.4, 12.0, 3.8 + 0.4)
    draw_box(ax, 12.0, 3.8, 5.3, 0.9,
             'Still run joint estimation:\ndelta*=-367.92s, b*=(-0.13,-0.17)m\nRMSE = 4.53m',
             PROCESS_COLOR, fontsize=8)

    arrow(ax, 3.0, 3.8 - 0.4, 7.5, 2.0)
    arrow(ax, 12.0, 3.8 - 0.4, 7.5, 2.0)

    draw_box(ax, 7.5, 1.6, 9.2, 0.95,
             'Output: 10Hz fused position trajectory\n(used as input for Problem 4)', IO_COLOR)
    arrow(ax, 7.5, 2.0 + 0.47, 7.5, 1.6 + 0.47)

    plt.tight_layout()
    fig.savefig('output/flowchart_q3.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q3 done")


def draw_q4():
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15.5)
    ax.axis('off')
    ax.set_title('Problem 4: Weighted Interval Scheduling for Task Planning', fontsize=11,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes_data = [
        (7, 14.5, 10.5, 0.95, 'Input: Problem-3 10Hz trajectory + Annex-4 targets\nS01-S18 (shoot), P01-P18 (photo)', IO_COLOR, 8.5),
        (7, 12.6, 10.5, 0.95, 'Extract per-point features:\nposition (x,y), speed v, acceleration a, heading psi', PROCESS_COLOR, 8.5),
        (7, 10.6, 10.5, 1.2, 'Adaptive thresholds (trajectory percentiles):\nShoot: d in [3,55]m, v<=30m/s, a<=276m/s2, prep 1.5s\nPhoto: d in [2,55]m, v<=20m/s, a<=227m/s2, prep 0.5s',
         '#FFF8E1', 7.5),
        (7, 8.9, 10.5, 1.1, 'Candidate window extraction:\nScan trajectory -> check constraints -> find feasible segments\n-> verify full prep period -> photo: extra angle check (>=30 deg)',
         PROCESS_COLOR, 8),
        (7, 7.0, 10.5, 0.95, 'Build weighted intervals: [start, end, weight]\nShoot weight=0.85, Photo weight=1.0',
         PROCESS_COLOR, 8.5),
    ]

    for n in nodes_data:
        x, y, w, h, text, color = n[:6]
        fs = n[6]
        draw_box(ax, x, y, w, h, text, color=color, fontsize=fs)

    for i in range(len(nodes_data) - 1):
        y1 = nodes_data[i][1] - nodes_data[i][3]/2
        y2 = nodes_data[i+1][1] + nodes_data[i+1][3]/2
        arrow(ax, 7, y1, 7, y2)

    y_dp = 4.9
    draw_box(ax, 7, y_dp, 10.5, 1.8,
             'Weighted Interval Scheduling DP (O(n log n)):\n'
             '  (1) Sort intervals by end time\n'
             '  (2) Binary search p[j] = last non-conflicting predecessor\n'
             '  (3) DP:  dp[j] = max(dp[j-1],  w_j + dp[p[j]+1])',
             color='#E3F2FD', fontsize=7.8)
    arrow(ax, 7, nodes_data[-1][1] - nodes_data[-1][3]/2, 7, y_dp + 0.9)

    y_bt = 2.5
    draw_box(ax, 7, y_bt, 10.5, 0.95,
             'DP backtrack -> optimal task sequence\n13 tasks: 4 shoot + 9 photo',
             PROCESS_COLOR, 8.5)
    arrow(ax, 7, y_dp - 0.9, 7, y_bt + 0.47)

    y_out = 1.0
    draw_box(ax, 7, y_out, 10.5, 0.95,
             'Output: fill result.xlsx\nP05: 3 multi-angle photos taken (angle diff >= 30 deg)',
             IO_COLOR, 8.5)
    arrow(ax, 7, y_bt - 0.47, 7, y_out + 0.47)

    note(ax, 13.0, 8.9, '101\ncandidates')
    note(ax, 13.0, 4.9, 'Global\noptimum\nguaranteed')

    plt.tight_layout()
    fig.savefig('output/flowchart_q4.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q4 done")


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    draw_q1()
    draw_q2()
    draw_q3()
    draw_q4()
    print("All English flowcharts saved!")
