"""
Generate Chinese-labeled flowcharts with comfortable single line spacing.
Output: wyf/code/output/flowchart_cn_q1.png ~ q4.png
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

LINE_SP = 0.026  # line spacing multiplier (single-spacing)

def draw_box(ax, x, y, w, h, text, color=PROCESS_COLOR, fontsize=8.5,
             bold=False, edge_width=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=BOX_STYLE, facecolor=color,
                         edgecolor=BOX_EDGE, linewidth=edge_width, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    lines = text.split('\n')
    n = len(lines)
    fs = fontsize
    if n >= 3:
        fs = fontsize - 0.5
    for i, line in enumerate(lines):
        yy = y + (n/2 - i - 0.5) * (fontsize * LINE_SP)
        ax.text(x, yy, line, ha='center', va='center', fontsize=fs,
                weight=weight, color=TEXT_COLOR, zorder=3)

def arrow(ax, x1, y1, x2, y2, rad=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.3,
                               connectionstyle=f'arc3,rad={rad}'), zorder=1)

def note(ax, x, y, text, fs=7.5, c='#666666'):
    ax.text(x, y, text, fontsize=fs, color=c, ha='center', va='center',
            style='italic')


# ============================================================
# Q1: Noiseless Time Alignment
# ============================================================
def draw_q1():
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_title('问题1：无噪声时间对齐流程', fontsize=12,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (5.5, 10.0, 7.2, 0.95, '输入：方式1数据(4Hz) + 方式2数据(5Hz)\n无噪声，仅开机时间不同步', IO_COLOR),
        (5.5, 8.4, 7.2, 0.95, '对方式1建立三次样条插值\n得到连续参考轨迹 p1(t)', PROCESS_COLOR),
        (5.5, 6.8, 7.2, 1.0,  '定义残差平方和代价函数\nJ(delta) = Sum ||p1(t2j - delta) - p2_raw||^2', PROCESS_COLOR),
        (5.5, 5.2, 7.2, 0.95, '粗搜索：delta 在 [-503, 1048]s 范围\n400点等间距采样 -> delta_coarse', PROCESS_COLOR),
        (5.5, 3.7, 7.2, 0.95, '精优化：以 delta_coarse 为中心\n+-30s bounded 标量优化 -> delta*', PROCESS_COLOR),
        (5.5, 2.1, 7.2, 1.0,  '时间对齐：t2' + "'" + ' = t2 - 198.43s\n合并数据，三次样条插值到 0.1s 间隔', PROCESS_COLOR),
        (5.5, 0.7, 7.2, 0.95, '输出：10Hz 融合位置轨迹\nRMSE ≈ 0，验证无噪声假设', IO_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)

    for i in range(len(nodes) - 1):
        arrow(ax, 5.5, nodes[i][1] - nodes[i][3]/2, 5.5, nodes[i+1][1] + nodes[i+1][3]/2)

    note(ax, 9.6, 5.2, 'RMSE\n≈1e-8 m')
    plt.tight_layout()
    fig.savefig('output/flowchart_cn_q1.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q1 (CN) done")


# ============================================================
# Q2: Joint Estimation with Noise & Bias
# ============================================================
def draw_q2():
    fig, ax = plt.subplots(figsize=(6.5, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_title('问题2：含噪声与系统偏差的联合估计流程', fontsize=12,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (4.2, 11.7, 6.5, 0.95, '输入：附件2数据\n含随机噪声 + 固定系统偏差 b', IO_COLOR),
        (4.2, 10.1, 6.5, 0.95, 'Savitzky-Golay 滤波\n(窗口长度21, 3阶多项式)', PROCESS_COLOR),
        (4.2, 8.5, 6.5, 0.95, '对方式1建立平滑样条 p1(t)\n联合代价函数 J(delta, b)', PROCESS_COLOR),
        (4.2, 7.0, 6.5, 0.95, '初始化：MSE 粗搜索 delta(0)\nb(0) = (0, 0)', PROCESS_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)

    for i in range(len(nodes) - 1):
        arrow(ax, 4.2, nodes[i][1] - nodes[i][3]/2, 4.2, nodes[i+1][1] + nodes[i+1][3]/2)

    # Iteration loop
    y_loop = 4.5
    draw_box(ax, 4.2, y_loop, 8.2, 3.2,
             '交替优化迭代（收敛条件：|db| < 1e-8）\n'
             '  Step 1: 固定 b，bounded 标量优化 delta\n'
             '  Step 2: 固定 delta，解析求解 b：\n'
             '       bx = mean(p2x_raw - p1x)\n'
             '       by = mean(p2y_raw - p1y)',
             color=LOOP_COLOR, fontsize=7.8)

    arrow(ax, 4.2, 7.0 - 0.47, 4.2, y_loop + 1.6)

    # Loop back arrow
    ax.annotate('', xy=(8.25, 6.1), xytext=(8.25, 4.8),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.2,
                               connectionstyle='arc3,rad=-0.5'), zorder=1)
    ax.text(9.7, 5.45, '迭代更新\ndelta, b', fontsize=7.2, color='#666666',
            ha='center', va='center')

    # Output after loop
    draw_box(ax, 4.2, 2.2, 6.5, 0.95,
             '收敛输出：delta*=50.46s, b*=(3.47, -1.83)m\n校正后 RMSE = 1.23m', PROCESS_COLOR)
    arrow(ax, 4.2, y_loop - 1.6, 4.2, 2.2 + 0.47)

    draw_box(ax, 4.2, 0.8, 6.5, 0.95,
             '最终输出：10Hz 融合位置轨迹\n(已校正系统偏差 + 时间对齐)', IO_COLOR)
    arrow(ax, 4.2, 2.2 - 0.47, 4.2, 0.8 + 0.47)

    note(ax, 11.6, 7.2, '利用 b 的\n线性可解性\n实现降维优化')
    plt.tight_layout()
    fig.savefig('output/flowchart_cn_q2.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q2 (CN) done")


# ============================================================
# Q3: Real Data Bias Detection + Fusion
# ============================================================
def draw_q3():
    fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('问题3：实际数据偏差检测与融合流程', fontsize=12,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes = [
        (7.5, 12.8, 9.2, 0.95, '输入：附件3 实际测量数据\n方式1(1381点) + 方式2(1621点)', IO_COLOR),
        (7.5, 11.1, 9.2, 0.95, 'Step 1: 假设无系统偏差\nMSE 代价函数时间对齐 -> delta0 = -367.93s', PROCESS_COLOR),
        (7.5, 9.5, 9.2, 0.95, 'Step 2: 在重叠时间点计算位置差异\nDx = x1 - x2_raw,  Dy = y1 - y2_raw', PROCESS_COLOR),
    ]

    for x, y, w, h, text, color in nodes:
        draw_box(ax, x, y, w, h, text, color=color)

    for i in range(len(nodes) - 1):
        arrow(ax, 7.5, nodes[i][1] - nodes[i][3]/2, 7.5, nodes[i+1][1] + nodes[i+1][3]/2)

    # Decision
    dy = 7.7
    draw_box(ax, 7.5, dy, 9.2, 1.1,
             '单样本 t 检验 (alpha = 0.01)\nH0: mu_x = mu_y = 0   vs   H1: mu != 0',
             color=DECISION_COLOR, fontsize=8.5, bold=True)
    arrow(ax, 7.5, 9.5 - 0.47, 7.5, dy + 0.55)

    # t-test result display
    ax.text(7.5, 6.5, '检验结果： X 方向 p=0.138，Y 方向 p=0.037\n两者均 p > 0.01，不能拒绝 H0（无显著系统偏差）',
            fontsize=7.8, color='#444444', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA',
                      edgecolor='#CCCCCC', linewidth=0.7))

    # Branch
    arrow(ax, 7.5, dy - 0.55, 3.0, 5.5)
    arrow(ax, 7.5, dy - 0.55, 12.0, 5.5)

    # Left: significant
    ax.text(1.5, 6.3, 'p < 0.01\n有显著偏差', fontsize=7.5, color='#CC0000',
            ha='center', va='center', weight='bold')
    draw_box(ax, 3.0, 5.2, 5.3, 0.8, '存在显著系统偏差\n-> 采用问题2的联合估计方法',
             color='#FCE4E4', fontsize=8, edge_width=1.2)
    arrow(ax, 3.0, 5.2 - 0.4, 3.0, 3.8 + 0.4)
    draw_box(ax, 3.0, 3.8, 5.3, 0.8, '交替优化 -> delta*, b*\n校正融合 -> 10Hz 轨迹',
             PROCESS_COLOR, fontsize=8)

    # Right: not significant (actual result)
    ax.text(13.5, 6.3, 'p > 0.01\n无显著偏差', fontsize=7.5, color='#006600',
            ha='center', va='center', weight='bold')
    draw_box(ax, 12.0, 5.2, 5.3, 0.8, '无显著系统偏差\n(|b| < 0.2m，可忽略)',
             color='#E8F5E9', fontsize=8, edge_width=1.2)
    arrow(ax, 12.0, 5.2 - 0.4, 12.0, 3.8 + 0.4)
    draw_box(ax, 12.0, 3.8, 5.3, 0.9,
             '仍进行联合估计验证：\ndelta*=-367.92s, b*=(-0.13, -0.17)m\nRMSE = 4.53m',
             PROCESS_COLOR, fontsize=8)

    # Merge
    arrow(ax, 3.0, 3.8 - 0.4, 7.5, 2.0)
    arrow(ax, 12.0, 3.8 - 0.4, 7.5, 2.0)

    draw_box(ax, 7.5, 1.6, 9.2, 0.95,
             '输出：10Hz 融合位置轨迹\n(作为问题4任务调度的输入)', IO_COLOR)
    arrow(ax, 7.5, 2.0 + 0.47, 7.5, 1.6 + 0.47)

    plt.tight_layout()
    fig.savefig('output/flowchart_cn_q3.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q3 (CN) done")


# ============================================================
# Q4: Task Scheduling
# ============================================================
def draw_q4():
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15.5)
    ax.axis('off')
    ax.set_title('问题4：加权区间调度任务优化流程', fontsize=12,
                 weight='bold', color=TEXT_COLOR, pad=14)

    nodes_data = [
        (7, 14.5, 10.5, 0.95, '输入：问题3的10Hz融合轨迹 + 附件4目标点\nS01~S18(射击目标) + P01~P18(拍照目标)',
         IO_COLOR, 8.5),
        (7, 12.6, 10.5, 0.95, '轨迹特征提取：每个时间点的\n位置(x,y), 线速度v, 加速度a, 航向角psi',
         PROCESS_COLOR, 8.5),
        (7, 10.6, 10.5, 1.2, '自适应约束阈值（基于轨迹百分位数）：\n'
         '射击：距离[3,55]m, 速度<=30m/s, 加速度<=276m/s2, 准备1.5s\n'
         '拍照：距离[2,55]m, 速度<=20m/s, 加速度<=227m/s2, 准备0.5s',
         '#FFF8E1', 7.5),
        (7, 8.9, 10.5, 1.1, '候选窗口提取：\n'
         '遍历轨迹点 -> 标记满足全部约束的点 -> 扫描连续可行段\n'
         '-> 检查完整准备期 -> 拍照目标额外检查角度差异 (>=30度)',
         PROCESS_COLOR, 8),
        (7, 7.0, 10.5, 0.95, '构建候选任务区间：[开始时刻, 结束时刻, 权重]\n'
         '射击权重 = 0.85，拍照权重 = 1.0',
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

    # DP core
    y_dp = 4.9
    draw_box(ax, 7, y_dp, 10.5, 1.8,
             '加权区间调度 DP 求解 (O(n log n))：\n'
             '  (1) 按结束时间升序排列所有候选区间\n'
             '  (2) 二分查找 p[j] = 最后一个不与 j 冲突的前驱\n'
             '  (3) 递推：dp[j] = max(dp[j-1],  w_j + dp[p[j]+1])',
             color='#E3F2FD', fontsize=7.8)
    arrow(ax, 7, nodes_data[-1][1] - nodes_data[-1][3]/2, 7, y_dp + 0.9)

    # Backtrack
    y_bt = 2.5
    draw_box(ax, 7, y_bt, 10.5, 0.95,
             'DP 回溯 -> 最优任务调度序列\n结果：13个任务（4射击 + 9拍照）',
             PROCESS_COLOR, 8.5)
    arrow(ax, 7, y_dp - 0.9, 7, y_bt + 0.47)

    # Output
    y_out = 1.0
    draw_box(ax, 7, y_out, 10.5, 0.95,
             '输出：填写 result.xlsx\nP05 目标完成3次多角度拍摄（角度差异>=30度）',
             IO_COLOR, 8.5)
    arrow(ax, 7, y_bt - 0.47, 7, y_out + 0.47)

    # Side annotations
    note(ax, 13.0, 8.9, '候选共\n101个')
    note(ax, 13.0, 4.9, '保证\n全局最优')

    plt.tight_layout()
    fig.savefig('output/flowchart_cn_q4.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("Q4 (CN) done")


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    draw_q1()
    draw_q2()
    draw_q3()
    draw_q4()
    print("All Chinese flowcharts saved!")
