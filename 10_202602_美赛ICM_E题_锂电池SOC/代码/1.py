import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# ---------------------- 1. 初始化画布（适配学术论文尺寸） ----------------------
fig, ax = plt.subplots(figsize=(10, 4))  # 宽10英寸，高4英寸，适合论文横向排版
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')  # 关闭坐标轴，仅保留电路图

# 定义通用参数（无填充、纯黑线条，符合物理电路图风格）
LINE_WIDTH = 2  # 电路线宽
FONT_SIZE = 12  # 标注字体大小
FONT_FAMILY = 'Times New Roman'  # 论文标准字体

# ---------------------- 2. 绘制电路组件（物理电路图标准符号，无填充） ----------------------
def draw_resistor(ax, x, y, width=0.8, height=0.4):
    """绘制电阻符号（锯齿线，无填充）"""
    # 电阻外框（仅线条，无填充）
    rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                             linewidth=LINE_WIDTH, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    # 锯齿线（电阻特征，5个齿）
    tooth_x = [x - width/2 + 0.1 + i*0.15 for i in range(6)]
    tooth_y_up = [y + height/4 for _ in tooth_x]
    tooth_y_down = [y - height/4 for _ in tooth_x]
    for i in range(5):
        ax.plot([tooth_x[i], tooth_x[i+1]], [tooth_y_down[i], tooth_y_up[i]], 
                'k-', linewidth=LINE_WIDTH)
        ax.plot([tooth_x[i+1], tooth_x[i+2] if i+2 < len(tooth_x) else tooth_x[i+1]], 
                [tooth_y_up[i], tooth_y_down[i+1] if i+1 < len(tooth_y_down) else tooth_y_down[i]], 
                'k-', linewidth=LINE_WIDTH)

def draw_capacitor(ax, x, y, width=0.8, height=0.4):
    """绘制电容符号（平行极板，无填充）"""
    # 电容外框（仅线条，无填充）
    rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                             linewidth=LINE_WIDTH, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    # 平行极板（电容特征）
    plate_x1 = x - 0.15
    plate_x2 = x + 0.15
    ax.plot([plate_x1, plate_x1], [y - height/4, y + height/4], 
            'k-', linewidth=LINE_WIDTH+1)
    ax.plot([plate_x2, plate_x2], [y - height/4, y + height/4], 
            'k-', linewidth=LINE_WIDTH+1)

def draw_parallel_box(ax, x, y, width=1.2, height=1.0):
    """绘制并联支路框（仅线条，无填充，标注R-C并联）"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height, 
                         boxstyle="round,pad=0.05", linewidth=LINE_WIDTH, 
                         edgecolor='black', facecolor='none')
    ax.add_patch(box)

# ---------------------- 3. 定义组件坐标（保证电路拓扑清晰） ----------------------
# 核心组件x坐标（从左到右：电压源→RC1→R0→RC2→负载→输出电压）
x_uocv = 1.0
x_rc1 = 3.0
x_r0 = 5.0
x_rc2 = 7.0
x_load = 8.5
x_u = 9.5

# 统一y坐标（保证电路水平对齐）
y_common = 2.0

# 并联支路内部组件坐标（RC1和RC2内的R、C）
y_r1 = y_common + 0.3
y_c1 = y_common - 0.3
y_r2 = y_common + 0.3
y_c2 = y_common - 0.3

# ---------------------- 4. 绘制完整电路（无填充，纯线条） ----------------------
# ① 标注开路电压源 U_ocv(SOC)
ax.text(x_uocv, y_common + 0.5, r'$U_{ocv}(SOC)$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ② 绘制第一级并联支路（R1-C1）
draw_parallel_box(ax, x_rc1, y_common)
draw_resistor(ax, x_rc1, y_r1, width=0.6, height=0.3)
ax.text(x_rc1, y_r1 + 0.4, r'$R_1$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')
draw_capacitor(ax, x_rc1, y_c1, width=0.6, height=0.3)
ax.text(x_rc1, y_c1 - 0.4, r'$C_1$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ③ 绘制欧姆内阻 R0
draw_resistor(ax, x_r0, y_common, width=0.8, height=0.4)
ax.text(x_r0, y_common + 0.5, r'$R_0$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ④ 绘制第二级并联支路（R2-C2）
draw_parallel_box(ax, x_rc2, y_common)
draw_resistor(ax, x_rc2, y_r2, width=0.6, height=0.3)
ax.text(x_rc2, y_r2 + 0.4, r'$R_2$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')
draw_capacitor(ax, x_rc2, y_c2, width=0.6, height=0.3)
ax.text(x_rc2, y_c2 - 0.4, r'$C_2$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ⑤ 绘制负载 Load
rect_load = patches.Rectangle((x_load - 0.2, y_common - 0.2), 0.4, 0.4, 
                              linewidth=LINE_WIDTH, edgecolor='black', facecolor='none')
ax.add_patch(rect_load)
ax.text(x_load, y_common + 0.5, r'$Load$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ⑥ 标注输出电压 U(t)
ax.text(x_u, y_common + 0.5, r'$U(t)$', fontsize=FONT_SIZE, 
        fontfamily=FONT_FAMILY, ha='center')

# ---------------------- 5. 绘制电路连接线（纯黑粗线，保证回路完整） ----------------------
# 串联主线（水平连接线）
ax.plot([x_uocv + 0.2, x_rc1 - 0.6], [y_common, y_common], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc1 + 0.6, x_r0 - 0.4], [y_common, y_common], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_r0 + 0.4, x_rc2 - 0.6], [y_common, y_common], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc2 + 0.6, x_load - 0.2], [y_common, y_common], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_load + 0.2, x_u - 0.2], [y_common, y_common], 'k-', linewidth=LINE_WIDTH)

# 并联支路内部连接线（R1/C1 与主线连接）
ax.plot([x_rc1 - 0.6, x_rc1 - 0.6], [y_common, y_r1], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc1 + 0.6, x_rc1 + 0.6], [y_common, y_r1], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc1 - 0.6, x_rc1 - 0.6], [y_common, y_c1], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc1 + 0.6, x_rc1 + 0.6], [y_common, y_c1], 'k-', linewidth=LINE_WIDTH)

ax.plot([x_rc2 - 0.6, x_rc2 - 0.6], [y_common, y_r2], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc2 + 0.6, x_rc2 + 0.6], [y_common, y_r2], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc2 - 0.6, x_rc2 - 0.6], [y_common, y_c2], 'k-', linewidth=LINE_WIDTH)
ax.plot([x_rc2 + 0.6, x_rc2 + 0.6], [y_common, y_c2], 'k-', linewidth=LINE_WIDTH)

# ---------------------- 6. 保存图片（论文标准矢量图，无失真） ----------------------
plt.tight_layout()
# 保存为PDF（矢量图，300dpi，支持论文无限放大）
plt.savefig('second_order_rc_circuit_python.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
# 可选：保存为PNG（高清位图）
# plt.savefig('second_order_rc_circuit_python.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()