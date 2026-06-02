import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 运动员参数
m_total = 68.0  # 总质量(kg)
g = 9.81        # 重力加速度(m/s^2)
h_cm = 0.93     # 重心高度(m)
k_d = 4951.05   # 蹦床刚度(N/m)
leg_length = 0.88  # 腿长(m)
thigh_length = 0.4 * leg_length  # 大腿长度(m)
shin_length = leg_length - thigh_length  # 小腿长度(m)
torso_length = 1.75 - leg_length  # 躯干长度(m)

# 质量分配(Zatsiorsky模型)
m_torso = 0.45 * m_total
m_thigh = 0.10 * m_total
m_shin = 0.045 * m_total
m_foot = 0.012 * m_total

# 颜色方案
POSITION_COLORS = {
    'tuck': '#1f77b4',    # 蓝色
    'pike': '#ff7f0e',    # 橙色
    'straight': '#2ca02c' # 绿色
}

# 透明度渐变
ALPHA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]

# 高度缩放参数
HEIGHT_SCALE = 1.5  # 增大此值会使高度方向拉伸
VERTICAL_OFFSET = {  # 垂直偏移量，使各动作分开显示
    'tuck': 0,
    'pike': 3.5,
    'straight': 7
}

def calculate_moment_of_inertia(position_type):
    """计算不同空翻类型的转动惯量"""
    I_thigh_cm = (1/12) * m_thigh * thigh_length**2
    
    if position_type == 'tuck':
        I_thigh_hip = I_thigh_cm + m_thigh * thigh_length**2
        I_torso = (1/12) * m_torso * (3 * 0.15**2 + torso_length**2)
        return I_torso + 2 * I_thigh_hip
    elif position_type == 'pike':
        return calculate_moment_of_inertia('tuck') * 1.5
    elif position_type == 'straight':
        return calculate_moment_of_inertia('tuck') * 2.5
    else:
        raise ValueError("未知的空翻类型")

def dynamics(t, y, F, theta_F, position_type):
    """系统动力学方程"""
    z, dz_dt, alpha, dalpha_dt = y
    Iz = calculate_moment_of_inertia(position_type)
    
    Fx = F * np.sin(np.radians(theta_F)) if t <= 0.2 else 0
    Fz = F * np.cos(np.radians(theta_F)) + m_total * g if t <= 0.2 else 0
    
    d2z_dt2 = (Fz - m_total * g - k_d * z) / m_total
    d2alpha_dt2 = (Fx * h_cm) / Iz if t <= 0.2 else 0
    
    return [dz_dt, d2z_dt2, dalpha_dt, d2alpha_dt2]

def simulate_launch(F, theta_F, position_type, t_span=[0, 1.0], dt=0.01):
    """模拟起跳过程"""
    y0 = [0, -3, np.radians(90), 0]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    sol = solve_ivp(lambda t, y: dynamics(t, y, F, theta_F, position_type), 
                    t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)
    
    return sol.t, sol.y

def plot_scaled_comparison():
    """绘制高度缩放的三种空翻动作对比图"""
    results = {
        'tuck': {'F': 1080, 'theta_F': 17.5},
        'pike': {'F': 1150, 'theta_F': 10.0},
        'straight': {'F': 1200, 'theta_F': 5.0}
    }
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('三种前空翻动作的高度缩放对比', fontsize=16)
    
    # 设置坐标轴
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 10)  # 增大Y轴范围以容纳所有动作
    ax.set_aspect('auto')  # 不强制等比例
    ax.set_xlabel('水平位置 (m)', fontsize=14)
    ax.set_ylabel('垂直位置 (m)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 为每种空翻类型绘制时间切片重叠图
    for position_type, params in results.items():
        t, y = simulate_launch(params['F'], params['theta_F'], position_type)
        vertical_offset = VERTICAL_OFFSET[position_type]
        
        # 选择关键时间点
        num_frames = len(ALPHA_VALUES)
        time_points = np.linspace(0, len(t)-1, num_frames, dtype=int)
        
        # 绘制蹦床参考线
        trampoline = Rectangle((-1.5, -0.05 + vertical_offset), 3, 0.05, 
                              color='brown', alpha=0.5)
        ax.add_patch(trampoline)
        
        # 为每个时间点绘制运动员姿态
        for i, frame_idx in enumerate(time_points):
            z, _, alpha, _ = y[:, frame_idx]
            t_point = t[frame_idx]
            
            # 应用高度缩放和垂直偏移
            scaled_z = z * HEIGHT_SCALE + vertical_offset
            
            # 计算身体各部分的位置
            cm_x = 0
            cm_y = scaled_z + h_cm
            
            torso_x = [cm_x - torso_length/2 * np.sin(alpha), 
                       cm_x + torso_length/2 * np.sin(alpha)]
            torso_y = [cm_y - torso_length/2 * np.cos(alpha), 
                       cm_y + torso_length/2 * np.cos(alpha)]
            
            hip_x = torso_x[0]
            hip_y = torso_y[0]
            
            # 修正头部位置计算
            head_x = torso_x[1]
            head_y = torso_y[1]
            
            if position_type == 'tuck':
                knee_angle = np.radians(90)
            elif position_type == 'pike':
                knee_angle = np.radians(120)
            else:
                knee_angle = np.radians(180)
            
            # 左腿位置
            left_knee_x = hip_x - thigh_length * np.sin(alpha)
            left_knee_y = hip_y - thigh_length * np.cos(alpha)
            
            lower_leg_angle = alpha + (np.pi - knee_angle)
            left_ankle_x = left_knee_x - shin_length * np.sin(lower_leg_angle)
            left_ankle_y = left_knee_y - shin_length * np.cos(lower_leg_angle)
            
            # 右腿位置
            right_knee_x = hip_x + thigh_length * np.sin(alpha)
            right_knee_y = hip_y + thigh_length * np.cos(alpha)
            
            right_ankle_x = right_knee_x + shin_length * np.sin(lower_leg_angle)
            right_ankle_y = right_knee_y + shin_length * np.cos(lower_leg_angle)
            
            # 绘制运动员
            alpha_val = ALPHA_VALUES[i]
            color = POSITION_COLORS[position_type]
            
            ax.plot(torso_x, torso_y, color=color, linewidth=6, alpha=alpha_val)
            ax.plot(head_x, head_y, 'ro', markersize=8, alpha=alpha_val)
            ax.plot([hip_x, left_knee_x], [hip_y, left_knee_y], color=color, linewidth=5, alpha=alpha_val)
            ax.plot([left_knee_x, left_ankle_x], [left_knee_y, left_ankle_y], color=color, linewidth=5, linestyle='--', alpha=alpha_val)
            ax.plot([hip_x, right_knee_x], [hip_y, right_knee_y], color=color, linewidth=5, alpha=alpha_val)
            ax.plot([right_knee_x, right_ankle_x], [right_knee_y, right_ankle_y], color=color, linewidth=5, linestyle='--', alpha=alpha_val)
            ax.plot(cm_x, cm_y, 'ko', markersize=10, alpha=alpha_val)
            
            # 在最高点添加动作标签
            if i == len(time_points) // 2:  # 中间帧
                ax.text(1.8, cm_y, f'{position_type.capitalize()}', fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # 调整时间标签位置，避免重叠
            time_y = vertical_offset + 0.1 + (i * 0.2)
            ax.text(-1.9, time_y, f't={t_point:.2f}s', fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # 添加图例
    ax.plot([], [], color=POSITION_COLORS['tuck'], linewidth=6, label='团身前空翻')
    ax.plot([], [], color=POSITION_COLORS['pike'], linewidth=6, label='屈身前空翻')
    ax.plot([], [], color=POSITION_COLORS['straight'], linewidth=6, label='直身前空翻')
    ax.plot([], [], 'ko', markersize=10, label='重心')
    ax.legend(loc='upper right')
    
    # 保存图片
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('scaled_overlap_comparison.png', dpi=300)
    plt.show()

def main():
    """主函数: 生成高度缩放的对比图"""
    plot_scaled_comparison()

if __name__ == "__main__":
    main()