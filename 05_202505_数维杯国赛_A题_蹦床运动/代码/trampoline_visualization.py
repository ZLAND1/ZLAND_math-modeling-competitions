import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
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

# 质量分配(Zatsiorsky模型)
m_torso = 0.45 * m_total
m_thigh = 0.10 * m_total
m_shin = 0.045 * m_total
m_foot = 0.012 * m_total

def calculate_moment_of_inertia(position_type):
    """计算不同空翻类型的转动惯量"""
    # 计算大腿绕质心的转动惯量
    I_thigh_cm = (1/12) * m_thigh * thigh_length**2
    
    if position_type == 'tuck':  # 团身
        # 使用平行轴定理计算大腿绕髋关节的转动惯量
        I_thigh_hip = I_thigh_cm + m_thigh * thigh_length**2
        # 躯干转动惯量(简化为圆柱体)
        I_torso = (1/12) * m_torso * (3 * 0.15**2 + (1.75 - leg_length)**2)
        # 总转动惯量
        return I_torso + 2 * I_thigh_hip
    elif position_type == 'pike':  # 屈体
        # 屈体转动惯量(介于团身和直体之间)
        return calculate_moment_of_inertia('tuck') * 1.5
    elif position_type == 'straight':  # 直体
        # 直体转动惯量(最大)
        return calculate_moment_of_inertia('tuck') * 2.5
    else:
        raise ValueError("未知的空翻类型")

def dynamics(t, y, F, theta_F, position_type):
    """系统动力学方程"""
    z, dz_dt, alpha, dalpha_dt = y
    Iz = calculate_moment_of_inertia(position_type)
    
    # 分解蹬伸力
    Fx = F * np.sin(np.radians(theta_F)) if t <= 0.2 else 0
    Fz = F * np.cos(np.radians(theta_F)) + m_total * g if t <= 0.2 else 0
    
    # 计算加速度
    d2z_dt2 = (Fz - m_total * g - k_d * z) / m_total
    d2alpha_dt2 = (Fx * h_cm) / Iz if t <= 0.2 else 0
    
    return [dz_dt, d2z_dt2, dalpha_dt, d2alpha_dt2]

def simulate_launch(F, theta_F, position_type, t_span=[0, 1.0], dt=0.01):
    """模拟起跳过程"""
    # 初始条件: z(0)=0, dz/dt(0)=-3m/s, alpha(0)=90度, dalpha/dt(0)=0
    y0 = [0, -3, np.radians(90), 0]
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # 求解微分方程
    sol = solve_ivp(lambda t, y: dynamics(t, y, F, theta_F, position_type), 
                    t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-6)
    
    return sol.t, sol.y

def plot_moment_of_inertia_comparison():
    """绘制三种空翻类型的转动惯量比较图"""
    positions = ['tuck', 'pike', 'straight']
    labels = ['团身', '屈体', '直体']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    I_values = [calculate_moment_of_inertia(pos) for pos in positions]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, I_values, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f} kg·m²', ha='center', va='bottom', fontsize=12)
    
    plt.title('不同空翻类型的转动惯量比较', fontsize=16)
    plt.ylabel('转动惯量 (kg·m²)', fontsize=14)
    plt.ylim(0, max(I_values) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('moment_of_inertia_comparison.png', dpi=300)
    plt.show()

def plot_force_decomposition():
    """绘制蹬伸力分解示意图"""
    plt.figure(figsize=(8, 8))
    
    # 绘制不同角度的力分解
    force_magnitude = 1200  # 示例力大小
    angles = [5, 15, 20]    # 不同空翻类型的典型角度
    labels = ['直体', '屈体', '团身']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    origin = (0, 0)
    
    for i, (angle, label, color) in enumerate(zip(angles, labels, colors)):
        # 分解力
        Fx = force_magnitude * np.sin(np.radians(angle))
        Fz = force_magnitude * np.cos(np.radians(angle)) + m_total * g
        
        # 绘制力矢量
        plt.arrow(origin[0], origin[1], Fx/10, Fz/10, 
                  head_width=0.2, head_length=0.3, fc=color, ec=color, 
                  length_includes_head=True, label=f'{label}: θ={angle}°')
        
        # 绘制分力
        plt.arrow(origin[0], origin[1], Fx/10, 0, 
                  head_width=0.1, head_length=0.2, fc='gray', ec='gray', 
                  linestyle='--', alpha=0.7)
        plt.arrow(origin[0] + Fx/10, origin[1], 0, Fz/10, 
                  head_width=0.1, head_length=0.2, fc='gray', ec='gray', 
                  linestyle='--', alpha=0.7)
        
        # 添加标签
        plt.text(Fx/20, -0.3, f'Fx={Fx:.0f}N', fontsize=10)
        plt.text(Fx/10 + 0.1, Fz/20, f'Fz={Fz:.0f}N', fontsize=10)
        
        # 移动原点以便显示多个力
        origin = (origin[0] + 5, origin[1])
    
    plt.title('不同空翻类型的蹬伸力分解示意图', fontsize=16)
    plt.xlabel('水平方向 (缩放单位)', fontsize=14)
    plt.ylabel('垂直方向 (缩放单位)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('force_decomposition.png', dpi=300)
    plt.show()

def plot_trajectory_and_kinematics(position_type, F, theta_F):
    """绘制运动轨迹和运动学参数"""
    t, y = simulate_launch(F, theta_F, position_type, t_span=[0, 1.0])
    
    # 提取状态变量
    z, dz_dt, alpha, dalpha_dt = y
    
    # 创建画布
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. 垂直位置-时间图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, z, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # 蹦床位置
    ax1.set_title('垂直位置 vs 时间', fontsize=14)
    ax1.set_xlabel('时间 (s)', fontsize=12)
    ax1.set_ylabel('垂直位置 (m)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 垂直速度-时间图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, dz_dt, 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # 零速度线
    ax2.set_title('垂直速度 vs 时间', fontsize=14)
    ax2.set_xlabel('时间 (s)', fontsize=12)
    ax2.set_ylabel('垂直速度 (m/s)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 角度-时间图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, np.degrees(alpha), 'm-', linewidth=2)
    ax3.set_title('躯干倾角 vs 时间', fontsize=14)
    ax3.set_xlabel('时间 (s)', fontsize=12)
    ax3.set_ylabel('角度 (度)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 角速度-时间图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, np.degrees(dalpha_dt), 'c-', linewidth=2)
    ax4.set_title('翻转角速度 vs 时间', fontsize=14)
    ax4.set_xlabel('时间 (s)', fontsize=12)
    ax4.set_ylabel('角速度 (度/s)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f'{position_type.capitalize()}前空翻的运动轨迹和运动学参数', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    plt.savefig(f'{position_type}_trajectory.png', dpi=300)
    plt.show()

def create_animation(position_type, F, theta_F):
    """创建运动员运动动画"""
    t, y = simulate_launch(F, theta_F, position_type, t_span=[0, 1.0])
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.set_title(f'{position_type.capitalize()}前空翻动作动画', fontsize=16)
    ax.set_xlabel('水平位置 (m)', fontsize=14)
    ax.set_ylabel('垂直位置 (m)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制蹦床
    trampoline = Rectangle((-1.5, -0.05), 3, 0.05, color='brown')
    ax.add_patch(trampoline)
    
    # 初始化运动员图形元素
    torso, = ax.plot([], [], 'r-', linewidth=6)
    left_thigh, = ax.plot([], [], 'b-', linewidth=5)
    left_shin, = ax.plot([], [], 'b-', linewidth=5)
    right_thigh, = ax.plot([], [], 'g-', linewidth=5)
    right_shin, = ax.plot([], [], 'g-', linewidth=5)
    cm_point, = ax.plot([], [], 'ko', markersize=8)  # 重心点
    
    # 时间文本
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    def init():
        torso.set_data([], [])
        left_thigh.set_data([], [])
        left_shin.set_data([], [])
        right_thigh.set_data([], [])
        right_shin.set_data([], [])
        cm_point.set_data([], [])
        time_text.set_text('')
        return torso, left_thigh, left_shin, right_thigh, right_shin, cm_point, time_text
    
    def update(frame):
        # 当前状态
        z, _, alpha, _ = y[:, frame]
        t_current = t[frame]
        
        # 计算身体各部分的位置
        torso_length = 1.75 - leg_length
        
        # 重心位置
        cm_x = 0
        cm_y = z + h_cm
        
        # 躯干位置
        torso_x = [cm_x - torso_length/2 * np.sin(alpha), 
                   cm_x + torso_length/2 * np.sin(alpha)]
        torso_y = [cm_y - torso_length/2 * np.cos(alpha), 
                   cm_y + torso_length/2 * np.cos(alpha)]
        
        # 髋关节位置(躯干底部)
        hip_x = torso_x[0]
        hip_y = torso_y[0]
        
        # 根据空翻类型确定腿部姿势
        if position_type == 'tuck':  # 团身: 膝盖弯曲
            knee_angle = np.radians(90)
        elif position_type == 'pike':  # 屈体: 膝盖微弯
            knee_angle = np.radians(120)
        else:  # 直体: 膝盖伸直
            knee_angle = np.radians(180)
        
        # 左腿位置
        left_knee_x = hip_x - thigh_length * np.sin(alpha)
        left_knee_y = hip_y - thigh_length * np.cos(alpha)
        
        lower_leg_angle = alpha + (np.pi - knee_angle)
        left_ankle_x = left_knee_x - shin_length * np.sin(lower_leg_angle)
        left_ankle_y = left_knee_y - shin_length * np.cos(lower_leg_angle)
        
        # 右腿位置(对称)
        right_knee_x = hip_x + thigh_length * np.sin(alpha)
        right_knee_y = hip_y + thigh_length * np.cos(alpha)
        
        right_ankle_x = right_knee_x + shin_length * np.sin(lower_leg_angle)
        right_ankle_y = right_knee_y + shin_length * np.cos(lower_leg_angle)
        
        # 更新图形
        torso.set_data(torso_x, torso_y)
        left_thigh.set_data([hip_x, left_knee_x], [hip_y, left_knee_y])
        left_shin.set_data([left_knee_x, left_ankle_x], [left_knee_y, left_ankle_y])
        right_thigh.set_data([hip_x, right_knee_x], [hip_y, right_knee_y])
        right_shin.set_data([right_knee_x, right_ankle_x], [right_knee_y, right_ankle_y])
        cm_point.set_data(cm_x, cm_y)
        
        # 更新时间文本
        time_text.set_text(f'时间: {t_current:.2f}s')
        
        return torso, left_thigh, left_shin, right_thigh, right_shin, cm_point, time_text
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, 
                        interval=50, blit=True, repeat=True)
    
    # 保存动画
    ani.save(f'{position_type}_animation.gif', writer='pillow', fps=20)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def compare_optimal_results():
    """比较三种空翻类型的优化结果"""
    # 已知的优化结果
    results = {
        'tuck': {'F': 1080, 'theta_F': 17.5, 'omega': 6.8, 'I': 3.862},
        'pike': {'F': 1150, 'theta_F': 10.0, 'omega': 4.5, 'I': 5.793},
        'straight': {'F': 1200, 'theta_F': 5.0, 'omega': 2.8, 'I': 9.655}
    }
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 蹬伸力大小比较
    positions = list(results.keys())
    labels = ['团身', '屈体', '直体']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    forces = [results[pos]['F'] for pos in positions]
    axes[0, 0].bar(labels, forces, color=colors, alpha=0.7)
    for i, force in enumerate(forces):
        axes[0, 0].text(i, force + 10, f'{force:.0f} N', ha='center', fontsize=12)
    axes[0, 0].set_title('不同空翻类型的蹬伸力大小比较', fontsize=14)
    axes[0, 0].set_ylabel('蹬伸力 (N)', fontsize=12)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 发力方向角比较
    angles = [results[pos]['theta_F'] for pos in positions]
    axes[0, 1].bar(labels, angles, color=colors, alpha=0.7)
    for i, angle in enumerate(angles):
        axes[0, 1].text(i, angle + 0.5, f'{angle:.1f}°', ha='center', fontsize=12)
    axes[0, 1].set_title('不同空翻类型的发力方向角比较', fontsize=14)
    axes[0, 1].set_ylabel('方向角 (度)', fontsize=12)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. 转动惯量比较
    I_values = [results[pos]['I'] for pos in positions]
    axes[1, 0].bar(labels, I_values, color=colors, alpha=0.7)
    for i, I in enumerate(I_values):
        axes[1, 0].text(i, I + 0.1, f'{I:.3f} kg·m²', ha='center', fontsize=12)
    axes[1, 0].set_title('不同空翻类型的转动惯量比较', fontsize=14)
    axes[1, 0].set_ylabel('转动惯量 (kg·m²)', fontsize=12)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. 翻转角速度比较
    omegas = [results[pos]['omega'] for pos in positions]
    axes[1, 1].bar(labels, omegas, color=colors, alpha=0.7)
    for i, omega in enumerate(omegas):
        axes[1, 1].text(i, omega + 0.1, f'{omega:.1f} rad/s', ha='center', fontsize=12)
    axes[1, 1].set_title('不同空翻类型的翻转角速度比较', fontsize=14)
    axes[1, 1].set_ylabel('翻转角速度 (rad/s)', fontsize=12)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('三种前空翻类型的优化结果比较', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    plt.savefig('comparison_results.png', dpi=300)
    plt.show()

def main():
    """主函数: 生成所有可视化图形"""
    # 1. 绘制转动惯量比较图
    plot_moment_of_inertia_comparison()
    
    # 2. 绘制蹬伸力分解示意图
    plot_force_decomposition()
    
    # 3. 绘制三种空翻类型的轨迹和运动学参数
    position_types = ['tuck', 'pike', 'straight']
    optimal_forces = [1080, 1150, 1200]  # 最优蹬伸力
    optimal_angles = [17.5, 10.0, 5.0]   # 最优方向角
    
    for pos, F, theta in zip(position_types, optimal_forces, optimal_angles):
        plot_trajectory_and_kinematics(pos, F, theta)
    
    # 4. 创建三种空翻类型的动画(取消注释以生成动画，需要较长时间)
    # for pos, F, theta in zip(position_types, optimal_forces, optimal_angles):
    #     create_animation(pos, F, theta)
    
    # 5. 比较三种空翻类型的优化结果
    compare_optimal_results()

if __name__ == "__main__":
    main()