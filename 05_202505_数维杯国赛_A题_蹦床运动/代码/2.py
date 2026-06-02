import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ================== 系统参数 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示设置
plt.rcParams['axes.unicode_minus'] = False     

m = 68.0          # 运动员质量 (kg)
g = 9.81          # 重力加速度 (m/s²)
k0 = 4951.05      # 线性刚度系数 (N/m)
beta = 1.2*k0     # 二次非线性刚度 (N/m²)
delta = 0.3*k0    # 三次非线性刚度 (N/m³)
c0 = 150.0        # 基础阻尼系数 (N·s/m)
gamma = 50.0      # 速度相关阻尼系数 (N·s²/m²)
mu = 0.4          # 摩擦系数
kx = 0.2*k0       # 水平刚度 (N/m)
rho = 1.225       # 空气密度 (kg/m³)
Cd = 1.2          # 阻力系数
A0 = 0.5         # 基准迎风面积 (m²)

# 多刚体参数
L_leg = 0.88      # 腿长 (m)
L_thigh = 0.352   # 大腿长度 (m)
L_shank = 0.528   # 小腿长度 (m)
m_thigh = 6.8     # 大腿质量 (kg)
m_shank = 4.08    # 小腿+足部质量 (kg)
m_trunk = m - m_thigh - m_shank  # 躯干质量

# ================== 动力学方程 ==================
def impact_ode(t, state, h0, theta):
    z, vz, x, vx, z_net = state
    
    # 空中阶段与触网阶段判断
    in_air = z > z_net
    
    # 多刚体质心计算
    z_trunk = z + 0.93  # 躯干位置假设（需根据实际模型调整）
    z_thigh = z - 0.5 * L_thigh * np.cos(theta)  # 大腿质心
    z_shank = z - L_thigh * np.cos(theta)         # 小腿质心
    z_COM = (m_trunk * z_trunk + m_thigh * z_thigh + m_shank * z_shank) / m
    
    # 空气阻力
    A = A0 * (1 - 0.6 * np.sin(theta))  # 迎风面积随角度变化
    Fd = -0.5 * Cd * rho * A * vz * abs(vz) if in_air else 0
    
    # 弹性力与阻尼力
    delta_z = z_COM - z_net
    F_spring = k0 * delta_z + beta * delta_z**2 + delta * delta_z**3
    F_damp = (c0 + gamma * abs(vz)) * vz * (1 + 0.5 * 1.14)  # η=1.14
    
    # 运动方程
    dzdt = vz
    dvzdt = (-m * g + Fd - F_spring - F_damp) / m if not in_air else (-g + Fd / m)
    dxdt = vx
    dvxdt = (kx * (x - 0.4 * 0.12) + mu * (F_spring + F_damp)) / m  # x_COP简化模型
    dz_netdt = 0  # 假设网面静止
    
    return [dzdt, dvzdt, dxdt, dvxdt, dz_netdt]

# ================== 数值求解（修正缩进） ==================
def simulate_impact(h0=2.0, theta=np.radians(0)):  # 函数定义行
    # 函数体必须缩进（4个空格或1个Tab）
    v0 = np.sqrt(2 * g * h0)  # 起跳初速度
    t_span = (0, 2)       # 仿真时间范围
    y0 = [h0, -v0, 0, 0, 0]  # 初始条件 [z, vz, x, vx, z_net]
    
    sol = solve_ivp(impact_ode, t_span, y0, args=(h0, theta),
                    method='RK45', max_step=0.01)
    return sol  # 函数体结束（缩进与函数内代码一致）

# ================== 结果可视化（修正缩进） ==================
def plot_results(sol, theta_deg):  # 函数定义行
    # 函数体必须缩进
    t = sol.t
    z, vz, x, vx, z_net = sol.y
    theta = np.radians(theta_deg)
    
    # 质心计算
    z_trunk = z + 0.93
    z_thigh = z - 0.5 * L_thigh * np.cos(theta)
    z_shank = z - L_thigh * np.cos(theta)
    z_COM = (m_trunk * z_trunk + m_thigh * z_thigh + m_shank * z_shank) / m
    delta_z = z_COM - z_net
    
    F_spring = k0 * delta_z + beta * delta_z**2 + delta * delta_z**3
    F_damp = (c0 + gamma * np.abs(vz)) * vz * (1 + 0.5 * 1.14)
    F_total = F_spring + F_damp
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, F_total, 'r-', label='总冲击力')
    plt.xlabel('时间 (s)')
    plt.ylabel('冲击力 (N)')
    plt.title(f'θ={theta_deg}° 时的冲击力曲线')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(t, vz, 'b-', label='垂直速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('垂直速度 (m/s)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(t, z, 'g-', label='人体顶点高度')
    plt.plot(t, z_COM, 'm--', label='质心高度')
    plt.xlabel('时间 (s)')
    plt.ylabel('高度 (m)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(z, vz, 'c-', label='相轨迹')
    plt.xlabel('位置 (m)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

# ================== 参数影响分析（修正缩进） ==================
def compare_parameters():  # 函数定义行
    # 函数体必须缩进
    plt.figure(figsize=(10, 6))
    
    # 不同起跳高度对比
    for h0 in [1.5, 2.0, 2.5]:
        sol = simulate_impact(h0=h0, theta=0)
        t = sol.t
        z = sol.y[0]
        theta = 0  # 弧度
        z_trunk = z + 0.93
        z_thigh = z - 0.5 * L_thigh * np.cos(theta)
        z_shank = z - L_thigh * np.cos(theta)
        z_COM = (m_trunk * z_trunk + m_thigh * z_thigh + m_shank * z_shank) / m
        delta_z = z_COM - sol.y[4]
        F_total = k0 * delta_z + beta * delta_z**2 + delta * delta_z**3
        plt.plot(t, F_total, label=f'起跳高度: {h0} m')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('冲击力 (N)')
    plt.title('不同起跳高度对冲击力的影响')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 不同曲腿角度对比
    plt.figure(figsize=(10, 6))
    for theta_deg in [0, 30, 60, 90]:
        theta = np.radians(theta_deg)
        sol = simulate_impact(h0=2.0, theta=theta)
        t = sol.t
        z = sol.y[0]
        z_trunk = z + 0.93
        z_thigh = z - 0.5 * L_thigh * np.cos(theta)
        z_shank = z - L_thigh * np.cos(theta)
        z_COM = (m_trunk * z_trunk + m_thigh * z_thigh + m_shank * z_shank) / m
        delta_z = z_COM - sol.y[4]
        F_total = k0 * delta_z + beta * delta_z**2 + delta * delta_z**3
        plt.plot(t, F_total, label=f'曲腿角度: {theta_deg}°')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('冲击力 (N)')
    plt.title('不同曲腿角度对冲击力的影响')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

# ================== 主程序（修正缩进） ==================
if __name__ == "__main__":  # 主程序入口
    # 函数体必须缩进
    m_trunk = m - m_thigh - m_shank  # 计算躯干质量
    print(f"躯干质量: {m_trunk:.2f} kg (总质量={m} kg)")
    
    # 单次仿真示例
    sol = simulate_impact(h0=2.0, theta=np.radians(60))
    print(f"仿真时间点数量: {len(sol.t)}, 状态变量维度: {sol.y.shape}")
    
    plot_results(sol, 60)
    compare_parameters()
    
    plt.show()  # 显示所有图像