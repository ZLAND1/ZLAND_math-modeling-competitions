import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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
    Fx = F * np.sin(np.radians(theta_F))
    Fz = F * np.cos(np.radians(theta_F)) + m_total * g
    
    # 计算加速度
    d2z_dt2 = (Fz - m_total * g - k_d * z) / m_total
    d2alpha_dt2 = (Fx * h_cm) / Iz
    
    return [dz_dt, d2z_dt2, dalpha_dt, d2alpha_dt2]

def objective(params, position_type):
    """优化目标函数: 最大化翻转角速度"""
    F, theta_F = params
    # 初始条件: z(0)=0, dz/dt(0)=-3m/s, alpha(0)=90度, dalpha/dt(0)=0
    y0 = [0, -3, np.radians(90), 0]
    t_span = [0, 0.2]
    t_eval = np.linspace(0, 0.2, 20)
    
    # 求解微分方程
    sol = solve_ivp(lambda t, y: dynamics(t, y, F, theta_F, position_type), 
                    t_span, y0, t_eval=t_eval, method='RK45')
    
    # 目标: 最大化最终的翻转角速度
    return -sol.y[3, -1]  # 取负号因为要最大化

def constraint(params):
    """约束条件: 起跳速度 >=2m/s"""
    F, theta_F = params
    y0 = [0, -3, np.radians(90), 0]
    t_span = [0, 0.2]
    t_eval = np.linspace(0, 0.2, 20)
    
    sol = solve_ivp(lambda t, y: dynamics(t, y, F, theta_F, 'tuck'), 
                    t_span, y0, t_eval=t_eval, method='RK45')
    
    # 返回起跳速度 - 2m/s, 必须>=0
    return sol.y[1, -1] - 2

def optimize_launch(position_type):
    """优化不同空翻类型的起跳参数"""
    # 设置约束条件
    cons = {'type': 'ineq', 'fun': constraint}
    
    # 设置参数边界
    bounds = [(m_total * g, m_total * g * 3),  # F的范围(1-3倍体重)
              (5, 25)]  # theta_F的范围(5-25度)
    
    # 初始猜测
    x0 = [m_total * g * 1.5, 15]
    
    # 执行优化
    result = minimize(objective, x0, args=(position_type), method='SLSQP', 
                      bounds=bounds, constraints=cons, options={'disp': True})
    
    return result.x, -result.fun  # 返回最优参数和最大翻转角速度

def main():
    """主函数: 计算并比较三种空翻类型"""
    positions = ['tuck', 'pike', 'straight']
    
    for position in positions:
        # 优化起跳参数
        (F_opt, theta_F_opt), max_omega = optimize_launch(position)
        
        # 计算分解后的力
        Fx_opt = F_opt * np.sin(np.radians(theta_F_opt))
        Fz_opt = F_opt * np.cos(np.radians(theta_F_opt)) + m_total * g
        
        # 计算转动惯量
        I = calculate_moment_of_inertia(position)
        
        # 打印结果
        print(f"\n{position.capitalize()} 前空翻结果:")
        print(f"最优蹬伸力: F = {F_opt:.2f} N")
        print(f"最优发力方向角: θ = {theta_F_opt:.2f} 度")
        print(f"垂直分力: Fz = {Fz_opt:.2f} N ({Fz_opt/(m_total*g):.2f} 倍体重)")
        print(f"水平分力: Fx = {Fx_opt:.2f} N")
        print(f"转动惯量: I = {I:.4f} kg·m^2")  # 修改了这一行
        print(f"最大翻转角速度: ω = {max_omega:.2f} rad/s")
        
        # 计算起跳速度
        y0 = [0, -3, np.radians(90), 0]
        t_span = [0, 0.2]
        t_eval = np.linspace(0, 0.2, 20)
        sol = solve_ivp(lambda t, y: dynamics(t, y, F_opt, theta_F_opt, position), 
                        t_span, y0, t_eval=t_eval, method='RK45')
        launch_velocity = sol.y[1, -1]
        print(f"起跳速度: v = {launch_velocity:.2f} m/s")

if __name__ == "__main__":
    main()