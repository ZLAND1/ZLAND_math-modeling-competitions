% 步骤3+步骤4：WOA-D-AKF联合估计SOC与Qeff + 仿真结果验证可视化
% 修复1：变量作用域问题（Q0_true透传）
% 修复2：维度不匹配问题（保证RMSE返回标量）
clear; clc; close all;

%% ===================== 1. 全局参数初始化 =====================
% ------------ 电池基础参数（参考磷酸铁锂电池，来自李心月论文） ------------
Q0_true = 3500;          % 真实额定容量（mAh），后续作为Qeff真实值
R0 = 0.05;               % 欧姆内阻（Ω）
R1 = 0.02;               % 电化学极化内阻（Ω）
C1 = 1000;               % 电化学极化电容（F）
R2 = 0.03;               % 浓差极化内阻（Ω）
C2 = 5000;               % 浓差极化电容（F）
Ts = 1;                  % 采样周期（s）
t_total = 3600;          % 仿真总时间（1小时，3600s）
SOC0_true = 1;           % 初始真实SOC（100%）
eta = 0.98;              % 充放电效率
voltage_noise_sigma = 0.005;  % 电压测量噪声标准差（5mV）

% ------------ WOA算法参数 ------------
pop_size = 20;           % 种群规模
max_iter = 40;           % 最大迭代次数
lb = [1e-6, 1e-8];       % 待优化参数下界 [Q, R]（AKF的噪声协方差）
ub = [1e-3, 1e-5];       % 待优化参数上界 [Q, R]

% ------------ 初始化变量 ------------
t = 0:Ts:t_total-Ts;     % 时间向量
N = length(t);           % 数据点数量

% ===================== 2. 生成动态工况数据（模拟智能手机真实使用场景） =====================
% 生成动态放电电流（高/中/低负载切换，模拟游戏/视频/待机）
I_dynamic = zeros(1, N);
for k = 1:N
    if mod(k, 600) < 120  % 每10分钟（600s）有2分钟（120s）高负载（游戏/导航）
        I_dynamic(k) = -1.5 + 0.3*randn();  % 高放电电流（~1.5A）
    elseif mod(k, 600) < 300  % 接下来3分钟中等负载（视频播放）
        I_dynamic(k) = -0.8 + 0.2*randn();  % 中等放电电流（~0.8A）
    else  % 剩余5分钟低负载（待机/后台）
        I_dynamic(k) = -0.1 + 0.05*randn();  % 低放电电流（~0.1A）
    end
end

% 基于二阶RC模型生成真实SOC、端电压（作为"实测"参考）
SOC_true = zeros(1, N);   % 真实SOC（1×N行向量）
U_true = zeros(1, N);     % 真实端电压（无噪声）
U_meas = zeros(1, N);     % 带噪声端电压（模拟实测）
U_OCV = zeros(1, N);      % 真实开路电压
U_R1 = zeros(1, N);       % 真实R1-C1极化电压
U_R2 = zeros(1, N);       % 真实R2-C2极化电压

% 初始值赋值
SOC_true(1) = SOC0_true;
U_R1(1) = 0;
U_R2(1) = 0;
tau1 = R1 * C1;           % 第一支路时间常数
tau2 = R2 * C2;           % 第二支路时间常数

% 递推生成真实数据
for k = 1:N-1
    % 步骤1：SOC→U_OCV（非线性映射，6阶多项式）
    soc = SOC_true(k);
    U_OCV(k) = 3.0 + 0.5*soc - 0.2*soc^2 + 0.1*soc^3 - 0.05*soc^4 + 0.01*soc^5;
    
    % 步骤2：更新极化电压
    U_R1(k+1) = U_R1(k) * exp(-Ts/tau1) + R1 * I_dynamic(k) * (1 - exp(-Ts/tau1));
    U_R2(k+1) = U_R2(k) * exp(-Ts/tau2) + R2 * I_dynamic(k) * (1 - exp(-Ts/tau2));
    
    % 步骤3：更新真实端电压
    U_true(k) = U_OCV(k) - R0*abs(I_dynamic(k)) - U_R1(k) - U_R2(k);
    
    % 步骤4：更新真实SOC（安时积分法）
    SOC_true(k+1) = SOC_true(k) - (eta * abs(I_dynamic(k)) * Ts) / (3600 * Q0_true);
    
    % 防止SOC越界
    if SOC_true(k+1) < 0
        SOC_true(k+1) = 0;
    end
end

% 补充最后一个点的数据
soc = SOC_true(N);
U_OCV(N) = 3.0 + 0.5*soc - 0.2*soc^2 + 0.1*soc^3 - 0.05*soc^4 + 0.01*soc^5;
U_true(N) = U_OCV(N) - R0*abs(I_dynamic(N)) - U_R1(N) - U_R2(N);

% 添加高斯白噪声，生成"实测"端电压
U_meas = U_true + voltage_noise_sigma * randn(size(U_true));

% ===================== 3. 定义AKF（自适应卡尔曼滤波）核心函数 =====================
% 【修改1】：添加Q0_true、SOC_true作为输入参数，解决变量作用域+维度问题
% 输入：保证所有输入维度匹配，返回标量RMSE
function [SOC_est, Qeff_est, rmse] = adaptive_kf(Q, R, I, U_meas, U_OCV, Ts, N, Q0_true, SOC_true)
    % 初始化AKF变量
    SOC_est = zeros(1, N);    % SOC估计值（1×N行向量，与SOC_true维度一致）
    Qeff_est = zeros(1, N);   % Qeff估计值（1×N行向量）
    P = eye(2);               % 协方差矩阵初始化
    x_hat = [1; Q0_true];     % 状态向量初始化 [SOC; Qeff]
    eta = 0.98;               % 充放电效率
    
    for k = 1:N-1
        % 步骤1：预测步
        F = [1, 0; 0, 1];     % 状态转移矩阵
        w = sqrt(Q) * randn(2, 1);  % 过程噪声向量
        x_hat_minus = F * x_hat + w;
        
        % 步骤2：更新协方差预测
        P_minus = F * P * F' + Q * eye(2);
        
        % 步骤3：测量更新步
        H = [1, 0];  % 测量矩阵
        v = sqrt(R) * randn();  % 测量噪声（标量）
        % 预测测量值（保证标量，避免维度错误）
        z_hat = U_OCV(k) - (eta * abs(I(k)) * Ts) / (3600 * x_hat_minus(2));
        
        % 卡尔曼增益（保证标量/正确矩阵维度）
        K = P_minus * H' / (H * P_minus * H' + R);
        
        % 步骤4：更新状态估计
        x_hat = x_hat_minus + K * (U_meas(k) - z_hat);
        
        % 步骤5：更新协方差矩阵
        P = (eye(2) - K * H) * P_minus;
        
        % 步骤6：存储估计结果（防止越界，保证1×N行向量）
        SOC_est(k+1) = max(0, min(1, x_hat(1)));
        Qeff_est(k+1) = max(2000, min(4000, x_hat(2)));
    end
    
    % 【修改2】：保证SOC_est和SOC_true同维度（均为1×N行向量），相减后为1×N行向量
    % 计算RMSE（最终返回标量，解决维度不匹配核心问题）
    error_vec = SOC_est - SOC_true;  % 1×N行向量，无广播，维度匹配
    rmse = sqrt(mean(error_vec .^ 2));  % mean后为标量，sqrt后仍为标量
end

% ===================== 4. 定义WOA的适应度函数（目标：最小化SOC估计RMSE） =====================
% 【修改3】：透传所有必要参数，保证返回标量fitness
function fitness = woa_fitness(x, I_dynamic, U_meas, U_OCV, Ts, N, Q0_true, SOC_true)
    Q = x(1);
    R = x(2);
    % 调用adaptive_kf，返回标量rmse
    [~, ~, rmse] = adaptive_kf(Q, R, I_dynamic, U_meas, U_OCV, Ts, N, Q0_true, SOC_true);
    fitness = rmse;  % 返回到标量，与fitness(i)维度匹配
end

% ===================== 5. WOA算法核心：优化AKF的Q和R =====================
% 初始化种群
pop = zeros(pop_size, 2);
for i = 1:pop_size
    pop(i, :) = lb + (ub - lb) .* rand(1, 2);
end

% 初始化适应度值和最优解（fitness为pop_size×1列向量，每个元素为标量）
fitness = zeros(pop_size, 1);
for i = 1:pop_size
    % 【修改4】：调用woa_fitness，返回标量赋值给fitness(i)，维度匹配
    fitness(i) = woa_fitness(pop(i, :), I_dynamic, U_meas, U_OCV, Ts, N, Q0_true, SOC_true);
end

[best_fitness, best_idx] = min(fitness);
best_x = pop(best_idx, :);  % 最优解 [Q_opt, R_opt]
best_fitness_history = zeros(max_iter, 1);  % 记录每代最优适应度

% WOA迭代寻优
for iter = 1:max_iter
    a = 2 - iter * (2 / max_iter);  % 系数a从2线性衰减到0
    
    for i = 1:pop_size
        r1 = rand(); r2 = rand();
        A = 2 * a * r1 - a;  % WOA公式34
        C = 2 * r2;           % WOA公式35
        
        % 情况1：包围猎物/收缩包围（|A| < 1）
        if abs(A) < 1
            D = abs(C * best_x - pop(i, :));
            new_pop = best_x - A * D;
        % 情况2：随机搜索（|A| ≥ 1）
        else
            rand_idx = randi(pop_size);
            rand_x = pop(rand_idx, :);
            D = abs(C * rand_x - pop(i, :));
            new_pop = rand_x - A * D;
        end
        
        % 情况3：泡沫网螺旋更新（p ≥ 0.5）
        p = rand();
        if p >= 0.5
            b = 1;  % 螺旋形状系数
            h = -1 + 2 * rand();  % 螺旋半径
            D = abs(best_x - pop(i, :));
            new_pop = D .* exp(b * h) .* cos(2 * pi * h) + best_x;
        end
        
        % 边界约束（防止超出[lb, ub]）
        new_pop = max(new_pop, lb);
        new_pop = min(new_pop, ub);
        
        % 【修改5】：更新适应度，返回标量，维度匹配
        new_fitness = woa_fitness(new_pop, I_dynamic, U_meas, U_OCV, Ts, N, Q0_true, SOC_true);
        if new_fitness < fitness(i)
            pop(i, :) = new_pop;
            fitness(i) = new_fitness;
        end
    end
    
    % 更新全局最优解
    [current_best_fitness, current_best_idx] = min(fitness);
    if current_best_fitness < best_fitness
        best_x = pop(current_best_idx, :);
        best_fitness = current_best_fitness;
    end
    best_fitness_history(iter) = best_fitness;
end

% 输出WOA优化结果
fprintf('================ WOA优化结果 ================\n');
fprintf('最优过程噪声协方差Q_opt = %.8f\n', best_x(1));
fprintf('最优测量噪声协方差R_opt = %.8f\n', best_x(2));
fprintf('优化后SOC估计最小RMSE = %.6f\n', best_fitness);

% ===================== 6. 用优化后的Q/R运行AKF，得到最终估计结果 =====================
% 【修改6】：调用adaptive_kf，获取最终标量RMSE和估计结果
[SOC_est, Qeff_est, final_rmse] = adaptive_kf(best_x(1), best_x(2), I_dynamic, U_meas, U_OCV, Ts, N, Q0_true, SOC_true);

% 计算SOC绝对误差（保证维度匹配，1×N行向量）
SOC_err = abs(SOC_est - SOC_true) * 100;  % 转换为百分比误差
max_err = max(SOC_err);

fprintf('================ 最终估计结果 ================\n');
fprintf('SOC估计最终RMSE = %.6f%%\n', final_rmse * 100);
fprintf('SOC估计最大绝对误差 = %.2f%%\n', max_err);
fprintf('Qeff估计平均值 = %.2f mAh（真实值=3500 mAh）\n', mean(Qeff_est));

% ===================== 7. 步骤4：仿真结果可视化（生成MCM报告可用图表） =====================
% 图表1：SOC真实值 vs WOA-D-AKF估计值对比
figure('Position', [100, 100, 1000, 700]);
subplot(2,1,1);
plot(t/60, SOC_true*100, 'k-', 'LineWidth', 2.5, 'DisplayName', '真实SOC');
hold on;
plot(t/60, SOC_est*100, 'b-.', 'LineWidth', 1.8, 'DisplayName', 'WOA-D-AKF估计SOC');
xlabel('时间（min）', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('SOC（%）', 'FontName', 'Times New Roman', 'FontSize', 12);
title('动态工况下SOC估计结果对比（仿真实验）', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontName', 'Times New Roman', 'FontSize', 10);
grid on; grid minor;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

% 图表2：SOC估计绝对误差
subplot(2,1,2);
plot(t/60, SOC_err, 'r-', 'LineWidth', 1.5, 'DisplayName', 'SOC绝对误差');
hold on;
plot(t/60, ones(1, N)*max_err, 'g--', 'LineWidth', 1, 'DisplayName', ['最大误差：' num2str(max_err, '%.2f') '%']);
xlabel('时间（min）', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('绝对误差（%）', 'FontName', 'Times New Roman', 'FontSize', 12);
title(['SOC估计误差分析（RMSE：' num2str(final_rmse*100, '%.2f') '%）'], 'FontSize', 14);
legend('Location', 'northeast', 'FontName', 'Times New Roman', 'FontSize', 10);
grid on; grid minor;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

% 图表3：WOA算法适应度收敛曲线（单独窗口，便于导出）
figure('Position', [200, 200, 800, 500]);
plot(1:max_iter, best_fitness_history*100, 'r-', 'LineWidth', 2);
xlabel('迭代次数', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('SOC估计RMSE（%）', 'FontName', 'Times New Roman', 'FontSize', 12);
title('WOA算法适应度收敛曲线（仿真实验）', 'FontSize', 14, 'FontWeight', 'bold');
grid on; grid minor;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

% ===================== 8. 导出高分辨率图片（MCM报告要求300dpi） =====================
print('-dpng', '-r300', 'SOC估计结果对比_动态工况.png');
print('-dpng', '-r300', 'WOA收敛曲线_优化Q_R.png');

% 保存关键结果数据（便于后续分析）
save('WOA-D-AKF仿真结果.mat', 't', 'SOC_true', 'SOC_est', 'Qeff_est', 'best_x', 'final_rmse');
fprintf('\n================ 仿真完成 ================\n');
fprintf('结果图片和数据已保存，可直接用于MCM报告！\n');