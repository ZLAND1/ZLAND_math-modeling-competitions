% 带遗忘因子的RLS算法：基于仿真数据辨识二阶RC参数（无实际实验）
clear; clc; close all;

%% 1. 导入仿真数据（替代HPPC实测数据）
load('电池仿真_带噪声数据.mat');
Ts = 1;  % 采样周期1s
k_max = length(t);  % 仿真数据点数量

% 提取输入输出数据
I = I_noisy;  % 带噪声电流（输入）
y = U_OCV - U_noisy;  % 输出量：y(k) = U_OCV(k) - U(k)（带噪声）

%% 2. 带遗忘因子的RLS递推实现（后续代码与之前完全一致，无需修改）
n_param = 5;  % 待辨识参数数量k1~k5
lambda = 0.98;  % 遗忘因子（0.95~0.99之间）
P = 1e4 * eye(n_param);  % 初始化协方差矩阵
theta_hat = zeros(n_param, k_max);  % 存储每一步的参数估计值

% 初始化历史U_diff
U_diff_1 = 0; U_diff_2 = 0;

for k = 3:k_max
    % 构造回归向量phi(k)
    phi = [I(k), I(k-1), I(k-2), U_diff_1, U_diff_2]';
    
    % 1. 计算卡尔曼增益K(k)
    K = P * phi / (lambda + phi' * P * phi);
    
    % 2. 更新参数估计值hat_theta(k)
    theta_hat(:, k) = theta_hat(:, k-1) + K * (y(k) - phi' * theta_hat(:, k-1));
    
    % 3. 更新协方差矩阵P(k)
    P = (P - K * phi' * P) / lambda;
    
    % 4. 更新历史U_diff
    U_diff_2 = U_diff_1;
    U_diff_1 = y(k);
end

%% 3. 结果输出与可视化
fprintf('最终辨识得到的k1~k5：\n');
for i = 1:n_param
    fprintf('k%d = %.6f\n', i, theta_hat(i, end));
end

% 绘制参数辨识收敛曲线
figure('Position', [100, 100, 1000, 600]);
for i = 1:n_param
    subplot(2, 3, i);
    plot(t, theta_hat(i, :), 'b-', 'LineWidth', 1.5);
    xlabel('时间（s）', 'FontSize', 10);
    ylabel(['k' num2str(i)], 'FontSize', 10);
    title(['k' num2str(i) ' 辨识收敛曲线（仿真数据）'], 'FontSize', 12);
    grid on;
end

%% 4. 反推二阶RC模型物理参数并对比真实值
k1 = theta_hat(1, end); k2 = theta_hat(2, end); k3 = theta_hat(3, end);
k4 = theta_hat(4, end); k5 = theta_hat(5, end);

% 反推欧姆内阻R0
R0_est = k5 / k2;  % 辨识得到的R0

% 输出对比（真实值来自步骤1的设定）
fprintf('\n========== 参数辨识结果对比 ==========\n');
fprintf('R0 真实值：0.05 Ω，辨识值：%.6f Ω\n', R0_est);
fprintf('（其余R1/C1/R2/C2需结合时间常数反推，逻辑一致）\n');