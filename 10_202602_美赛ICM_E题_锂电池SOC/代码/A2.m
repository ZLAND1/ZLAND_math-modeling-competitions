% 给纯净仿真数据添加高斯白噪声，模拟实际实验测量误差
load('电池仿真_纯净数据.mat');

%% 1. 设定噪声参数（模拟实测误差，可调整）
voltage_noise_sigma = 0.005;  % 电压测量噪声标准差（0.005V，对应5mV，符合实际设备精度）
current_noise_sigma = 0.01;   % 电流测量噪声标准差（0.01A，对应10mA）

%% 2. 添加高斯白噪声
U_noisy = U + voltage_noise_sigma * randn(size(U));  % 带噪声端电压
I_noisy = I + current_noise_sigma * randn(size(I));  % 带噪声电流

%% 3. 保存带噪声数据
save('电池仿真_带噪声数据.mat', 't', 'I_noisy', 'SOC', 'U_noisy', 'U_OCV', 'U_R1', 'U_R2');
fprintf('带噪声数据生成完成！保存为「电池仿真_带噪声数据.mat」\n');

%% 4. 对比可视化
figure('Position', [200, 200, 1000, 500]);
plot(t/60, U, 'b-', 'LineWidth', 2, 'DisplayName', '纯净电压');
hold on;
plot(t/60, U_noisy, 'r--', 'LineWidth', 1.5, 'DisplayName', '带噪声电压');
xlabel('时间（min）'); ylabel('端电压（V）');
title('纯净电压与带噪声电压对比');
legend('Location', 'best');
grid on;