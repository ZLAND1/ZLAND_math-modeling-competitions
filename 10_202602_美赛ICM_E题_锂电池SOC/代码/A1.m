% 二阶RC电池模型：生成纯净无噪声仿真数据（替代实际HPPC实验）
% 输出：时间、电流、SOC、端电压、开路电压、极化电压
clear; clc; close all;

%% 1. 设定电池基础参数（参考磷酸铁锂电池，来自李心月论文）
Q0 = 3500;          % 额定容量（mAh）
R0 = 0.05;          % 欧姆内阻（Ω）
R1 = 0.02;          % 电化学极化内阻（Ω）
C1 = 1000;          % 电化学极化电容（F）
R2 = 0.03;          % 浓差极化内阻（Ω）
C2 = 5000;          % 浓差极化电容（F）
Ts = 1;             % 采样周期（s），对应1Hz采样
t_total = 3600;     % 仿真总时间（1小时）
I_discharge = -1;   % 放电电流（A），负号为放电（行业惯例）
SOC0 = 1;           % 初始SOC（100%）

%% 2. 初始化变量
t = 0:Ts:t_total-Ts;  % 时间向量
N = length(t);        % 数据点数量
SOC = zeros(1, N);    % SOC向量
U = zeros(1, N);      % 端电压向量
U_OCV = zeros(1, N);  % 开路电压向量
U_R1 = zeros(1, N);   % R1-C1支路极化电压
U_R2 = zeros(1, N);   % R2-C2支路极化电压
I = ones(1, N) * I_discharge;  % 放电电流向量（恒流放电）

% 初始值赋值
SOC(1) = SOC0;
U_R1(1) = 0;
U_R2(1) = 0;

%% 3. 核心：基于二阶RC模型递推生成数据（对应连续时间方程离散化）
tau1 = R1 * C1;  % 第一支路时间常数
tau2 = R2 * C2;  % 第二支路时间常数
eta = 0.98;      % 充放电效率（常温下）

for k = 1:N-1
    % 步骤1：计算当前SOC对应的开路电压U_OCV（非线性映射，文献常用6阶多项式）
    soc = SOC(k);
    U_OCV(k) = 3.0 + 0.5*soc - 0.2*soc^2 + 0.1*soc^3 - 0.05*soc^4 + 0.01*soc^5;
    
    % 步骤2：更新极化电压U_R1、U_R2（离散化RC支路响应）
    U_R1(k+1) = U_R1(k) * exp(-Ts/tau1) + R1 * I(k) * (1 - exp(-Ts/tau1));
    U_R2(k+1) = U_R2(k) * exp(-Ts/tau2) + R2 * I(k) * (1 - exp(-Ts/tau2));
    
    % 步骤3：更新端电压U（二阶RC核心电压方程）
    U(k) = U_OCV(k) - R0*I(k) - U_R1(k) - U_R2(k);
    
    % 步骤4：更新SOC（安时积分法，离散化）
    SOC(k+1) = SOC(k) - (eta * abs(I(k)) * Ts) / (3600 * Q0);  % 3600为单位换算（s→h）
    
    % 防止SOC超出0~1范围
    if SOC(k+1) < 0
        SOC(k+1) = 0;
    end
end

% 补充最后一个点的OCV和端电压
U_OCV(N) = 3.0 + 0.5*SOC(N) - 0.2*SOC(N)^2 + 0.1*SOC(N)^3 - 0.05*SOC(N)^4 + 0.01*SOC(N)^5;
U(N) = U_OCV(N) - R0*I(N) - U_R1(N) - U_R2(N);

%% 4. 保存仿真数据（后续参数估计、模型验证直接调用）
save('电池仿真_纯净数据.mat', 't', 'I', 'SOC', 'U', 'U_OCV', 'U_R1', 'U_R2');
fprintf('纯净数据生成完成！保存为「电池仿真_纯净数据.mat」\n');

%% 5. 简单可视化（查看数据趋势）
figure('Position', [100, 100, 1000, 600]);
subplot(2,1,1);
plot(t/60, SOC*100, 'k-', 'LineWidth', 2);
xlabel('时间（min）'); ylabel('SOC（%）');
title('恒流放电SOC变化曲线（纯净仿真数据）');
grid on;

subplot(2,1,2);
plot(t/60, U, 'b-', 'LineWidth', 2);
xlabel('时间（min）'); ylabel('端电压（V）');
title('恒流放电端电压变化曲线（纯净仿真数据）');
grid on;