% 1. 数据准备（实验数据或Simulink仿真输出，此处以模拟数据为例，可替换为实际数据）
clear; clc; close all;
SOC = 0:0.01:1;  % 荷电状态（0~100%）
T_list = [-40, -20, 0, 25];  % 温度工况（℃）
U_matrix = zeros(length(T_list), length(SOC));  % 电压存储矩阵

% 模拟不同温度下的电压-SOC关系（基于磷酸铁锂电池OCV特性，拟合文献数据）
for i = 1:length(T_list)
    T = T_list(i);
    % 电压模型：考虑温度对OCV的修正，参考李心月论文式3-3
    U_OCV = 3.2 + 0.3*SOC - 0.1*SOC.^2 + 0.05*SOC.^3;  % 基础OCV-SOC曲线
    T_correction = 0.001*(25 - T)*SOC;  % 温度修正项（低温电压降低）
    U_matrix(i,:) = U_OCV + T_correction;
end

% 2. 绘制图表（文献风格：简洁、清晰、信息完整）
figure('Position', [100, 100, 800, 600]);  % 设置图窗大小
colors = ['r', 'g', 'b', 'k'];  % 文献常用单色（避免花哨）
linestyles = {'-', '--', '-.', ':'};  % 不同温度用不同线型区分

for i = 1:length(T_list)
    plot(SOC*100, U_matrix(i,:), ...
         'Color', colors(i), ...
         'LineStyle', linestyles{i}, ...
         'LineWidth', 2, ...  % 线条加粗（期刊要求1.5~2pt）
         'DisplayName', [num2str(T_list(i)) '℃']);
end

% 3. 图表样式优化（符合学术规范）
xlabel('SOC (%)', 'FontName', 'Times New Roman', 'FontSize', 12);  % 坐标轴标签（物理量+单位）
ylabel('端电压 (V)', 'FontName', 'Times New Roman', 'FontSize', 12);
title('不同温度下恒流放电电压-SOC曲线', 'FontName', '宋体', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontName', 'Times New Roman', 'FontSize', 10);  % 图例位置最优
grid on; grid minor;  % 显示网格（文献常用细网格）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);  % 坐标轴刻度字体
axis([0, 100, 2.5, 3.8]);  % 固定坐标轴范围（避免数据浮动导致对比不直观）

% 4. 导出高分辨率图片（期刊要求300dpi以上）
print('-dpng', '-r300', '电压SOC曲线_不同温度.png');