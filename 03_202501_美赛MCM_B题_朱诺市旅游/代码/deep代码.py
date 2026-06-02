# -*- coding: utf-8 -*-
"""
朱诺市可持续旅游多目标优化系统
目标：最大化经济收益 + 最小化环境成本 + 最大化社会效益
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================
# 1. 基准数据体系（2023年）
# ======================
class BaseData:
    """核心数据基准值（单位：百万美元）"""
    def __init__(self):
        # 经济系统
        self.revenue_base = 420.0    # 旅游总收入
        self.tax_rate_base = 0.08    # 基准税率
        
        # 环境系统
        self.env_cost_base = 13.5    # 碳排放成本
        self.glacier_cost = 4.0      # 冰川退缩成本
        
        # 社会系统
        self.housing_cost = 25.0     # 住房成本影响
        self.satisfaction_base = 0.50 # 满意度基准

# ======================
# 2. 多目标优化模型
# ======================
class SustainableTourismModel:
    def __init__(self):
        self.data = BaseData()
        
        # 优化参数范围
        self.bounds = {
            'visitors': (140, 165),   # 游客量（万人次）
            'tax_rate': (0.06, 0.10), # 税率
            'eco_invest': (0.35, 0.45) # 环保投入比例
        }
        
        # 动态权重系统
        self.weights = {
            'economic': 0.6,
            'environment': 0.3,
            'social': 0.1
        }
    
    def economic_model(self, V, T):
        """经济收益模型：包含边际递减效应"""
        base = self.data.revenue_base * (V/165)
        tax_effect = 1 + 0.7*T  # 税收弹性系数
        return base * tax_effect * np.exp(-0.005*(V-160))  # 160万后收益递减
    
    def environmental_model(self, V, E):
        """环境成本模型：包含治理效益"""
        base_cost = self.data.env_cost_base * (V/165)
        glacier_cost = self.data.glacier_cost * (1 + 0.002*(V-160))  # 超载加剧
        mitigation = 0.25 * E / 0.35  # 环保投入治理系数
        return (base_cost + glacier_cost) * (1 - mitigation)
    
    def social_model(self, V, E):
        """社会效益模型：平衡满意度与住房成本"""
        satisfaction = self.data.satisfaction_base + 0.15*(E/0.45) - 0.0008*(V-150)
        housing_impact = self.data.housing_cost * (V/150)**1.2
        return 80*satisfaction - housing_impact
    
    def objective(self, params):
        """综合目标函数"""
        V, T, E = params
        economic = self.economic_model(V, T)
        env_cost = self.environmental_model(V, E)
        social = self.social_model(V, E)
        return -(self.weights['economic']*economic 
                - self.weights['environment']*env_cost
                + self.weights['social']*social)
    
    def constraints(self, params):
        """动态约束系统"""
        V, T, E = params
        return [
            V <= 165,  # 游客量硬约束
            T <= 0.10, # 税率上限
            E >= 0.35 + 0.005*(V-140)  # 环保投入动态下限
        ]

# ======================
# 3. 贝叶斯优化引擎
# ======================
class OptimizationEngine:
    def __init__(self, model):
        self.model = model
        
    def create_study(self):
        study = optuna.create_study()
        study.optimize(self._objective, n_trials=500, n_jobs=-1)
        return study
    
    def _objective(self, trial):
        V = trial.suggest_float('V', *self.model.bounds['visitors'])
        T = trial.suggest_float('T', *self.model.bounds['tax_rate'])
        E = trial.suggest_float('E', *self.model.bounds['eco_invest'])
        
        params = [V, T, E]
        if not all(c >=0 for c in self.model.constraints(params)):
            return np.inf
        return self.model.objective(params)

# ======================
# 4. 可视化系统
# ======================
class ResultVisualizer:
    @staticmethod
    def plot_3d_space(study):
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')
        
        df = study.trials_dataframe()
        valid = df[df['state'] == 'COMPLETE']
        sc = ax.scatter(valid['params_V'], valid['params_T'], valid['params_E'],
                        c=-valid['value'], cmap='viridis', s=40, alpha=0.7)
        
        ax.set_xlabel('Visitors (万)', labelpad=12)
        ax.set_ylabel('Tax Rate', labelpad=12)
        ax.set_zlabel('Eco Investment', labelpad=12)
        plt.title("三维参数空间优化分布", y=1.0, pad=20)
        fig.colorbar(sc, label='综合效益值', pad=0.1)
        plt.show()
    
    @staticmethod
    def plot_pareto_front(study):
        fig, ax = plt.subplots(figsize=(10,6))
        costs = [-t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ax.plot(sorted(costs), 'b-o')
        ax.set_xlabel('试验次数')
        ax.set_ylabel('综合效益值')
        ax.set_title("帕累托前沿收敛趋势")
        ax.grid(True)
        plt.show()

# ======================
# 5. 执行与结果输出
# ======================
if __name__ == "__main__":
    # 初始化模型
    model = SustainableTourismModel()
    engine = OptimizationEngine(model)
    
    # 运行优化
    study = engine.create_study()
    
    # 输出最优结果
    best = study.best_params
    print(f"""
=== 最优可持续旅游策略 ===
游客量：{best['V']:.1f} 万人次
税率：{best['T']:.1%}
环保投入：{best['E']:.1%}
-------------------------
预期经济效益：{model.economic_model(best['V'], best['T']):.1f} 百万美元
环境治理成本：{model.environmental_model(best['V'], best['E']):.1f} 百万美元
社会综合效益：{model.social_model(best['V'], best['E']):.1f} 百万美元
综合得分：{-study.best_value:.1f}
""")
    
    # 可视化分析
    ResultVisualizer.plot_3d_space(study)
    ResultVisualizer.plot_pareto_front(study)