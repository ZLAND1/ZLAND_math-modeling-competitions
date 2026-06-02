import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

# 定义一个多目标优化问题
problem = get_problem("zdt1")

# 初始化 NSGA-II 算法
algorithm = NSGA2(pop_size=100)

# 运行优化
res = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               verbose=False)

# 输出结果
print("最优解:", res.X)
print("最优目标值:", res.F)