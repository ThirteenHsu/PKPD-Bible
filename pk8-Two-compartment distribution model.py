import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.integrate import solve_ivp, odeint
from sklearn.linear_model import LinearRegression 
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
objectives:
• To analyze a dataset exhibiting bi-phasic decline 双相下降
• To apply five different parameterizations of this behavior 5种参数估计的方法
• To specify the model as an integrated solution and a set of differential equations 模型的积分解和微分方程

"""

dose = 100 # μg iv bolus of compound X

def load_data(plot_data=True):
    # Original data
    data = {
        'Time (h)': [0.08, 0.24, 0.44, 0.73, 1.29, 1.65, 2.02, 2.50, 3.03, 3.47, 4.00, 5.00, 7.02, 11.02, 23.00, 29.02, 35.03, 47.22],
        'C_iv (μg·L⁻¹)': [1.80, 1.38, 1.17, 1.03, 0.96, 0.79, 0.83, 0.79, 0.72, 0.64, 0.67, 0.59, 0.52, 0.41, 0.24, 0.15, 0.12, 0.07],
    }

    df = pd.DataFrame(data)


    time_points = df['Time (h)'].values
    concentrations_iv = df['C_iv (μg·L⁻¹)'].values

    if plot_data:
        # 创建原始数据散点图
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, concentrations_iv, color='red', marker='o', linestyle='-', markersize=6, linewidth=1.5)
        plt.xlabel('Time (min)', fontsize=12)
        plt.ylabel('Concentrations (μg·L⁻¹)', fontsize=12)
        plt.yscale('log')
        plt.yticks([3, 1, 0.1, 0.03], labels=['3', '1', '0.1', '0.03'])
        plt.grid(True, which="both", ls="-", color='0.7', alpha=0.3)
        plt.show()
    return time_points, concentrations_iv

def traditional_bi_phase_model(time_points, concentration, plot=False):
    """
    传统的双相模型：
    1. 定义原方程 Cp = A * exp(-α*t) + B * exp(-β*t)
    2. 使用curve_fit进行参数估计，得到A, α, B, β
    """
    # 定义原方程 Cp = A * exp(-α*t) + B * exp(-β*t)
    def bi_phase_model(t, A, alpha, B, beta):
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t)

    # 定义多组初始参数猜测值 [A, α, B, β]
    initial_guess_sets = [
        [50.0, 1.0, 50.0, 0.1],      # 当前使用的初始值
        [10.0, 0.5, 10.0, 0.05],     # 较小的初始值
        [100.0, 2.0, 100.0, 0.5],    # 较大的初始值
        [25.0, 1.5, 75.0, 0.2],       # 随机组合1
        [80.0, 0.8, 20.0, 0.15]       # 随机组合2
    ]
    
    # 设置参数边界，确保所有参数为正值
    bounds = ([0, 0, 0, 0], [1000, 10, 1000, 5])

    # 高级参数估计方法集合
    optimization_results = {}
    
    # 方法1: 差分进化算法 (Differential Evolution) - 全局优化
    print("方法1: 差分进化算法 (Differential Evolution)")
    def objective_function(params):
        A, alpha, B, beta = params
        try:
            predicted = bi_phase_model(time_points, A, alpha, B, beta)
            residuals = concentration - predicted
            return np.sum(residuals**2)  # 最小化残差平方和
        except:
            return np.inf
    
    try:
        # 使用差分进化算法进行全局优化
        bounds_de = [(0.1, 1000), (0.01, 10), (0.1, 1000), (0.001, 5)]
        result_de = differential_evolution(objective_function, bounds_de, 
                                         seed=42, maxiter=1000, popsize=15,
                                         atol=1e-8, tol=1e-8)
        
        if result_de.success:
            popt_de = result_de.x
            # 使用差分进化结果作为初始值进行精细化拟合
            popt_refined, pcov_refined = curve_fit(bi_phase_model, time_points, concentration,
                                                 p0=popt_de, bounds=bounds, maxfev=10000)
            
            # 计算拟合质量指标
            ss_res = np.sum((concentration - bi_phase_model(time_points, *popt_refined)) ** 2)
            ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
            r_squared_de = 1 - (ss_res / ss_tot)
            
            # 计算AIC
            n = len(concentration)
            mse = ss_res / n
            aic_de = n * np.log(mse) + 2 * 4  # 4个参数
            
            param_errors_de = np.sqrt(np.diag(pcov_refined))
            rse_percent_de = (param_errors_de / np.abs(popt_refined)) * 100
            
            optimization_results['differential_evolution'] = {
                'params': popt_refined,
                'pcov': pcov_refined,
                'r_squared': r_squared_de,
                'aic': aic_de,
                'param_errors': param_errors_de,
                'rse_percent': rse_percent_de,
                'method': 'Differential Evolution + Refinement'
            }
            print(f"  差分进化算法成功 - R²: {r_squared_de:.4f}, AIC: {aic_de:.2f}")
        else:
            print("  差分进化算法失败")
    except Exception as e:
        print(f"  差分进化算法出错: {e}")
    
    # 方法2: 改进的多起点优化 (Enhanced Multi-start)
    print("方法2: 改进的多起点优化")
    
    # 使用拉丁超立方采样生成更好的初始值
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=4, seed=42)
    n_starts = 20
    sample = sampler.random(n=n_starts)
    
    # 将样本映射到参数边界
    bounds_array = np.array([[0.1, 1000], [0.01, 10], [0.1, 1000], [0.001, 5]])
    initial_guess_lhs = qmc.scale(sample, bounds_array[:, 0], bounds_array[:, 1])
    
    best_results = []
    
    for i, initial_guess in enumerate(initial_guess_lhs):
        try:
            popt, pcov = curve_fit(bi_phase_model, time_points, concentration,
                                  p0=initial_guess, bounds=bounds, maxfev=10000)
            
            # 计算拟合质量
            ss_res = np.sum((concentration - bi_phase_model(time_points, *popt)) ** 2)
            ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # 计算AIC
            n = len(concentration)
            mse = ss_res / n
            aic = n * np.log(mse) + 2 * 4
            
            param_errors = np.sqrt(np.diag(pcov))
            rse_percent = (param_errors / np.abs(popt)) * 100
            
            best_results.append({
                'params': popt,
                'pcov': pcov,
                'r_squared': r_squared,
                'aic': aic,
                'param_errors': param_errors,
                'rse_percent': rse_percent,
                'initial_guess': initial_guess
            })
            
        except Exception as e:
            continue
    
    if best_results:
        # 选择R²最高的结果
        best_multistart = max(best_results, key=lambda x: x['r_squared'])
        optimization_results['multistart_lhs'] = {
            **best_multistart,
            'method': 'Latin Hypercube Multi-start'
        }
        print(f"  多起点优化成功 ({len(best_results)}/{n_starts}次成功) - 最佳R²: {best_multistart['r_squared']:.4f}")
    else:
        print("  多起点优化全部失败")
    
    # 方法3: 贝叶斯优化 (使用scipy.optimize.minimize的多种算法)
    print("方法3: 多算法集成优化")
    
    algorithms = ['L-BFGS-B', 'TNC', 'SLSQP']
    for alg in algorithms:
        try:
            # 使用最佳多起点结果作为初始值
            if 'multistart_lhs' in optimization_results:
                initial_guess = optimization_results['multistart_lhs']['params']
            else:
                initial_guess = [50.0, 1.0, 50.0, 0.1]
            
            result = minimize(objective_function, initial_guess, 
                            method=alg, bounds=bounds_array)
            
            if result.success:
                popt_min, pcov_min = curve_fit(bi_phase_model, time_points, concentration,
                                             p0=result.x, bounds=bounds, maxfev=10000)
                
                ss_res = np.sum((concentration - bi_phase_model(time_points, *popt_min)) ** 2)
                ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
                r_squared_min = 1 - (ss_res / ss_tot)
                
                n = len(concentration)
                mse = ss_res / n
                aic_min = n * np.log(mse) + 2 * 4
                
                param_errors_min = np.sqrt(np.diag(pcov_min))
                rse_percent_min = (param_errors_min / np.abs(popt_min)) * 100
                
                optimization_results[f'minimize_{alg.lower()}'] = {
                    'params': popt_min,
                    'pcov': pcov_min,
                    'r_squared': r_squared_min,
                    'aic': aic_min,
                    'param_errors': param_errors_min,
                    'rse_percent': rse_percent_min,
                    'method': f'Minimize ({alg})'
                }
                print(f"  {alg}算法成功 - R²: {r_squared_min:.4f}")
        except Exception as e:
            print(f"  {alg}算法失败: {e}")
    
    # 选择最佳结果
    if not optimization_results:
        print("所有优化方法均失败，无法进行参数估计。")
        return None, None, None, None, None, None, None
    
    # 根据R²选择最佳结果
    best_method = max(optimization_results.keys(), 
                     key=lambda k: optimization_results[k]['r_squared'])
    best_result = optimization_results[best_method]
    
    best_popt = best_result['params']
    best_pcov = best_result['pcov']
    best_r_squared = best_result['r_squared']
    best_param_errors = best_result['param_errors']
    best_rse_percent = best_result['rse_percent']
    
    # 计算拟合曲线
    t_fit = np.linspace(time_points.min(), time_points.max(), 100)
    c_fit = bi_phase_model(t_fit, *best_popt)
    best_t_fit = t_fit
    best_c_fit = c_fit
    
    # 计算参数相关性矩阵
    correlation_matrix = np.corrcoef(best_pcov)
    
    print(f"\n最佳优化方法: {best_result['method']}")
    print("所有方法比较:")
    for method, result in optimization_results.items():
        print(f"  {result['method']}: R² = {result['r_squared']:.4f}, AIC = {result['aic']:.2f}")
    
    # 添加参数相关性分析
    param_names = ['A', 'α', 'B', 'β']
    print(f"\n参数相关性矩阵 (最佳方法: {best_result['method']}):")
    for i, name1 in enumerate(param_names):
        for j, name2 in enumerate(param_names):
            if i <= j:
                corr = correlation_matrix[i, j] if i < len(correlation_matrix) and j < len(correlation_matrix[0]) else 0
                print(f"  {name1}-{name2}: {corr:.3f}")
    
    if best_popt is None:
        print("所有优化方法均失败，无法进行参数估计。")
        return None, None, None, None, None, None, None

    A, alpha, B, beta = best_popt

    # 打印最佳参数估计结果
    print(f"\nTraditional Bi-phase Model - 最佳估计参数:")
    print(f"  A = {A:.4f} ± {best_param_errors[0]:.4f} μg·L⁻¹ (RSE: {best_rse_percent[0]:.1f}%)")
    print(f"  α = {alpha:.4f} ± {best_param_errors[1]:.4f} h⁻¹ (RSE: {best_rse_percent[1]:.1f}%)")
    print(f"  B = {B:.4f} ± {best_param_errors[2]:.4f} μg·L⁻¹ (RSE: {best_rse_percent[2]:.1f}%)")
    print(f"  β = {beta:.4f} ± {best_param_errors[3]:.4f} h⁻¹ (RSE: {best_rse_percent[3]:.1f}%)")
    print(f"R² = {best_r_squared:.4f}")
    
    if plot:
        # 创建综合诊断图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1: 拟合结果比较
        ax1.plot(time_points, concentration, 'ro', label='Observed data', markersize=6)
        ax1.plot(best_t_fit, best_c_fit, 'b-', label=f'Best Fit ({best_result["method"]})', linewidth=2)
        
        # 如果有多个结果，显示前3个最佳结果
        colors = ['g--', 'm--', 'c--']
        sorted_methods = sorted(optimization_results.keys(), 
                              key=lambda k: optimization_results[k]['r_squared'], reverse=True)
        for i, method in enumerate(sorted_methods[1:4]):  # 显示第2-4名
            result = optimization_results[method]
            c_fit_temp = bi_phase_model(best_t_fit, *result['params'])
            ax1.plot(best_t_fit, c_fit_temp, colors[i], 
                    label=f'{result["method"]} (R²={result["r_squared"]:.3f})', 
                    linewidth=1, alpha=0.7)
        
        ax1.set_xlabel('Time (h)', fontsize=12)
        ax1.set_ylabel('Concentration (μg·L⁻¹)', fontsize=12)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Model Fitting Comparison')
        
        # 子图2: 残差分析
        residuals = concentration - bi_phase_model(time_points, *best_popt)
        ax2.scatter(time_points, residuals, color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (h)', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Residuals vs Time')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 方法性能比较
        methods = [result['method'] for result in optimization_results.values()]
        r_squared_values = [result['r_squared'] for result in optimization_results.values()]
        aic_values = [result['aic'] for result in optimization_results.values()]
        
        ax3_twin = ax3.twinx()
        bars1 = ax3.bar(range(len(methods)), r_squared_values, alpha=0.7, color='blue', label='R²')
        bars2 = ax3_twin.bar([x+0.4 for x in range(len(methods))], aic_values, alpha=0.7, color='red', width=0.4, label='AIC')
        
        ax3.set_xlabel('Optimization Method', fontsize=12)
        ax3.set_ylabel('R²', color='blue', fontsize=12)
        ax3_twin.set_ylabel('AIC', color='red', fontsize=12)
        ax3.set_title('Method Performance Comparison')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([m.split('(')[0].strip() for m in methods], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 参数不确定性
        param_names = ['A', 'α', 'B', 'β']
        param_values = best_popt
        param_errors = best_param_errors
        
        ax4.errorbar(range(len(param_names)), param_values, yerr=param_errors, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax4.set_xlabel('Parameters', fontsize=12)
        ax4.set_ylabel('Parameter Values', fontsize=12)
        ax4.set_title('Parameter Estimates with Uncertainty')
        ax4.set_xticks(range(len(param_names)))
        ax4.set_xticklabels(param_names)
        ax4.grid(True, alpha=0.3)
        
        # 在每个参数点上标注RSE%
        for i, (val, err, rse) in enumerate(zip(param_values, param_errors, best_rse_percent)):
            ax4.annotate(f'RSE: {rse:.1f}%', (i, val + err), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # 额外的诊断信息
        print(f"\n=== 诊断信息 ===")
        print(f"数据点数量: {len(concentration)}")
        print(f"参数数量: {len(best_popt)}")
        print(f"自由度: {len(concentration) - len(best_popt)}")
        print(f"平均相对标准误差: {np.mean(best_rse_percent):.1f}%")
        
        # 检查参数的合理性
        A, alpha, B, beta = best_popt
        print(f"\n=== 参数合理性检查 ===")
        print(f"初始浓度 (A+B): {A+B:.3f} μg·L⁻¹")
        print(f"快速相半衰期: {np.log(2)/max(alpha, beta):.2f} h")
        print(f"慢速相半衰期: {np.log(2)/min(alpha, beta):.2f} h")
        print(f"分布相/消除相比例: {max(A,B)/min(A,B):.2f}")
        
        # 模型选择建议
        if best_r_squared > 0.99:
            print(f"\n=== 模型质量评估 ===")
            print("✓ 拟合质量优秀 (R² > 0.99)")
        elif best_r_squared > 0.95:
            print(f"\n=== 模型质量评估 ===")
            print("✓ 拟合质量良好 (R² > 0.95)")
        else:
            print(f"\n=== 模型质量评估 ===")
            print("⚠ 拟合质量需要改进 (R² < 0.95)")
            
        if np.mean(best_rse_percent) < 20:
            print("✓ 参数精度良好 (平均RSE < 20%)")
        else:
            print("⚠ 参数精度较低 (平均RSE > 20%)")
            
        # 最高相关性检查
        if len(correlation_matrix) > 1:
            max_corr = 0
            max_pair = ""
            for i, name1 in enumerate(param_names):
                for j, name2 in enumerate(param_names):
                    if i < j and i < len(correlation_matrix) and j < len(correlation_matrix[0]):
                        corr = abs(correlation_matrix[i, j])
                        if corr > max_corr:
                            max_corr = corr
                            max_pair = f"{name1}-{name2}"
            
            if max_corr > 0.9:
                print(f"⚠ 高参数相关性: {max_pair} ({max_corr:.3f})")
            else:
                print("✓ 参数相关性可接受")
    
    return best_popt, best_pcov, best_r_squared, best_t_fit, best_c_fit, best_param_errors, best_rse_percent

def takada_distribution_model(time_points, concentration, D_iv, plot=False):
    """
    takada，asada和colburn提出的分布容积与时间的关系:
    1. 定义原方程 Cp = D_iv* exp(-β*t)/(Vc + Vt)
    2. 组织隔室分布容积是时间的函数：Vt = (Vmax * t)/(kd + t)
    3. 使用curve_fit进行参数估计，得到β, Vmax, kd, Vc (D_iv=100μg为固定值)
    """
    # 定义原方程 Cp = D_iv* exp(-β*t)/(Vc + Vt)，D_iv为固定值
    def takada_model(t, beta, Vmax, kd, Vc):
        Vt = (Vmax * t)/(kd + t)
        return D_iv * np.exp(-beta * t)/(Vc + Vt)
    
    # 设置初始参数猜测值 [β, Vmax, kd, Vc]
    initial_guess = [0.1, 50.0, 10.0, 20.0]  # 基于生理参数范围
    
    # 设置参数边界，确保所有参数为正值
    bounds = ([0.01, 1.0, 0.1, 1.0], [5.0, 500.0, 100.0, 200.0])
    
    try:
        # 使用curve_fit进行参数估计
        popt, pcov = curve_fit(takada_model, time_points, concentration,
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        beta, Vmax, kd, Vc = popt

        # 计算参数标准误差和置信区间
        param_errors = np.sqrt(np.diag(pcov))
        from scipy.stats import t
        dof = len(time_points) - len(popt)  # 自由度
        t_val = t.ppf(0.975, dof)  # 95%置信区间
        
        # 计算参数相对标准误差 (RSE%)
        rse_percent = (param_errors / np.abs(popt)) * 100

        # 打印参数估计结果
        print(f"Takada Distribution Model - Estimated parameters:")
        print(f"  D_iv = {D_iv:.4f} μg (fixed)")
        print(f"  β = {beta:.4f} ± {param_errors[0]:.4f} h⁻¹ (RSE: {rse_percent[0]:.1f}%)")
        print(f"  Vmax = {Vmax:.4f} ± {param_errors[1]:.4f} L·h⁻¹ (RSE: {rse_percent[1]:.1f}%)")
        print(f"  kd = {kd:.4f} ± {param_errors[2]:.4f} μg·L⁻¹ (RSE: {rse_percent[2]:.1f}%)")
        print(f"  Vc = {Vc:.4f} ± {param_errors[3]:.4f} L (RSE: {rse_percent[3]:.1f}%)")
        
        # 计算拟合曲线
        t_fit = np.linspace(time_points.min(), time_points.max(), 100)
        c_fit = takada_model(t_fit, *popt)
        
        # 计算R²
        ss_res = np.sum((concentration - takada_model(time_points, *popt)) ** 2)
        ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² = {r_squared:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, concentration, 'ro', label='Observed data', markersize=6)
            plt.plot(t_fit, c_fit, 'g-', label='Takada Distribution Model', linewidth=2)
            plt.xlabel('Time (h)', fontsize=12)
            plt.ylabel('Concentration (μg·L⁻¹)', fontsize=12)
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Takada Distribution Model Fitting')
            plt.show()
        
        return popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent
        
    except Exception as e:
        print(f"Takada Distribution Model - Parameter estimation failed: {e}")
        return None, None, None, None, None, None, None

def cplburn_distribution_model(time_points, concentration, D_iv, plot=False):
    """
    colburn提出的分布容积与时间的关系：
    1. 定义原方程 Cp = D_iv* exp(-β*t)/(Vc + Vt)
    2. 组织隔室分布容积是时间的函数：Vt = Vmax * (1 - exp(-kv * t))
    3. 使用curve_fit进行参数估计，得到β, Vmax, kv, Vc (D_iv=100μg为固定值)
    """
    # 定义原方程 Cp = D_iv* exp(-β*t)/(Vc + Vt)，D_iv为固定值
    def cplburn_model(t, beta, Vmax, kv, Vc):
        Vt = Vmax * (1 - np.exp(-kv * t))
        return D_iv * np.exp(-beta * t)/(Vc + Vt)
    
    # 设置初始参数猜测值 [β, Vmax, kv, Vc]
    initial_guess = [0.1, 50.0, 1.0, 20.0]  # 基于生理参数范围
    
    # 设置参数边界，确保所有参数为正值
    bounds = ([0.01, 1.0, 0.01, 1.0], [5.0, 500.0, 10.0, 200.0])
    
    try:
        # 使用curve_fit进行参数估计
        popt, pcov = curve_fit(cplburn_model, time_points, concentration,
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        beta, Vmax, kv, Vc = popt

        # 计算参数标准误差和置信区间
        param_errors = np.sqrt(np.diag(pcov))
        from scipy.stats import t
        dof = len(time_points) - len(popt)  # 自由度
        t_val = t.ppf(0.975, dof)  # 95%置信区间
        
        # 计算参数相对标准误差 (RSE%)
        rse_percent = (param_errors / np.abs(popt)) * 100

        # 打印参数估计结果
        print(f"Colburn Distribution Model - Estimated parameters:")
        print(f"  D_iv = {D_iv:.4f} μg (fixed)")
        print(f"  β = {beta:.4f} ± {param_errors[0]:.4f} h⁻¹ (RSE: {rse_percent[0]:.1f}%)")
        print(f"  Vmax = {Vmax:.4f} ± {param_errors[1]:.4f} L·h⁻¹ (RSE: {rse_percent[1]:.1f}%)")
        print(f"  kv = {kv:.4f} ± {param_errors[2]:.4f} h⁻¹ (RSE: {rse_percent[2]:.1f}%)")
        print(f"  Vc = {Vc:.4f} ± {param_errors[3]:.4f} L (RSE: {rse_percent[3]:.1f}%)")
        
        # 计算拟合曲线
        t_fit = np.linspace(time_points.min(), time_points.max(), 100)
        c_fit = cplburn_model(t_fit, *popt)
        
        # 计算R²
        ss_res = np.sum((concentration - cplburn_model(time_points, *popt)) ** 2)
        ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² = {r_squared:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, concentration, 'ro', label='Observed data', markersize=6)
            plt.plot(t_fit, c_fit, 'm-', label='Colburn Distribution Model', linewidth=2)
            plt.xlabel('Time (h)', fontsize=12)
            plt.ylabel('Concentration (μg·L⁻¹)', fontsize=12)
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Colburn Distribution Model Fitting')
            plt.show()
        
        return popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent
        
    except Exception as e:
        print(f"Colburn Distribution Model - Parameter estimation failed: {e}")
        return None, None, None, None, None, None, None

def reparameter_cl_method(time_points, concentration, D_iv, plot=False):
    """
    对传统的二室模型进行重参数化，原因在于：
        1. 模型的可识别问题：当不同的参数组合可以产生相同的结果，这些参数就是不可识别的；
        2. 参数生理意义的模糊性：传统的二室模型的参数的生理意义不如CL和V等参数那么明确；
        
    操作步骤：
        1. 定义原方程 Cp = a * (D_iv/Cl - B/β) * exp(-a * t) + B * exp(-β * t)
        2. 使用curve_fit进行参数估计, 得到β, Cl, a, B
    """
    # 定义原方程 Cp = a * (D_iv/Cl - B/β) * exp(-a * t) + B * exp(-β * t)
    def reparameter_model(t, beta, Cl, a, B):
        return a * (D_iv/Cl - B/beta) * np.exp(-a * t) + B * np.exp(-beta * t)
    
    # 设置初始参数猜测值 [β, Cl, a, B]
    initial_guess = [0.1, 10.0, 1.0, 0.5]  # 基于生理参数范围
    
    # 设置参数边界，确保所有参数为正值
    bounds = ([0.01, 1.0, 0.01, 0.01], [5.0, 100.0, 10.0, 50.0])
    
    # 使用curve_fit进行参数估计，增加最大迭代次数
    try:
        popt, pcov = curve_fit(reparameter_model, time_points, concentration, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        beta, Cl, a, B = popt
        
        # 计算参数标准误差和置信区间
        param_errors = np.sqrt(np.diag(pcov))
        from scipy.stats import t
        dof = len(time_points) - len(popt)  # 自由度
        t_val = t.ppf(0.975, dof)  # 95%置信区间
        
        # 计算参数相对标准误差 (RSE%)
        rse_percent = (param_errors / np.abs(popt)) * 100
        
        # 打印参数估计结果
        print(f"Reparameterized CL Model - Estimated parameters:")
        print(f"  D_iv = {D_iv:.4f} μg (fixed)")
        print(f"  β = {beta:.4f} ± {param_errors[0]:.4f} h⁻¹ (RSE: {rse_percent[0]:.1f}%)")
        print(f"  Cl = {Cl:.4f} ± {param_errors[1]:.4f} L·h⁻¹ (RSE: {rse_percent[1]:.1f}%)")
        print(f"  a = {a:.4f} ± {param_errors[2]:.4f} h⁻¹ (RSE: {rse_percent[2]:.1f}%)")
        print(f"  B = {B:.4f} ± {param_errors[3]:.4f} μg·L⁻¹ (RSE: {rse_percent[3]:.1f}%)")
        
        # 计算拟合曲线
        t_fit = np.linspace(time_points.min(), time_points.max(), 100)
        c_fit = reparameter_model(t_fit, *popt)
        
        # 计算R²
        ss_res = np.sum((concentration - reparameter_model(time_points, *popt)) ** 2)
        ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² = {r_squared:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, concentration, 'ro', label='Observed data', markersize=6)
            plt.plot(t_fit, c_fit, 'c-', label='Reparameterized CL Model', linewidth=2)
            plt.xlabel('Time (h)', fontsize=12)
            plt.ylabel('Concentration (μg·L⁻¹)', fontsize=12)
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Reparameterized CL Model Fitting')
            plt.show()
        
        return popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent
        
    except RuntimeError as e:
        print(f"Reparameterized CL Model - Parameter estimation failed: {e}")
        print("Using default parameters for demonstration")
        return None, None, None, None, None, None, None

def two_compartment_model_ode_fit(t, CL, CLd, Vc, Vt):
    """
    用于curve_fit的二室模型ODE求解函数
    标准二室模型微分方程：
    - 中央室：Vc * dCp/dt = In - CL * Cp - CLd * Cp + CLd * Ct
    - 外周室：Vt * dCt/dt = CLd * Cp - CLd * Ct
    对于IV bolus给药，In在t=0时为dose，t>0时为0（通过初始条件处理）
    """
    def model_ode(y, t):
        Cp, Ct = y
        # 对于IV bolus，没有持续输入，In = 0
        dCp_dt = (0 - CL * Cp - CLd * Cp + CLd * Ct) / Vc
        dCt_dt = (CLd * Cp - CLd * Ct) / Vt
        return [dCp_dt, dCt_dt]
    
    # 初始条件：静脉注射后，所有药物都在中央室
    # 初始浓度 = dose / Vc，外周室初始浓度为0
    y0 = [dose / Vc, 0]
    
    # 求解ODE
    sol = odeint(model_ode, y0, t)
    
    # 返回血浆浓度
    return sol[:, 0]

def differential_equation_method(time_points, concentration, D_iv, plot=False):
    """
    微分方程方法:
    1. 定义微分方程
        -中央室的方程：Vc * dCp/dt = D_iv - CL * Cp - CLd * Cp + CLd * Ct
        -组织隔室的方程：Vt * dCt/dt = CLd * Cp - CLd * Ct
    2. 使用curve_fit进行参数估计, 得到CL, CLd, Vc, Vt
    """
    # 设置初始参数猜测值
    initial_guess = [1.0, 0.5, 10.0, 5.0]  # CL, CLd, Vc, Vt
    
    # 设置参数边界，确保所有参数为正值，设置合理的上界
    bounds = ([0.01, 0.01, 0.01, 0.01], [100.0, 50.0, 200.0, 200.0])
    
    # 使用curve_fit进行参数估计
    try:
        popt, pcov = curve_fit(two_compartment_model_ode_fit, time_points, concentration, 
                              p0=initial_guess, maxfev=5000, bounds=bounds)
        CL, CLd, Vc, Vt = popt
        
        # 计算参数标准误差和置信区间
        param_errors = np.sqrt(np.diag(pcov))
        from scipy.stats import t
        dof = len(time_points) - len(popt)  # 自由度
        t_val = t.ppf(0.975, dof)  # 95%置信区间
        
        # 计算参数相对标准误差 (RSE%)
        rse_percent = (param_errors / np.abs(popt)) * 100
        
        # 打印参数估计结果
        print(f"Differential Equation Model - Estimated parameters:")
        print(f"  CL = {CL:.4f} ± {param_errors[0]:.4f} L·h⁻¹ (RSE: {rse_percent[0]:.1f}%)")
        print(f"  CLd = {CLd:.4f} ± {param_errors[1]:.4f} L·h⁻¹ (RSE: {rse_percent[1]:.1f}%)")
        print(f"  Vc = {Vc:.4f} ± {param_errors[2]:.4f} L (RSE: {rse_percent[2]:.1f}%)")
        print(f"  Vt = {Vt:.4f} ± {param_errors[3]:.4f} L (RSE: {rse_percent[3]:.1f}%)")
        
        # 计算拟合曲线
        t_fit = np.linspace(time_points.min(), time_points.max(), 100)
        c_fit = two_compartment_model_ode_fit(t_fit, *popt)
        
        # 计算R²
        ss_res = np.sum((concentration - two_compartment_model_ode_fit(time_points, *popt)) ** 2)
        ss_tot = np.sum((concentration - np.mean(concentration)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² = {r_squared:.4f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, concentration, 'ro', label='Observed data', markersize=6)
            plt.plot(t_fit, c_fit, 'orange', label='Differential Equation Model', linewidth=2)
            plt.xlabel('Time (h)', fontsize=12)
            plt.ylabel('Concentration (μg·L⁻¹)', fontsize=12)
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Differential Equation Model Fitting')
            plt.show()
        
        return popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent
        
    except Exception as e:
        print(f"Differential Equation Model - Parameter estimation failed: {e}")
        print("Using default parameters for demonstration")
        return None, None, None, None, None, None, None

def calculate_aic(n, mse, num_params):
    """计算AIC (Akaike Information Criterion)"""
    return n * np.log(mse) + 2 * num_params

def compare_all_models(time_points, concentrations_iv, D_iv=100, plot_individual=False, plot_comparison=True):
    """
    比较所有五种模型的拟合效果
    """
    print("="*80)
    print("五种二室模型参数估计方法的比较分析")
    print("="*80)
    
    models = {}
    colors = ['b-', 'g-', 'm-', 'c-', 'orange']
    labels = ['Traditional Bi-phase', 'Takada Distribution', 'Colburn Distribution', 
              'Reparameterized CL', 'Differential Equation']
    
    # 运行所有模型
    print("\n1. 传统双相模型 (Traditional Bi-phase Model)")
    print("-" * 50)
    result1 = traditional_bi_phase_model(time_points, concentrations_iv, plot=plot_individual)
    if result1[0] is not None:
        models['Traditional'] = result1
    
    print("\n2. Takada分布模型 (Takada Distribution Model)")
    print("-" * 50)
    result2 = takada_distribution_model(time_points, concentrations_iv, D_iv, plot=plot_individual)
    if result2[0] is not None:
        models['Takada'] = result2
    
    print("\n3. Colburn分布模型 (Colburn Distribution Model)")
    print("-" * 50)
    result3 = cplburn_distribution_model(time_points, concentrations_iv, D_iv, plot=plot_individual)
    if result3[0] is not None:
        models['Colburn'] = result3
    
    print("\n4. 重参数化CL模型 (Reparameterized CL Model)")
    print("-" * 50)
    result4 = reparameter_cl_method(time_points, concentrations_iv, D_iv, plot=plot_individual)
    if result4[0] is not None:
        models['Reparameterized'] = result4
    
    print("\n5. 微分方程模型 (Differential Equation Model)")
    print("-" * 50)
    result5 = differential_equation_method(time_points, concentrations_iv, D_iv, plot=plot_individual)
    if result5[0] is not None:
        models['Differential'] = result5
    
    # 模型比较分析
    print("\n" + "="*80)
    print("模型比较分析 (Model Comparison Analysis)")
    print("="*80)
    
    comparison_data = []
    for i, (name, result) in enumerate(models.items()):
        # 更新解包以适应新的返回值格式
        if len(result) == 7:  # 新格式：包含param_errors和rse_percent
            popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent = result
        else:  # 旧格式：兼容性处理
            popt, pcov, r_squared, t_fit, c_fit = result
            param_errors = None
            rse_percent = None
        
        # 计算MSE
        if name == 'Traditional':
            def model_func(t, *params):
                A, alpha, B, beta = params
                return A * np.exp(-alpha * t) + B * np.exp(-beta * t)
        elif name == 'Takada':
            def model_func(t, *params):
                beta, Vmax, kd, Vc = params
                Vt = (Vmax * t)/(kd + t)
                return D_iv * np.exp(-beta * t)/(Vc + Vt)
        elif name == 'Colburn':
            def model_func(t, *params):
                beta, Vmax, kv, Vc = params
                Vt = Vmax * (1 - np.exp(-kv * t))
                return D_iv * np.exp(-beta * t)/(Vc + Vt)
        elif name == 'Reparameterized':
            def model_func(t, *params):
                beta, Cl, a, B = params
                return a * (D_iv/Cl - B/beta) * np.exp(-a * t) + B * np.exp(-beta * t)
        elif name == 'Differential':
            def model_func(t, *params):
                return two_compartment_model_ode_fit(t, *params)
        
        predicted = model_func(time_points, *popt)
        mse = np.mean((concentrations_iv - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # 计算AIC
        num_params = len(popt)
        aic = calculate_aic(len(time_points), mse, num_params)
        
        # 计算平均RSE%（如果可用）
        avg_rse = np.mean(rse_percent) if rse_percent is not None else None
        
        comparison_data.append({
            'Model': name,
            'R²': r_squared,
            'MSE': mse,
            'RMSE': rmse,
            'AIC': aic,
            'Parameters': num_params,
            'Avg_RSE': avg_rse
        })
    
    # 打印比较表格
    print(f"{'Model':<15} {'R²':<8} {'MSE':<10} {'RMSE':<10} {'AIC':<10} {'Avg RSE%':<10} {'Params':<8}")
    print("-" * 85)
    for data in comparison_data:
        rse_str = f"{data['Avg_RSE']:.1f}" if data['Avg_RSE'] is not None else "N/A"
        print(f"{data['Model']:<15} {data['R²']:<8.4f} {data['MSE']:<10.6f} {data['RMSE']:<10.4f} {data['AIC']:<10.2f} {rse_str:<10} {data['Parameters']:<8}")
    
    # 找出最佳模型
    best_r2 = max(comparison_data, key=lambda x: x['R²'])
    best_aic = min(comparison_data, key=lambda x: x['AIC'])
    
    print(f"\n最佳拟合度 (R²): {best_r2['Model']} (R² = {best_r2['R²']:.4f})")
    print(f"最佳AIC: {best_aic['Model']} (AIC = {best_aic['AIC']:.2f})")
    
    # 参数精确度分析
    print(f"\n参数估计精确度分析:")
    print("-" * 50)
    valid_rse_models = [data for data in comparison_data if data['Avg_RSE'] is not None]
    if valid_rse_models:
        best_precision = min(valid_rse_models, key=lambda x: x['Avg_RSE'])
        worst_precision = max(valid_rse_models, key=lambda x: x['Avg_RSE'])
        print(f"最佳参数精确度: {best_precision['Model']} (平均RSE: {best_precision['Avg_RSE']:.1f}%)")
        print(f"最差参数精确度: {worst_precision['Model']} (平均RSE: {worst_precision['Avg_RSE']:.1f}%)")
        
        # RSE分类
        excellent = [d for d in valid_rse_models if d['Avg_RSE'] < 10]
        good = [d for d in valid_rse_models if 10 <= d['Avg_RSE'] < 20]
        acceptable = [d for d in valid_rse_models if 20 <= d['Avg_RSE'] < 30]
        poor = [d for d in valid_rse_models if d['Avg_RSE'] >= 30]
        
        print(f"\n参数精确度分级:")
        print(f"  优秀 (RSE < 10%): {[d['Model'] for d in excellent]}")
        print(f"  良好 (10% ≤ RSE < 20%): {[d['Model'] for d in good]}")
        print(f"  可接受 (20% ≤ RSE < 30%): {[d['Model'] for d in acceptable]}")
        print(f"  较差 (RSE ≥ 30%): {[d['Model'] for d in poor]}")
    
    # 参数相关性分析
    print(f"\n参数相关性分析:")
    print("-" * 50)
    for name, result in models.items():
        if len(result) == 7:  # 新格式：包含param_errors和rse_percent
            popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent = result
            
            # 计算相关系数矩阵
            correlation_matrix = np.corrcoef(pcov)
            
            # 获取参数名称
            if name == 'Traditional':
                param_names = ['A', 'α', 'B', 'β']
            elif name == 'Takada':
                param_names = ['β', 'Vmax', 'kd', 'Vc']
            elif name == 'Colburn':
                param_names = ['β', 'Vmax', 'kv', 'Vc']
            elif name == 'Reparameterized':
                param_names = ['β', 'Cl', 'a', 'B']
            elif name == 'Differential':
                param_names = ['CL', 'CLd', 'Vc', 'Vt']
            else:
                param_names = [f'P{i+1}' for i in range(len(popt))]
            
            print(f"\n{name} Model 参数相关性:")
            
            # 找出高相关性的参数对 (|r| > 0.8)
            high_corr_pairs = []
            n_params = len(popt)
            for i in range(n_params):
                for j in range(i+1, n_params):
                    corr_coef = correlation_matrix[i, j]
                    if abs(corr_coef) > 0.8:
                        high_corr_pairs.append((param_names[i], param_names[j], corr_coef))
            
            if high_corr_pairs:
                print("  高相关性参数对 (|r| > 0.8):")
                for p1, p2, corr in high_corr_pairs:
                    print(f"    {p1} - {p2}: r = {corr:.3f}")
            else:
                print("  无高相关性参数对 (|r| > 0.8)")
            
            # 计算条件数 (condition number) 来评估参数可识别性
            try:
                cond_num = np.linalg.cond(pcov)
                print(f"  协方差矩阵条件数: {cond_num:.2e}")
                if cond_num > 1e12:
                    print("    警告: 条件数过大，参数可能存在识别性问题")
                elif cond_num > 1e6:
                    print("    注意: 条件数较大，参数估计可能不稳定")
                else:
                    print("    参数识别性良好")
            except:
                print("  无法计算条件数")
    
    # 绘制比较图
    if plot_comparison and models:
        plt.figure(figsize=(12, 8))
        
        # 绘制观测数据
        plt.plot(time_points, concentrations_iv, 'ro', label='Observed data', 
                markersize=8, markerfacecolor='red', markeredgecolor='darkred', linewidth=2)
        
        # 绘制所有模型的拟合曲线
        model_names = list(models.keys())
        for i, (name, result) in enumerate(models.items()):
            popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent = result
            plt.plot(t_fit, c_fit, colors[i], label=f'{labels[i]} (R²={r_squared:.3f})', 
                    linewidth=2, alpha=0.8)
        
        plt.xlabel('Time (h)', fontsize=14)
        plt.ylabel('Concentration (μg·L⁻¹)', fontsize=14)
        plt.yscale('log')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.title('五种二室模型拟合效果比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 绘制残差分析图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(models.items()):
            if i >= 5:  # 最多显示5个模型
                break
                
            popt, pcov, r_squared, t_fit, c_fit, param_errors, rse_percent = result
            
            # 计算残差
            if name == 'Traditional':
                def model_func(t, *params):
                    A, alpha, B, beta = params
                    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)
            elif name == 'Takada':
                def model_func(t, *params):
                    beta, Vmax, kd, Vc = params
                    Vt = (Vmax * t)/(kd + t)
                    return D_iv * np.exp(-beta * t)/(Vc + Vt)
            elif name == 'Colburn':
                def model_func(t, *params):
                    beta, Vmax, kv, Vc = params
                    Vt = Vmax * (1 - np.exp(-kv * t))
                    return D_iv * np.exp(-beta * t)/(Vc + Vt)
            elif name == 'Reparameterized':
                def model_func(t, *params):
                    beta, Cl, a, B = params
                    return a * (D_iv/Cl - B/beta) * np.exp(-a * t) + B * np.exp(-beta * t)
            elif name == 'Differential':
                def model_func(t, *params):
                    return two_compartment_model_ode_fit(t, *params)
            
            predicted = model_func(time_points, *popt)
            residuals = concentrations_iv - predicted
            
            axes[i].scatter(predicted, residuals, alpha=0.7)
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'{name} Model')
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        if len(models) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('残差分析 (Residual Analysis)', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    return models, comparison_data

# 主程序执行
if __name__ == "__main__":
    # 加载数据
    time_points, concentrations_iv = load_data(plot_data=False)
    
    # 运行所有模型比较
    models, comparison_data = compare_all_models(time_points, concentrations_iv, D_iv=dose, 
                                               plot_individual=False, plot_comparison=True)
