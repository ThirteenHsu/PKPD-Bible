import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit, differential_evolution, minimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression 
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
objectives:
• To simultaneously fit single and multiple (steady state) dose data
• To analyze four different absorption/disposition models, 分析4个不同的模型
• To discuss ways of discrimination between competing models, 讨论不同模型之间的区分方法
• To combine pharmacokinetic findings to pharmacodynamic goals, 结合 pharmacokinetic 结果到 pharmacodynamic 目标


一室口服吸收模型（由于没有找到数据，暂时用pk3的观测数据代替）

主要参数:
  ka: 药物的吸收速率常数 (例如: 1/min)。
  V: 药物的分布容积 (例如: L)。
  ke: 药物的消除速率常数 (例如: 1/min)。
  C: 血浆药物浓度 (例如: mg/L)。
  t_lag: 药物进入体循环后的延迟时间 (例如: min)。

微分方程1:不考虑t_lag的模型
    dAg/dt = -ka * Ag  胃肠道隔室药物的吸收过程
    dAc/dt = ka * Ag - ke * Ac  体内药物的吸收和消除过程

微分方程2: 考虑t_lag的模型
    当t < t_lag:
        dAg/dt = 0  药物进入体循环前，胃肠道隔室药物浓度为0
        dAc/dt = 0  体内药物浓度为0
    当t >= t_lag:
        dAg/dt = -ka * Ag  胃肠道隔室药物的吸收过程
        dAc/dt = ka * Ag - ke * Ac  体内药物的吸收和消除过程

原方程3: 当ka与ke非常相近的是否，对原始的模型进行洛必达法则的极限
    C = (K' * F * Dpo / V) * t * e^(-K' * t)

原方程4：考虑t_lag的情况
    C = (K' * F * Dpo / V) * (t - t_lag) * e^(-K' * (t - t_lag))

"""

dose_oral = 20 * 1000  # μg (20 mg) 

def load_data(plot_data=True):
    # Original data
    data = {
        'Time (h)': [0.51, 1.00, 1.50, 2.00, 3.01, 4.01, 5.00, 6.02, 7.01, 8.00, 9.02, 10.01],
        'Concentration (μg/L)': [10.24, 46.48, 56.40, 56.40, 66.66, 87.30, 69.77, 45.26, 29.52, 15.56, 10.54, 9.42],
    }

    df = pd.DataFrame(data)
    time_points = df['Time (h)'].values
    concentrations = df['Concentration (μg/L)'].values

    if plot_data:
        # 创建原始数据散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(time_points, concentrations, color='gray', s=80, alpha=0.8, edgecolors='black', linewidth=1)
        plt.xlabel('Time (h)', fontsize=12)
        plt.ylabel('Concentration (μg L⁻¹)', fontsize=12)
        plt.title('原始数据散点图', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 12)
        plt.ylim(0, 100)

        # 设置坐标轴刻度
        plt.xticks(np.arange(0, 12.1, 1))
        plt.yticks(np.arange(0, 100.1, 10))

        # 美化图表
        plt.tight_layout()
        plt.show()

    return time_points, concentrations

def calculate_auc_trapezoidal(time_points, concentrations):
    """
    使用线性梯形法计算药时曲线的AUC。
    参数:
        time_points (array-like): 时间点数组。
        concentrations (array-like): 对应时间点的浓度数组。
    返回:
        float: 计算得到的AUC值。
    """
    if len(time_points) != len(concentrations):
        raise ValueError("时间点和浓度数组的长度必须相同。")
    if len(time_points) < 2:
        return 0.0  # 至少需要两个点才能计算AUC

    auc = 0.0
    for i in range(len(time_points) - 1):
        # 梯形面积 = (C_i + C_{i+1}) / 2 * (t_{i+1} - t_i)
        auc += (concentrations[i] + concentrations[i+1]) / 2 * (time_points[i+1] - time_points[i])
    return auc

print("\n" + "="*50)

time_points, concentrations = load_data()

# ============================================================================
# 基于L23-48描述的四个药代动力学模型
# ============================================================================

def differential_model_1_no_lag_ode(t, y, ka, ke):
    """
    微分方程1: 不考虑t_lag的模型
    dAg/dt = -ka * Ag  胃肠道隔室药物的吸收过程
    dAc/dt = ka * Ag - ke * Ac  体内药物的吸收和消除过程
    """
    Ag, Ac = y
    dAg_dt = -ka * Ag
    dAc_dt = ka * Ag - ke * Ac
    return [dAg_dt, dAc_dt]

def model_1_differential_no_lag(t, ka, ke, v_f):
    """
    模型1: 微分方程1的数值解 (不考虑t_lag)
    
    参数:
        t: 时间数组 (h)
        ka: 吸收速率常数 (1/h)
        ke: 消除速率常数 (1/h)
        v_f: 表观分布容积 V/F (L)
    """
    # 初始条件: Ag(0) = dose_oral, Ac(0) = 0
    y0 = [dose_oral, 0.0]
    
    # 求解微分方程
    sol = solve_ivp(
        lambda t_val, y_val: differential_model_1_no_lag_ode(t_val, y_val, ka, ke),
        [0, max(t)], y0, t_eval=t, method='RK45', rtol=1e-8
    )
    
    if sol.success:
        # 浓度 = 中央室药量 / 表观分布容积
        concentrations = sol.y[1] / v_f
        return concentrations
    else:
        return np.zeros_like(t)

def differential_model_2_with_lag_ode(t, y, ka, ke, tlag):
    """
    微分方程2: 考虑t_lag的模型
    当t < t_lag: dAg/dt = 0, dAc/dt = 0
    当t >= t_lag: dAg/dt = -ka * Ag, dAc/dt = ka * Ag - ke * Ac
    """
    Ag, Ac = y
    
    if t < tlag:
        dAg_dt = 0.0
        dAc_dt = 0.0
    else:
        dAg_dt = -ka * Ag
        dAc_dt = ka * Ag - ke * Ac
    
    return [dAg_dt, dAc_dt]

def model_2_differential_with_lag(t, ka, ke, v_f, tlag):
    """
    模型2: 微分方程2的数值解 (考虑t_lag)
    
    参数:
        t: 时间数组 (h)
        ka: 吸收速率常数 (1/h)
        ke: 消除速率常数 (1/h)
        v_f: 表观分布容积 V/F (L)
        tlag: 滞后时间 (h)
    """
    # 初始条件: Ag(0) = dose_oral, Ac(0) = 0
    y0 = [dose_oral, 0.0]
    
    # 求解微分方程
    sol = solve_ivp(
        lambda t_val, y_val: differential_model_2_with_lag_ode(t_val, y_val, ka, ke, tlag),
        [0, max(t)], y0, t_eval=t, method='RK45', rtol=1e-8
    )
    
    if sol.success:
        # 浓度 = 中央室药量 / 表观分布容积
        concentrations = sol.y[1] / v_f
        return concentrations
    else:
        return np.zeros_like(t)

def model_3_analytical_ka_equals_ke(t, k_prime, v_f, f):
    """
    模型3: 原方程3 - Ka=Ke时的洛必达法则结果
    C = (K' * F * Dpo / V) * t * e^(-K' * t)
    
    参数:
        t: 时间 (h)
        k_prime: Ka = Ke = K' (1/h)
        v_f: 表观分布容积 V/F (L)
        f: 生物利用度 F (无量纲)
    """
    return (k_prime * f * dose_oral / v_f) * t * np.exp(-k_prime * t)

def model_4_analytical_ka_equals_ke_with_lag(t, k_prime, v_f, f, tlag):
    """
    模型4: 原方程4 - Ka=Ke且考虑t_lag的情况
    当t < tlag时，C = 0
    当t >= tlag时，C = (K' * F * Dpo / V) * (t - tlag) * e^(-K' * (t - tlag))
    
    参数:
        t: 时间 (h)
        k_prime: Ka = Ke = K' (1/h)
        v_f: 表观分布容积 V/F (L)
        f: 生物利用度 F (无量纲)
        tlag: 滞后时间 (h)
    """
    conc = np.zeros_like(t)
    mask = t >= tlag
    t_eff = t[mask] - tlag
    conc[mask] = (k_prime * f * dose_oral / v_f) * t_eff * np.exp(-k_prime * t_eff)
    return conc

def calculate_fit_metrics(observed, predicted, n_params):
    """
    计算拟合质量指标
    
    参数:
        observed: 观测值数组
        predicted: 预测值数组
        n_params: 参数个数
    
    返回:
        r2, rmse, aic, bic: 拟合质量指标
    """
    # R²
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # RMSE
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    
    # AIC和BIC
    n = len(observed)
    mse = ss_res / n
    aic = n * np.log(mse) + 2 * n_params
    bic = n * np.log(mse) + n_params * np.log(n)
    
    # WRSS (Weighted Residual Sum of Squares)
    # 假设权重为 1/observed^2，这在药代动力学中是常见的加权方式
    # 为了避免除以零，对observed为零的值进行处理，例如替换为非常小的数
    weighted_observed = np.where(observed == 0, 1e-6, observed)
    weights = 1 / (weighted_observed ** 2)
    wrss = np.sum(weights * (observed - predicted) ** 2)

    return r2, rmse, aic, bic, wrss

def fit_four_models_comparison():
    """
    对四个模型进行参数拟合并比较结果
    """
    print("\n" + "="*80)
    print("基于L23-48描述的四个药代动力学模型比较分析")
    print("="*80)
    
    # 获取实验数据
    time_data, conc_data = load_data(plot_data=False)
    
    # 存储所有模型的拟合结果
    models_results = {}
    
    # 模型1: 微分方程1 (不考虑t_lag)
    print("\n模型1: 微分方程 - 不考虑t_lag")
    print("dAg/dt = -ka * Ag")
    print("dAc/dt = ka * Ag - ke * Ac")
    print("-" * 50)
    try:
        # 初始猜测: [ka, ke, v_f]
        initial_guess_1 = [1.0, 0.3, 50.0]
        bounds_1 = ([0.01, 0.01, 1.0], [10.0, 5.0, 500.0])
        
        popt_1, pcov_1 = curve_fit(
            model_1_differential_no_lag, time_data, conc_data,
            p0=initial_guess_1, bounds=bounds_1, maxfev=5000
        )  # curve_fit传的是原函数或求解后的微分方程
        
        # 计算拟合质量
        pred_1 = model_1_differential_no_lag(time_data, *popt_1)
        r2_1, rmse_1, aic_1, bic_1, wrss_1 = calculate_fit_metrics(conc_data, pred_1, len(popt_1))
        
        models_results['Model 1 (微分方程-无lag)'] = {
            'params': popt_1,
            'param_names': ['Ka', 'Ke', 'V/F'],
            'param_errors': np.sqrt(np.diag(pcov_1)),
            'r2': r2_1,
            'rmse': rmse_1,
            'aic': aic_1,
            'bic': bic_1,
            'wrss': wrss_1, # Added WRSS
            'predictions': pred_1,
            'model_type': 'differential'
        }
        
        print(f"Ka = {popt_1[0]:.4f} ± {np.sqrt(pcov_1[0,0]):.4f} h⁻¹")
        print(f"Ke = {popt_1[1]:.4f} ± {np.sqrt(pcov_1[1,1]):.4f} h⁻¹")
        print(f"V/F = {popt_1[2]:.2f} ± {np.sqrt(pcov_1[2,2]):.2f} L")
        print(f"R² = {r2_1:.4f}, RMSE = {rmse_1:.2f}, AIC = {aic_1:.2f}")
        
    except Exception as e:
        print(f"模型1拟合失败: {e}")
        models_results['Model 1 (微分方程-无lag)'] = None
    
    # 模型2: 微分方程2 (考虑t_lag)
    print("\n模型2: 微分方程 - 考虑t_lag")
    print("当t < t_lag: dAg/dt = 0, dAc/dt = 0")
    print("当t >= t_lag: dAg/dt = -ka * Ag, dAc/dt = ka * Ag - ke * Ac")
    print("-" * 50)
    try:
        # 初始猜测: [ka, ke, v_f, tlag]
        initial_guess_2 = [1.0, 0.3, 50.0, 0.3]
        bounds_2 = ([0.01, 0.01, 1.0, 0.0], [10.0, 5.0, 500.0, 2.0])
        
        popt_2, pcov_2 = curve_fit(
            model_2_differential_with_lag, time_data, conc_data,
            p0=initial_guess_2, bounds=bounds_2, maxfev=5000
        )
        
        # 计算拟合质量
        pred_2 = model_2_differential_with_lag(time_data, *popt_2)
        r2_2, rmse_2, aic_2, bic_2, wrss_2 = calculate_fit_metrics(conc_data, pred_2, len(popt_2))
        
        models_results['Model 2 (微分方程-有lag)'] = {
            'params': popt_2,
            'param_names': ['Ka', 'Ke', 'V/F', 'Tlag'],
            'param_errors': np.sqrt(np.diag(pcov_2)),
            'r2': r2_2,
            'rmse': rmse_2,
            'aic': aic_2,
            'bic': bic_2,
            'wrss': wrss_2, # Added WRSS
            'predictions': pred_2,
            'model_type': 'differential'
        }
        
        print(f"Ka = {popt_2[0]:.4f} ± {np.sqrt(pcov_2[0,0]):.4f} h⁻¹")
        print(f"Ke = {popt_2[1]:.4f} ± {np.sqrt(pcov_2[1,1]):.4f} h⁻¹")
        print(f"V/F = {popt_2[2]:.2f} ± {np.sqrt(pcov_2[2,2]):.2f} L")
        print(f"Tlag = {popt_2[3]:.4f} ± {np.sqrt(pcov_2[3,3]):.4f} h")
        print(f"R² = {r2_2:.4f}, RMSE = {rmse_2:.2f}, AIC = {aic_2:.2f}")
        
    except Exception as e:
        print(f"模型2拟合失败: {e}")
        models_results['Model 2 (微分方程-有lag)'] = None
    
    # 模型3: 原方程3 (Ka=Ke的洛必达法则结果)
    print("\n模型3: 解析解 - Ka=Ke时的洛必达法则")
    print("C = (K' * F * Dpo / V) * t * e^(-K' * t)")
    print("-" * 50)
    try:
        # 初始猜测: [k_prime, v_f, f]
        initial_guess_3 = [0.5, 50.0, 1.0]
        bounds_3 = ([0.01, 1.0, 0.1], [5.0, 500.0, 1.0])
        
        popt_3, pcov_3 = curve_fit(
            model_3_analytical_ka_equals_ke, time_data, conc_data,
            p0=initial_guess_3, bounds=bounds_3, maxfev=5000
        )
        
        # 计算拟合质量
        pred_3 = model_3_analytical_ka_equals_ke(time_data, *popt_3)
        r2_3, rmse_3, aic_3, bic_3, wrss_3 = calculate_fit_metrics(conc_data, pred_3, len(popt_3))
        
        models_results['Model 3 (解析解-Ka=Ke)'] = {
            'params': popt_3,
            'param_names': ["K'", 'V/F', 'F'],
            'param_errors': np.sqrt(np.diag(pcov_3)),
            'r2': r2_3,
            'rmse': rmse_3,
            'aic': aic_3,
            'bic': bic_3,
            'wrss': wrss_3, 
            'predictions': pred_3,
            'model_type': 'analytical'
        }
        
        print(f"K' = {popt_3[0]:.4f} ± {np.sqrt(pcov_3[0,0]):.4f} h⁻¹")
        print(f"V/F = {popt_3[1]:.2f} ± {np.sqrt(pcov_3[1,1]):.2f} L")
        print(f"F = {popt_3[2]:.4f} ± {np.sqrt(pcov_3[2,2]):.4f}")
        print(f"R² = {r2_3:.4f}, RMSE = {rmse_3:.2f}, AIC = {aic_3:.2f}")
        
    except Exception as e:
        print(f"模型3拟合失败: {e}")
        models_results['Model 3 (解析解-Ka=Ke)'] = None
    
    # 模型4: 原方程4 (Ka=Ke且考虑t_lag)
    print("\n模型4: 解析解 - Ka=Ke且考虑t_lag")
    print("当t < tlag: C = 0")
    print("当t >= tlag: C = (K' * F * Dpo / V) * (t - tlag) * e^(-K' * (t - tlag))")
    print("-" * 50)
    try:
        # 初始猜测: [k_prime, v_f, f, tlag]
        initial_guess_4 = [0.5, 50.0, 1.0, 0.3]
        bounds_4 = ([0.01, 1.0, 0.1, 0.0], [5.0, 500.0, 1.0, 2.0])
        
        popt_4, pcov_4 = curve_fit(
            model_4_analytical_ka_equals_ke_with_lag, time_data, conc_data,
            p0=initial_guess_4, bounds=bounds_4, maxfev=5000
        )
        
        # 计算拟合质量
        pred_4 = model_4_analytical_ka_equals_ke_with_lag(time_data, *popt_4)
        r2_4, rmse_4, aic_4, bic_4, wrss_4 = calculate_fit_metrics(conc_data, pred_4, len(popt_4))
        
        models_results['Model 4 (解析解-Ka=Ke+lag)'] = {
            'params': popt_4,
            'param_names': ["K'", 'V/F', 'F', 'Tlag'],
            'param_errors': np.sqrt(np.diag(pcov_4)),
            'r2': r2_4,
            'rmse': rmse_4,
            'aic': aic_4,
            'bic': bic_4,
            'wrss': wrss_4, 
            'predictions': pred_4,
            'model_type': 'analytical'
        }
        
        print(f"K' = {popt_4[0]:.4f} ± {np.sqrt(pcov_4[0,0]):.4f} h⁻¹")
        print(f"V/F = {popt_4[1]:.2f} ± {np.sqrt(pcov_4[1,1]):.2f} L")
        print(f"F = {popt_4[2]:.4f} ± {np.sqrt(pcov_4[2,2]):.4f}")
        print(f"Tlag = {popt_4[3]:.4f} ± {np.sqrt(pcov_4[3,3]):.4f} h")
        print(f"R² = {r2_4:.4f}, RMSE = {rmse_4:.2f}, AIC = {aic_4:.2f}")
        
    except Exception as e:
        print(f"模型4拟合失败: {e}")
        models_results['Model 4 (解析解-Ka=Ke+lag)'] = None
    
    return models_results, time_data, conc_data

def visualize_four_models_comparison(models_results, time_data, conc_data):
    """
    可视化四个模型的拟合结果对比
    """
    # 生成更密集的时间点用于绘制平滑曲线
    t_fine = np.linspace(0, max(time_data), 200)
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制实验数据点
    plt.scatter(time_data, conc_data, color='red', s=100, alpha=0.8, 
                edgecolors='darkred', linewidth=2, label='实验数据', zorder=10)
    
    # 定义颜色和线型 - 使用更明显的区别
    colors = ['blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':']
    linewidths = [3, 3, 3, 3]  # 增加线宽
    
    # 绘制每个模型的拟合曲线
    model_names = list(models_results.keys())
    plot_count = 0
    
    for i, (model_name, result) in enumerate(models_results.items()):
        if result is not None:
            
            # 根据模型类型生成预测曲线
            if '微分方程-无lag' in model_name:
                pred_fine = model_1_differential_no_lag(t_fine, *result['params'])
                short_name = "微分方程(无lag)"
            elif '微分方程-有lag' in model_name:
                pred_fine = model_2_differential_with_lag(t_fine, *result['params'])
                short_name = "微分方程(有lag)"
            elif '解析解-Ka=Ke)' in model_name:
                pred_fine = model_3_analytical_ka_equals_ke(t_fine, *result['params'])
                short_name = "解析解(Ka=Ke)"
            elif '解析解-Ka=Ke+lag' in model_name:
                pred_fine = model_4_analytical_ka_equals_ke_with_lag(t_fine, *result['params'])
                short_name = "解析解(Ka=Ke+lag)"
            else:
                print(f"警告: 未识别的模型类型: {model_name}")
                continue
            
            # 检查预测值是否有效
            if np.any(np.isnan(pred_fine)) or np.any(np.isinf(pred_fine)):
                print(f"警告: 模型 {model_name} 的预测值包含NaN或Inf")
                continue
                
            
            
            plt.plot(t_fine, pred_fine, 
                    color=colors[plot_count % len(colors)], 
                    linestyle=linestyles[plot_count % len(linestyles)], 
                    linewidth=linewidths[plot_count % len(linewidths)], 
                    label=f"{short_name} (R²={result['r2']:.3f})", 
                    alpha=0.9)
            plot_count += 1
    
    print(f"总共绘制了 {plot_count} 个模型")
    
    # 图形美化
    plt.xlabel('时间 (h)', fontsize=14)
    plt.ylabel('浓度 (μg/L)', fontsize=14)
    plt.title('四个药代动力学模型拟合结果对比', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(time_data) * 1.1)
    plt.ylim(0, max(conc_data) * 1.1)
    
    plt.tight_layout()
    plt.show()

def generate_comparison_table(models_results):
    """
    生成参数估计结果对比表格 - 使用 pandas DataFrame 格式
    """
    print("\n" + "="*80)
    print("四个模型的参数估计和拟合效果对比表")
    print("="*80)
    
    # 准备数据
    table_data = []
    
    for model_name, result in models_results.items():
        if result is not None:
            params = result['params']
            param_names = result['param_names']
            param_errors = result['param_errors']
            
            # 为每个参数创建一行数据
            for i, (param_name, param_val, param_err) in enumerate(zip(param_names, params, param_errors)):
                row = {
                    '模型': model_name if i == 0 else '',  # 只在第一行显示模型名称
                    '参数': param_name,
                    '估计值': f"{param_val:.4f}",
                    '标准误差': f"{param_err:.4f}",
                    'R²': f"{result['r2']:.4f}" if i == 0 else '',
                    'RMSE': f"{result['rmse']:.2f}" if i == 0 else '',
                    'AIC': f"{result['aic']:.2f}" if i == 0 else '',
                    'BIC': f"{result['bic']:.2f}" if i == 0 else '',
                    'WRSS': f"{result['wrss']:.2f}" if i == 0 else ''
                }
                table_data.append(row)
        else:
            # 拟合失败的模型
            row = {
                '模型': model_name,
                '参数': '拟合失败',
                '估计值': '-',
                '标准误差': '-',
                'R²': '-',
                'RMSE': '-',
                'AIC': '-',
                'BIC': '-',
                'WRSS': '-'
            }
            table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # 打印表格
    print(df.to_string(index=False))
    
    # 模型排名
    print("\n" + "="*50)
    print("模型拟合效果排名 (基于AIC值，越小越好):")
    print("="*50)
    
    valid_models = {k: v for k, v in models_results.items() if v is not None}
    if valid_models:
        # 创建排名 DataFrame
        ranking_data = []
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['aic'])
        
        for i, (model_name, result) in enumerate(sorted_models, 1):
            ranking_data.append({
                '排名': i,
                '模型': model_name,
                'AIC': f"{result['aic']:.2f}",
                'R²': f"{result['r2']:.4f}",
                'RMSE': f"{result['rmse']:.2f}",
                'WRSS': f"{result['wrss']:.2f}"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        print(ranking_df.to_string(index=False))

# 执行参数拟合
if __name__ == "__main__":
    try:
        # 执行四个模型比较分析
        print("\n开始执行四个模型比较分析...")
        models_results, time_data, conc_data = fit_four_models_comparison()
        print("四个模型比较分析完成！")
        
        # 可视化模型比较结果
        print("\n开始可视化模型比较结果...")
        visualize_four_models_comparison(models_results, time_data, conc_data)
        print("可视化完成！")
        
        # 生成对比表格
        print("\n开始生成对比表格...")
        generate_comparison_table(models_results)
        print("对比表格生成完成！")
        
        print("\n所有分析完成！")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


