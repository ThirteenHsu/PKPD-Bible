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

# Basic information
dose_oral = 100  # μg

# Original data
data = {
    'Time (min)': [10, 15, 20, 30, 40, 60, 90, 120, 180, 210, 240, 300, 360],
    'Concentration (μg/L)': [0, 0.28, 0.55, 1.2, 2, 1.95, 1.85, 1.6, 0.86, 0.78, 0.6, 0.21, 0.18],
}

df = pd.DataFrame(data)
time_points = df['Time (min)'].values
concentrations = df['Concentration (μg/L)'].values

print("Original Data:")
print(df)

# # 创建原始数据散点图
# plt.figure(figsize=(10, 6))
# plt.scatter(time_points, concentrations, color='gray', s=80, alpha=0.8, edgecolors='black', linewidth=1)
# plt.xlabel('Time (min)', fontsize=12)
# plt.ylabel('Concentration (μg L⁻¹)', fontsize=12)
# plt.title('原始数据散点图', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.xlim(0, 400)
# plt.ylim(0, 2.2)

# # 设置坐标轴刻度
# plt.xticks(np.arange(0, 401, 50))
# plt.yticks(np.arange(0, 2.1, 0.2))

# # 美化图表
# plt.tight_layout()
# plt.show()

print("\n" + "="*50)
print("Method 1: Improved Feathering Method")
print("="*50)

# 方法1: 改进的残差法 (Improved Feathering Method)
def improved_feathering_method():
    """
    改进的残差法参数估算
    
    残差法原理：
    1. 先用末端消除相数据估算ke（消除速率常数）
    2. 将消除相外推到整个时间段，得到外推浓度
    3. 用观测浓度减去外推浓度得到残差
    4. 残差代表吸收相的贡献，用来估算ka（吸收速率常数）
    5. 通过浓度开始上升的时间点估算tlag（滞后时间）
    """
    
    # 步骤1: 从末端消除相估算ke
    # 选择最后4-5个非零浓度点进行线性回归
    non_zero_indices = np.where(concentrations > 0)[0]  # 找到所有非零浓度点的索引
    n_terminal = min(4, len(non_zero_indices))  # 使用最后4个点或所有可用点
    terminal_indices = non_zero_indices[-n_terminal:]  # 选择末端点的索引
    
    terminal_time = time_points[terminal_indices]  # 末端时间点
    terminal_conc = concentrations[terminal_indices]  # 末端浓度
    
    # 对浓度取对数后进行线性回归 (ln(C) = ln(A) - ke*t)
    log_conc = np.log(terminal_conc)

    model_ke = LinearRegression()
    model_ke.fit(terminal_time.reshape(-1, 1), log_conc)
    slope_ke = np.abs(model_ke.coef_[0]) # 消除速率常数
    intercept_b0 = model_ke.intercept_  # ln(A)
    log_extra = model_ke.predict(time_points.reshape(-1, 1))
    C_extrap = np.exp(log_extra)
    
    print(f"末端拟合截距 intercept_b0 = {intercept_b0:.4f}")
    print(f"末端拟合点数: {len(terminal_indices)}")
    print(f"消除速率常数 ke = {slope_ke:.2f} min^-1")
    print(f"末端拟合 R² = {model_ke.score(terminal_time.reshape(-1, 1), log_conc):.4f}")

    # # 绘制第一张图：原始数据散点图和末端消除相拟合线，semilogy创建半对数刻度
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(time_points, concentrations, 'ko', label='原始观测数据', markersize=6)
    # # 只绘制末端拟合点和拟合线
    # plt.semilogy(time_points, C_extrap, 'ro--', label=f'末端拟合线 (ke={slope_ke:.4f})', linewidth=2, markersize=6)
    # plt.xlabel('时间 (min)')
    # plt.ylabel('浓度 (μg/L, 对数尺度)')
    # plt.title('步骤1: 末端消除相拟合')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    
    # 步骤2.计算残差
    # 将消除相外推到所有时间点
    residuals = C_extrap - concentrations  # 残差 = 外推浓度 - 观测浓度
    
    # 步骤3: 选择吸收相数据点估算ka
    # 条件：残差>0（表示吸收贡献），浓度>0，不在末端相
    absorption_mask = (residuals > 0) & (concentrations > 0) & (~np.isin(np.arange(len(time_points)), terminal_indices))
    absorption_indices = np.where(absorption_mask)[0]
    
    if len(absorption_indices) >= 2:
        print(f"可用吸收相数据点总数: {len(absorption_indices)}")
        print(f"吸收相数据点时间: {time_points[absorption_indices]}")
        print(f"对应的残差值: {residuals[absorption_indices]}")
        
        # 比较不同点数选择对tlag计算的影响
        print("\n" + "="*60)
        print("不同点数选择对tlag计算的影响分析:")
        print("="*60)
        
        results_comparison = []
        max_points = min(len(absorption_indices), 8)  # 最多测试8个点
        
        for n_points in range(2, max_points + 1):
            try:
                # 选择前n_points个吸收相数据点
                time_abs = time_points[absorption_indices[:n_points]]
                residuals_abs = residuals[absorption_indices[:n_points]]
                
                # 检查残差是否都为正数
                if np.any(residuals_abs <= 0):
                    print(f"使用{n_points}个点: 发现非正残差值，跳过")
                    continue
                
                # 对残差取对数后进行线性回归
                log_residuals = np.log(residuals_abs)
                model_ka = LinearRegression()
                model_ka.fit(time_abs.reshape(-1, 1), log_residuals)
                slope_ka = np.abs(model_ka.coef_[0])
                intercept_ka = model_ka.intercept_
                r_squared = model_ka.score(time_abs.reshape(-1, 1), log_residuals)
                
                # 计算tlag
                tlag = (intercept_ka - intercept_b0)/(slope_ka - slope_ke)
                
                # 存储结果
                results_comparison.append({
                    'n_points': n_points,
                    'ka': slope_ka,
                    'intercept_ka': intercept_ka,
                    'tlag': tlag,
                    'r_squared': r_squared,
                    'time_points': time_abs.tolist(),
                    'residuals': residuals_abs.tolist()
                })
                
                print(f"使用{n_points}个点: ka={slope_ka:.4f}, tlag={tlag:.2f}min, R²={r_squared:.4f}")
                
            except Exception as e:
                print(f"使用{n_points}个点时出错: {e}")
        
        # 创建详细的比较表格
        if results_comparison:
            print("\n" + "="*80)
            print("详细比较结果:")
            print("="*80)
            print(f"{'点数':<4} {'ka (min⁻¹)':<12} {'tlag (min)':<12} {'R²':<8} {'使用的时间点':<20}")
            print("-" * 80)
            
            for result in results_comparison:
                time_str = str(result['time_points'][:3]) + "..." if len(result['time_points']) > 3 else str(result['time_points'])
                print(f"{result['n_points']:<4} {result['ka']:<12.4f} {result['tlag']:<12.2f} {result['r_squared']:<8.4f} {time_str:<20}")
            
            # 可视化tlag随点数的变化
            plt.figure(figsize=(12, 8))
            
            # 子图1: tlag随点数变化
            plt.subplot(2, 2, 1)
            n_points_list = [r['n_points'] for r in results_comparison]
            tlag_list = [r['tlag'] for r in results_comparison]
            plt.plot(n_points_list, tlag_list, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('使用的数据点数')
            plt.ylabel('tlag (min)')
            plt.title('tlag随数据点数的变化')
            plt.grid(True, alpha=0.3)
            
            # 子图2: ka随点数变化
            plt.subplot(2, 2, 2)
            ka_list = [r['ka'] for r in results_comparison]
            plt.plot(n_points_list, ka_list, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('使用的数据点数')
            plt.ylabel('ka (min⁻¹)')
            plt.title('ka随数据点数的变化')
            plt.grid(True, alpha=0.3)
            
            # 子图3: R²随点数变化
            plt.subplot(2, 2, 3)
            r2_list = [r['r_squared'] for r in results_comparison]
            plt.plot(n_points_list, r2_list, 'go-', linewidth=2, markersize=8)
            plt.xlabel('使用的数据点数')
            plt.ylabel('R²')
            plt.title('拟合优度随数据点数的变化')
            plt.grid(True, alpha=0.3)
            
            # 子图4: 不同点数下的拟合线对比
            plt.subplot(2, 2, 4)
            plt.semilogy(time_points, concentrations, 'ko', label='原始数据', markersize=6)
            plt.semilogy(time_points, C_extrap, 'r--', label=f'末端拟合线', linewidth=2)
            
            # 绘制不同点数下的吸收相拟合线
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta']
            for i, result in enumerate(results_comparison[:8]):  # 最多显示8条线
                time_abs = np.array(result['time_points'])
                residual_predict = result['intercept_ka'] + (-result['ka']) * time_abs
                color = colors[i % len(colors)]
                plt.semilogy(time_abs, np.exp(residual_predict), 
                           color=color, linewidth=2, 
                           label=f'{result["n_points"]}点: tlag={result["tlag"]:.1f}min')
            
            plt.xlabel('时间 (min)')
            plt.ylabel('浓度 (μg/L, 对数尺度)')
            plt.title('不同点数下的拟合线对比')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 推荐最佳点数
            best_result = max(results_comparison, key=lambda x: x['r_squared'])
            print(f"\n推荐使用 {best_result['n_points']} 个点 (R²最高: {best_result['r_squared']:.4f})")
            print(f"对应的参数: ka={best_result['ka']:.4f} min⁻¹, tlag={best_result['tlag']:.2f} min")
            
            # 使用推荐的点数作为最终结果
            slope_ka = best_result['ka']
            intercept_ka = best_result['intercept_ka']
            tlag = best_result['tlag']
            time_abs = np.array(best_result['time_points'])
            residuals_abs = np.array(best_result['residuals'])
            log_residuals = np.log(residuals_abs)
            
            # 重新计算residual_predict用于后续绘图
            model_ka = LinearRegression()
            model_ka.fit(time_abs.reshape(-1, 1), log_residuals)
            residual_predict = model_ka.predict(time_abs.reshape(-1, 1))
            
            print(f"\n最终采用参数:")
            print(f"吸收速率常数 ka = {slope_ka:.4f} min^-1")
            print(f"吸收相拟合截距 intercept_ka = {intercept_ka:.4f}")
            print(f"吸收相拟合 R² = {best_result['r_squared']:.4f}")  
            print(f"滞后时间 tlag = {tlag:.2f} min")
        
        else:
            print("无法进行有效的吸收相拟合")
            return None, None, None, None, C_extrap, residuals

        # # 绘制第三张图：原始数据、末端消除拟合线和吸收拟合线
        # plt.figure(figsize=(10, 6))
        # plt.semilogy(time_points, concentrations, 'ko', label='原始观测数据', markersize=8)
        # plt.semilogy(time_points, C_extrap, 'ro--', label=f'末端拟合线 (ke={slope_ke:.4f})', linewidth=2)
        # plt.semilogy(time_abs, residuals_abs, 'bo', label='吸收相残差', markersize=8)
        # plt.semilogy(time_abs, np.exp(residual_predict), 'b-', label=f'吸收相拟合线 (ka={slope_ka:.4f})', linewidth=3)
        # plt.axvline(x=tlag, color='g', linestyle='--', label=f'滞后时间 tlag = {tlag:.2f} min')
        # plt.xlabel('时间 (min)')
        # plt.ylabel('浓度 (μg/L, 对数尺度)')
        # plt.title('原始数据、末端消除拟合线和吸收拟合线')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.show()
    else:
        print("吸收相数据不足，使用默认值 ka = 0.02 min^-1")
        

    
    # 步骤5: 估算V/F（表观分布容积）
    cmax = np.max(concentrations)  # 最大浓度
    v_f = dose_oral / cmax  # 粗略估算：V/F = 剂量/Cmax
    
    return slope_ke, best_result['ka'], best_result['tlag'], v_f, C_extrap, residuals

# 方法2: 非线性拟合法 (Nonlinear Fitting Method)
def one_compartment_oral_model_ode(t, ka, ke, v_f, tlag):
    """
    一室口服给药模型的微分方程求解
    
    微分方程组：
    dAa/dt = -ka * Aa  (吸收室)
    dAc/dt = ka * Aa - ke * Ac  (中央室)
    
    初始条件：
    t = tlag 时：Aa(tlag) = 剂量, Ac(tlag) = 0
    t < tlag 时：C(t) = 0
    
    参数说明：
    - t: 时间
    - ka: 吸收速率常数
    - ke: 消除速率常数  
    - v_f: 表观分布容积 (V/F)
    - tlag: 吸收滞后时间
    """
    t = np.asarray(t)
    conc = np.zeros_like(t, dtype=float)
    
    # 只在 t > tlag 时计算浓度（滞后时间之前浓度为0）
    mask = t > tlag
    
    if np.any(mask):
        t_solve = t[mask]
        
        def ode_system(t_eff, y):
            """
            微分方程组
            y[0] = Aa (吸收室药量)
            y[1] = Ac (中央室药量)
            """
            Aa, Ac = y
            dAa_dt = -ka * Aa
            dAc_dt = ka * Aa - ke * Ac
            return [dAa_dt, dAc_dt]
        
        # 初始条件：t = tlag 时，所有药物在吸收室
        y0 = [dose_oral, 0.0]  # [Aa(tlag), Ac(tlag)]
        
        # 求解微分方程
        t_span = (0, t_solve[-1] - tlag)  # 从tlag开始的相对时间
        t_eval = t_solve - tlag  # 评估时间点（相对于tlag）
        
        try:
            sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, 
                          method='RK45', rtol=1e-8, atol=1e-10)
            
            if sol.success:
                # 浓度 = 中央室药量 / 表观分布容积
                conc[mask] = sol.y[1] / v_f
            else:
                print(f"微分方程求解失败: {sol.message}")
                # 回退到解析解
                return one_compartment_oral_model_analytical(t, ka, ke, v_f, tlag)
                
        except Exception as e:
            print(f"微分方程求解出错: {e}")
            # 回退到解析解
            return one_compartment_oral_model_analytical(t, ka, ke, v_f, tlag)
    
    return conc

def one_compartment_oral_model_analytical(t, ka, ke, v_f, tlag):
    """
    一室口服给药模型的解析解（备用方法）
    """
    t = np.asarray(t)
    conc = np.zeros_like(t, dtype=float)
    
    # 只在 t > tlag 时计算浓度（滞后时间之前浓度为0）
    mask = t > tlag
    t_eff = t[mask] - tlag  # 有效时间 = 实际时间 - 滞后时间
    
    # 避免 ka = ke 的数值问题
    if abs(ka - ke) < 1e-10:
        # 当 ka ≈ ke 时使用特殊公式（洛必达法则的结果）
        conc[mask] = (dose_oral * ka / v_f) * t_eff * np.exp(-ke * t_eff)
    else:
        # 标准的一室口服给药公式
        conc[mask] = (dose_oral * ka / (v_f * (ka - ke))) * (np.exp(-ke * t_eff) - np.exp(-ka * t_eff))
    
    return conc

# 使用微分方程求解作为主要方法
def one_compartment_oral_model(t, ka, ke, v_f, tlag):
    """
    一室口服给药模型 - 使用微分方程求解
    """
    return one_compartment_oral_model_ode(t, ka, ke, v_f, tlag)

def nonlinear_fitting_comprehensive():
    """
    综合非线性拟合法 - 包含全局优化和不同参数组合比较
    
    功能：
    1. 全局优化策略：多起始点、差分进化算法
    2. 不同参数组合优化：2参数、3参数、4参数
    3. 优化过程分析和结果比较
    4. 自动选择最优策略
    """
    
    # 获取残差法的初始估算值
    ke_init, ka_init, tlag_init, vf_init, _, _ = improved_feathering_method()
    
    print(f"\n=== 综合非线性拟合分析 ===")
    print(f"初始估算值: ka={ka_init:.4f}, ke={ke_init:.4f}, V/F={vf_init:.2f}, tlag={tlag_init:.2f}")
    
    results = {}
    
    # ==================== 1. 全局优化策略比较 ====================
    print(f"\n--- 1. 全局优化策略比较 ---")
    
    # 1.1 标准curve_fit（局部优化）
    results['local'] = fit_with_curve_fit(ka_init, ke_init, vf_init, tlag_init)
    
    # 1.2 多起始点优化
    results['multi_start'] = fit_with_multi_start()
    
    # 1.3 差分进化算法（全局优化）
    results['differential_evolution'] = fit_with_differential_evolution()
    
    # ==================== 2. 不同参数组合优化 ====================
    print(f"\n--- 2. 不同参数组合优化比较 ---")
    
    # 2.1 固定tlag，优化3参数 (ka, ke, V/F)
    results['3param_fixed_tlag'] = fit_3param_fixed_tlag(ka_init, ke_init, vf_init, tlag_init)
    
    # 2.2 固定V/F，优化3参数 (ka, ke, tlag)
    results['3param_fixed_vf'] = fit_3param_fixed_vf(ka_init, ke_init, vf_init, tlag_init)
    
    # 2.3 固定ke，优化3参数 (ka, V/F, tlag)
    results['3param_fixed_ke'] = fit_3param_fixed_ke(ka_init, ke_init, vf_init, tlag_init)
    
    # 2.4 优化2参数 (ka, ke)，固定V/F和tlag
    results['2param_ka_ke'] = fit_2param_ka_ke(ka_init, ke_init, vf_init, tlag_init)
    
    # ==================== 3. 结果分析和比较 ====================
    print(f"\n--- 3. 优化结果综合分析 ---")
    analyze_optimization_results(results)
    
    # 选择最优结果
    best_method, best_result = select_best_result(results)
    
    print(f"\n=== 推荐最优方法: {best_method} ===")
    print_result_details(best_result)
    
    return best_result

def fit_with_curve_fit(ka_init, ke_init, vf_init, tlag_init):
    """标准curve_fit优化（局部优化）"""
    try:
        initial_guess = [ka_init, ke_init, vf_init, tlag_init]
        bounds = ([0.001, 0.001, 1, 0], [1, 1, 200, 30])
        
        popt, pcov = curve_fit(one_compartment_oral_model, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, ke_fit, vf_fit, tlag_fit = popt
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fit)
        r_squared = calculate_r_squared(concentrations, pred_conc)
        
        result = {
            'method': 'curve_fit (局部优化)',
            'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fit, 'tlag': tlag_fit,
            'pred_conc': pred_conc, 'r_squared': r_squared,
            'success': True, 'n_params': 4
        }
        
        print(f"curve_fit: R²={r_squared:.4f}, ka={ka_fit:.4f}, ke={ke_fit:.4f}")
        return result
        
    except Exception as e:
        print(f"curve_fit失败: {e}")
        return {'success': False, 'method': 'curve_fit (局部优化)'}

def fit_with_multi_start():
    """多起始点优化"""
    print("执行多起始点优化...")
    
    # 生成多个起始点
    n_starts = 10
    np.random.seed(42)  # 确保可重复性
    
    best_result = None
    best_r2 = -np.inf
    
    for i in range(n_starts):
        try:
            # 随机生成起始点（在合理范围内）
            ka_start = np.random.uniform(0.005, 0.1)
            ke_start = np.random.uniform(0.005, 0.05)
            vf_start = np.random.uniform(10, 100)
            tlag_start = np.random.uniform(0, 20)
            
            initial_guess = [ka_start, ke_start, vf_start, tlag_start]
            bounds = ([0.001, 0.001, 1, 0], [1, 1, 200, 30])
            
            popt, pcov = curve_fit(one_compartment_oral_model, time_points, concentrations, 
                                  p0=initial_guess, bounds=bounds, maxfev=5000)
            
            ka_fit, ke_fit, vf_fit, tlag_fit = popt
            pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fit)
            r_squared = calculate_r_squared(concentrations, pred_conc)
            
            if r_squared > best_r2:
                best_r2 = r_squared
                best_result = {
                    'method': '多起始点优化',
                    'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fit, 'tlag': tlag_fit,
                    'pred_conc': pred_conc, 'r_squared': r_squared,
                    'success': True, 'n_params': 4, 'best_start': i+1
                }
                
        except Exception:
            continue
    
    if best_result:
        print(f"多起始点优化: R²={best_result['r_squared']:.4f} (最佳起始点: {best_result['best_start']}/{n_starts})")
        return best_result
    else:
        print("多起始点优化全部失败")
        return {'success': False, 'method': '多起始点优化'}

def fit_with_differential_evolution():
    """差分进化算法（全局优化）"""
    print("执行差分进化全局优化...")
    
    def objective_function(params):
        ka, ke, vf, tlag = params
        try:
            pred_conc = one_compartment_oral_model(time_points, ka, ke, vf, tlag)
            mse = np.mean((concentrations - pred_conc) ** 2)
            return mse
        except:
            return 1e10  # 返回大值表示失败
    
    try:
        # 参数边界
        bounds = [(0.001, 1), (0.001, 1), (1, 200), (0, 30)]
        
        result = differential_evolution(objective_function, bounds, 
                                      seed=42, maxiter=1000, popsize=15)
        
        if result.success:
            ka_fit, ke_fit, vf_fit, tlag_fit = result.x
            pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fit)
            r_squared = calculate_r_squared(concentrations, pred_conc)
            
            result_dict = {
                'method': '差分进化算法',
                'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fit, 'tlag': tlag_fit,
                'pred_conc': pred_conc, 'r_squared': r_squared,
                'success': True, 'n_params': 4, 'iterations': result.nit
            }
            
            print(f"差分进化: R²={r_squared:.4f}, 迭代次数={result.nit}")
            return result_dict
        else:
            print(f"差分进化失败: {result.message}")
            return {'success': False, 'method': '差分进化算法'}
            
    except Exception as e:
        print(f"差分进化出错: {e}")
        return {'success': False, 'method': '差分进化算法'}

def fit_3param_fixed_tlag(ka_init, ke_init, vf_init, tlag_fixed):
    """固定tlag，优化3参数 (ka, ke, V/F)"""
    def model_3param(t, ka, ke, vf):
        return one_compartment_oral_model(t, ka, ke, vf, tlag_fixed)
    
    try:
        initial_guess = [ka_init, ke_init, vf_init]
        bounds = ([0.001, 0.001, 1], [1, 1, 200])
        
        popt, pcov = curve_fit(model_3param, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, ke_fit, vf_fit = popt
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fixed)
        r_squared = calculate_r_squared(concentrations, pred_conc)
        
        result = {
            'method': '3参数优化(固定tlag)',
            'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fit, 'tlag': tlag_fixed,
            'pred_conc': pred_conc, 'r_squared': r_squared,
            'success': True, 'n_params': 3
        }
        
        print(f"3参数(固定tlag={tlag_fixed:.2f}): R²={r_squared:.4f}")
        return result
        
    except Exception as e:
        print(f"3参数优化(固定tlag)失败: {e}")
        return {'success': False, 'method': '3参数优化(固定tlag)'}

def fit_3param_fixed_vf(ka_init, ke_init, vf_fixed, tlag_init):
    """固定V/F，优化3参数 (ka, ke, tlag)"""
    def model_3param(t, ka, ke, tlag):
        return one_compartment_oral_model(t, ka, ke, vf_fixed, tlag)
    
    try:
        initial_guess = [ka_init, ke_init, tlag_init]
        bounds = ([0.001, 0.001, 0], [1, 1, 30])
        
        popt, pcov = curve_fit(model_3param, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, ke_fit, tlag_fit = popt
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fixed, tlag_fit)
        r_squared = calculate_r_squared(concentrations, pred_conc)
        
        result = {
            'method': '3参数优化(固定V/F)',
            'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fixed, 'tlag': tlag_fit,
            'pred_conc': pred_conc, 'r_squared': r_squared,
            'success': True, 'n_params': 3
        }
        
        print(f"3参数(固定V/F={vf_fixed:.2f}): R²={r_squared:.4f}")
        return result
        
    except Exception as e:
        print(f"3参数优化(固定V/F)失败: {e}")
        return {'success': False, 'method': '3参数优化(固定V/F)'}

def fit_3param_fixed_ke(ka_init, ke_fixed, vf_init, tlag_init):
    """固定ke，优化3参数 (ka, V/F, tlag)"""
    def model_3param(t, ka, vf, tlag):
        return one_compartment_oral_model(t, ka, ke_fixed, vf, tlag)
    
    try:
        initial_guess = [ka_init, vf_init, tlag_init]
        bounds = ([0.001, 1, 0], [1, 200, 30])
        
        popt, pcov = curve_fit(model_3param, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, vf_fit, tlag_fit = popt
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fixed, vf_fit, tlag_fit)
        r_squared = calculate_r_squared(concentrations, pred_conc)
        
        result = {
            'method': '3参数优化(固定ke)',
            'ka': ka_fit, 'ke': ke_fixed, 'vf': vf_fit, 'tlag': tlag_fit,
            'pred_conc': pred_conc, 'r_squared': r_squared,
            'success': True, 'n_params': 3
        }
        
        print(f"3参数(固定ke={ke_fixed:.4f}): R²={r_squared:.4f}")
        return result
        
    except Exception as e:
        print(f"3参数优化(固定ke)失败: {e}")
        return {'success': False, 'method': '3参数优化(固定ke)'}

def fit_2param_ka_ke(ka_init, ke_init, vf_fixed, tlag_fixed):
    """优化2参数 (ka, ke)，固定V/F和tlag"""
    def model_2param(t, ka, ke):
        return one_compartment_oral_model(t, ka, ke, vf_fixed, tlag_fixed)
    
    try:
        initial_guess = [ka_init, ke_init]
        bounds = ([0.001, 0.001], [1, 1])
        
        popt, pcov = curve_fit(model_2param, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, ke_fit = popt
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fixed, tlag_fixed)
        r_squared = calculate_r_squared(concentrations, pred_conc)
        
        result = {
            'method': '2参数优化(ka,ke)',
            'ka': ka_fit, 'ke': ke_fit, 'vf': vf_fixed, 'tlag': tlag_fixed,
            'pred_conc': pred_conc, 'r_squared': r_squared,
            'success': True, 'n_params': 2
        }
        
        print(f"2参数(ka,ke): R²={r_squared:.4f}")
        return result
        
    except Exception as e:
        print(f"2参数优化失败: {e}")
        return {'success': False, 'method': '2参数优化(ka,ke)'}

def calculate_r_squared(observed, predicted):
    """计算R²值"""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)

def analyze_optimization_results(results):
    """分析优化结果"""
    print(f"\n{'方法':<20} {'成功':<6} {'R²':<8} {'参数数':<6} {'ka':<8} {'ke':<8} {'V/F':<8} {'tlag':<8}")
    print("-" * 80)
    
    for key, result in results.items():
        if result['success']:
            print(f"{result['method']:<20} {'是':<6} {result['r_squared']:<8.4f} {result['n_params']:<6} "
                  f"{result['ka']:<8.4f} {result['ke']:<8.4f} {result['vf']:<8.2f} {result['tlag']:<8.2f}")
        else:
            print(f"{result['method']:<20} {'否':<6} {'N/A':<8} {'N/A':<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

def select_best_result(results):
    """选择最优结果"""
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if not successful_results:
        return None, None
    
    # 按R²排序，选择最高的
    best_key = max(successful_results.keys(), key=lambda k: successful_results[k]['r_squared'])
    return best_key, successful_results[best_key]

def print_result_details(result):
    """打印详细结果"""
    if result:
        print(f"方法: {result['method']}")
        print(f"吸收速率常数 ka = {result['ka']:.4f} min^-1")
        print(f"消除速率常数 ke = {result['ke']:.4f} min^-1")
        print(f"表观分布容积 V/F = {result['vf']:.2f} L")
        print(f"滞后时间 tlag = {result['tlag']:.2f} min")
        print(f"拟合优度 R² = {result['r_squared']:.4f}")
        print(f"优化参数数量 = {result['n_params']}")

def nonlinear_fitting():
    """
    原始非线性拟合法（保持向后兼容）
    """
    # 获取残差法的初始估算值作为起始点
    ke_init, ka_init, tlag_init, vf_init, _, _ = improved_feathering_method()
    
    # 设置初始参数猜测值
    initial_guess = [ka_init, ke_init, vf_init, tlag_init]
    
    # 设置参数边界（防止参数取不合理的值）
    bounds = ([0.001, 0.001, 1, 0],      # 下界: ka, ke, V/F, tlag
              [1, 1, 200, 30])           # 上界
    
    try:
        # 执行非线性拟合
        popt, pcov = curve_fit(one_compartment_oral_model, time_points, concentrations, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        ka_fit, ke_fit, vf_fit, tlag_fit = popt  # 拟合得到的最优参数
        
        # 计算拟合的预测浓度
        pred_conc = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fit)
        
        # 计算拟合优度 R²
        ss_res = np.sum((concentrations - pred_conc) ** 2)  # 残差平方和
        ss_tot = np.sum((concentrations - np.mean(concentrations)) ** 2)  # 总平方和
        r_squared = 1 - (ss_res / ss_tot)  # R² = 1 - SSres/SStot
        
        print(f"\n非线性拟合结果:")
        print(f"吸收速率常数 ka = {ka_fit:.4f} min^-1")
        print(f"消除速率常数 ke = {ke_fit:.4f} min^-1")
        print(f"表观分布容积 V/F = {vf_fit:.2f} L")
        print(f"滞后时间 tlag = {tlag_fit:.2f} min")
        print(f"拟合优度 R² = {r_squared:.4f}")
        
        return ka_fit, ke_fit, vf_fit, tlag_fit, pred_conc, r_squared
        
    except Exception as e:
        print(f"非线性拟合失败: {e}")
        return None

# 执行参数估算和结果比较
print("="*50)
print("方法1: 改进的残差法 (Improved Feathering Method)")
print("="*50)
ke_res, ka_res, tlag_res, vf_res, extrap_conc, residuals = improved_feathering_method()


print("\n" + "="*50)
print("方法2: 综合非线性拟合法 (Comprehensive Nonlinear Fitting)")
print("="*50)

# 执行综合非线性拟合分析
comprehensive_result = nonlinear_fitting_comprehensive()

if comprehensive_result and comprehensive_result['success']:
    ka_fit = comprehensive_result['ka']
    ke_fit = comprehensive_result['ke']
    vf_fit = comprehensive_result['vf']
    tlag_fit = comprehensive_result['tlag']
    pred_conc_fit = comprehensive_result['pred_conc']
    r_squared = comprehensive_result['r_squared']
    
    # 创建对比图表
    plt.figure(figsize=(15, 10))
    
    # 子图1: 观测值 vs 预测值对比
    plt.subplot(2, 2, 1)
    plt.plot(time_points, concentrations, 'ko-', label='观测值', markersize=8)
    
    # 残差法预测结果
    pred_conc_res = one_compartment_oral_model(time_points, ka_res, ke_res, vf_res, tlag_res)
    plt.plot(time_points, pred_conc_res, 'r--', label=f'残差法预测', linewidth=2)
    
    # 综合优化预测结果
    plt.plot(time_points, pred_conc_fit, 'b-', label=f'综合优化 (R²={r_squared:.3f})', linewidth=2)
    
    plt.axvline(x=tlag_fit, color='blue', linestyle=':', alpha=0.7, label=f'综合优化tlag={tlag_fit:.1f}min')
    plt.axvline(x=tlag_res, color='red', linestyle=':', alpha=0.7, label=f'残差法tlag={tlag_res:.1f}min')
    plt.xlabel('时间 (min)')
    plt.ylabel('浓度 (μg/L)')
    plt.title('模型拟合效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 残差分析
    plt.subplot(2, 2, 2)
    residuals_fit = concentrations - pred_conc_fit
    plt.plot(time_points, residuals_fit, 'bo-', label='拟合残差')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (min)')
    plt.ylabel('残差 (μg/L)')
    plt.title('残差分析（检验拟合质量）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 半对数图对比
    plt.subplot(2, 2, 3)
    plt.semilogy(time_points, concentrations, 'ko-', label='观测值', markersize=8)
    plt.semilogy(time_points, pred_conc_fit, 'b-', label='综合优化', linewidth=2)
    plt.xlabel('时间 (min)')
    plt.ylabel('浓度 (μg/L, 对数尺度)')
    plt.title('半对数尺度对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 参数对比柱状图
    plt.subplot(2, 2, 4)
    methods = ['残差法', '综合优化']
    ka_values = [ka_res, ka_fit]
    ke_values = [ke_res, ke_fit]
    tlag_values = [tlag_res, tlag_fit]
    
    x = np.arange(len(methods))
    width = 0.25
    
    plt.bar(x - width, ka_values, width, label='ka (min^-1)', alpha=0.8)
    plt.bar(x, ke_values, width, label='ke (min^-1)', alpha=0.8)
    plt.bar(x + width, [t/100 for t in tlag_values], width, label='tlag/100 (min)', alpha=0.8)
    
    plt.xlabel('估算方法')
    plt.ylabel('参数值')
    plt.title('两种方法的参数估算对比')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 添加详细的参数对比表格
    print("\n" + "="*80)
    print("详细参数对比表格 (残差法 vs 综合非线性拟合)")
    print("="*80)
    
    # 计算残差法的其他参数
    t_half_res = np.log(2) / ke_res  # 消除半衰期
    cl_f_res = vf_res * ke_res       # 表观清除率
    pred_conc_res = one_compartment_oral_model(time_points, ka_res, ke_res, vf_res, tlag_res)
    ss_res_res = np.sum((concentrations - pred_conc_res) ** 2)
    ss_tot = np.sum((concentrations - np.mean(concentrations)) ** 2)
    r_squared_res = 1 - (ss_res_res / ss_tot)
    
    # 计算非线性拟合的其他参数
    t_half_fit = np.log(2) / ke_fit  # 消除半衰期
    cl_f_fit = vf_fit * ke_fit       # 表观清除率
    
    # 计算参数差异（相对差异百分比）
    ka_diff = abs(ka_fit - ka_res) / ka_res * 100 if ka_res != 0 else 0
    ke_diff = abs(ke_fit - ke_res) / ke_res * 100 if ke_res != 0 else 0
    vf_diff = abs(vf_fit - vf_res) / vf_res * 100 if vf_res != 0 else 0
    tlag_diff = abs(tlag_fit - tlag_res) / tlag_res * 100 if tlag_res != 0 else 0
    t_half_diff = abs(t_half_fit - t_half_res) / t_half_res * 100 if t_half_res != 0 else 0
    cl_f_diff = abs(cl_f_fit - cl_f_res) / cl_f_res * 100 if cl_f_res != 0 else 0
    r2_diff = abs(r_squared - r_squared_res) / r_squared_res * 100 if r_squared_res != 0 else 0
    
    # 创建对比表格
    comparison_data = {
        '参数': ['ka (min⁻¹)', 'ke (min⁻¹)', 'V/F (L)', 'tlag (min)', 't₁/₂ (min)', 'CL/F (L/min)', 'R²'],
        '残差法': [f'{ka_res:.4f}', f'{ke_res:.4f}', f'{vf_res:.2f}', f'{tlag_res:.2f}', 
                  f'{t_half_res:.2f}', f'{cl_f_res:.4f}', f'{r_squared_res:.4f}'],
        '非线性拟合': [f'{ka_fit:.4f}', f'{ke_fit:.4f}', f'{vf_fit:.2f}', f'{tlag_fit:.2f}', 
                     f'{t_half_fit:.2f}', f'{cl_f_fit:.4f}', f'{r_squared:.4f}'],
        '相对差异(%)': [f'{ka_diff:.1f}%', f'{ke_diff:.1f}%', f'{vf_diff:.1f}%', f'{tlag_diff:.1f}%', 
                      f'{t_half_diff:.1f}%', f'{cl_f_diff:.1f}%', f'{r2_diff:.1f}%']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 打印表格
    print(comparison_df.to_string(index=False, justify='center'))
    
    # 添加方法评价
    print(f"\n" + "="*60)
    print("方法评价与建议:")
    print("="*60)
    
    if r_squared > r_squared_res:
        print(f"✓ 非线性拟合的R²更高 ({r_squared:.4f} vs {r_squared_res:.4f})")
        print("  建议采用非线性拟合结果")
    else:
        print(f"✓ 残差法的R²更高 ({r_squared_res:.4f} vs {r_squared:.4f})")
        print("  建议采用残差法结果")
    
    # 参数稳定性评价
    high_diff_params = []
    if ka_diff > 20: high_diff_params.append(f'ka ({ka_diff:.1f}%)')
    if ke_diff > 20: high_diff_params.append(f'ke ({ke_diff:.1f}%)')
    if vf_diff > 20: high_diff_params.append(f'V/F ({vf_diff:.1f}%)')
    if tlag_diff > 20: high_diff_params.append(f'tlag ({tlag_diff:.1f}%)')
    
    if high_diff_params:
        print(f"\n⚠️  以下参数在两种方法间差异较大 (>20%):")
        for param in high_diff_params:
            print(f"   - {param}")
        print("   建议进一步验证或使用更多数据点")
    else:
        print(f"\n✓ 两种方法的参数估算结果相对一致 (差异<20%)")
        print("  参数估算较为可靠")
    
    # 输出最终推荐参数
    print("\n" + "="*50)
    print("推荐参数 (基于综合优化结果):")
    print("="*50)
    print(f"吸收速率常数 (ka): {ka_fit:.4f} min^-1")
    print(f"消除速率常数 (ke): {ke_fit:.4f} min^-1")
    print(f"表观分布容积 (V/F): {vf_fit:.2f} L")
    print(f"吸收滞后时间 (tlag): {tlag_fit:.2f} min")
    print(f"拟合优度 (R²): {r_squared:.4f}")
    
    # 计算其他重要的药动学参数
    t_half = np.log(2) / ke_fit  # 消除半衰期
    cl_f = vf_fit * ke_fit       # 表观清除率
    
    print(f"\n其他药动学参数:")
    print(f"消除半衰期 (t1/2): {t_half:.2f} min")
    print(f"表观清除率 (CL/F): {cl_f:.4f} L/min")

else:
    print("综合优化失败，使用残差法结果")
    ka_fit, ke_fit, vf_fit, tlag_fit = ka_res, ke_res, vf_res, tlag_res
    pred_conc_fit = one_compartment_oral_model(time_points, ka_fit, ke_fit, vf_fit, tlag_fit)
    
    # 计算R²
    ss_res = np.sum((concentrations - pred_conc_fit) ** 2)
    ss_tot = np.sum((concentrations - np.mean(concentrations)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"使用残差法参数: ka={ka_fit:.4f}, ke={ke_fit:.4f}, V/F={vf_fit:.2f}, tlag={tlag_fit:.2f}")
    print(f"拟合优度 R² = {r_squared:.4f}")
