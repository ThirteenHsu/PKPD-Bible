import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression 
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
objectives:
• To fit mono-, bi- and tri-exponential models to intravenous bolus data
• To apply the model library
• To apply non-compartmental analysis, curve stripping, and nonlinear regression


"""

dose = 100 # μg iv bolus

def load_data(plot_data=True):
    # Original data
    data = {
        'Time (h)': [5.56, 10.51, 15.15, 20.10, 30.32, 45.80, 60.66, 90.39, 121.06, 150.48, 180.53, 240.01, 301.35, 360.20],
        'C_iv (μg·L⁻¹)': [1.63, 1.41, 1.32, 1.13, 0.98, 0.81, 0.75, 0.59, 0.54, 0.47, 0.42, 0.35, 0.33, 0.25],
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
        plt.yticks([2, 1, 0.1, 0.02], labels=['2', '1', '0.1', '0.02'])
        plt.title('C_iv vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="-", color='0.7', alpha=0.3)
        plt.show()
    return time_points, concentrations_iv


# Helper function for trapezoidal rule
def calculate_trapezoidal_area(x, y):
    area = 0
    for i in range(len(x) - 1):
        area += (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2
    return area

def nca_analysis_statistical_moment(time_points, concentrations):
    """
    执行非房室分析 (NCA) 以计算药代动力学 (PK) 参数。

    该函数根据给定的时间和浓度数据，计算以下 PK 参数：
    - 终端消除速率常数 (λ): 通过对末端消除相的对数浓度-时间数据进行线性回归获得。
    - 终端消除相 R²: 衡量末端消除相线性回归拟合优度的统计量。
    - 零到末次采样点曲线下面积 (AUC(0-last)): 使用线性/对数梯形法计算。
    - 零到无穷大曲线下面积 (AUC(0-∞)): AUC(0-last) + C_last / λ。
    - 零到末次采样点矩量曲线下面积 (AUMC(0-last)): 使用线性/对数梯形法计算。
    - 零到无穷大矩量曲线下面积 (AUMC(0-∞)): AUMC(0-last) + t_last * C_last / λ + C_last / λ**2。
    - 清除率 (CL): Dose / AUC(0-∞)。
    - 平均滞留时间 (MRT): AUMC(0-∞) / AUC(0-∞)。
    - 稳态分布容积 (Vss): CL * MRT。

    参数:
    time_points (array-like): 药物浓度采样的时间点。
    concentrations (array-like): 对应时间点的药物浓度。
    """
    # 将浓度数据转换为 NumPy 数组，并确保时间点和浓度数据长度一致
    concentrations = np.array(concentrations)
    time_points = np.array(time_points)

    # 确保数据已按时间排序，这是进行梯形法计算和线性回归的前提
    sorted_indices = np.argsort(time_points)
    time_points = time_points[sorted_indices]
    concentrations = concentrations[sorted_indices]

    # 1. 计算 λ (终端消除速率常数)
    # λ 描述了药物在消除相从体内清除的速率。通过对浓度-时间曲线的末端部分进行线性回归（半对数图），
    # 斜率的绝对值即为 λ。
    # 首先，识别终端消除相：通常是浓度-时间曲线末尾的几个点。
    # 这里我们假设使用最后3个非零浓度点进行线性回归。
    if len(time_points) < 3:
        print("数据点不足，无法计算 λ。")
        return

    # 找到非零浓度值及其对应的时间点，因为 ln(0) 是未定义的。
    non_zero_concentrations_indices = concentrations > 0
    log_concentrations = np.log(concentrations[non_zero_concentrations_indices])
    terminal_time_points = time_points[non_zero_concentrations_indices]

    if len(terminal_time_points) < 3:
        print("非零浓度数据点不足，无法计算 λ。")
        return

    # 尝试使用最后3个点进行回归，如果可用点少于3个，则使用所有可用点。
    num_points_for_lambda = min(3, len(terminal_time_points))
    terminal_time_points_for_lambda = terminal_time_points[-num_points_for_lambda:]
    log_concentrations_for_lambda = log_concentrations[-num_points_for_lambda:]

    # 执行线性回归，获取斜率、截距、R值等。
    slope, intercept, r_value, p_value, std_err = linregress(terminal_time_points_for_lambda, log_concentrations_for_lambda)
    
    # λ 是斜率的绝对值。
    lam = -slope
    print(f"终端消除速率常数 (λ): {lam:.4f} h⁻¹")
    print(f"终端消除相 R²: {r_value**2:.4f}")

    # 2. 计算 AUC(0-last) - 零到末次采样点曲线下面积
    # 使用线性梯形法计算从时间零点到最后一个可测量浓度点之间的曲线下面积。
    auc_0_last = calculate_trapezoidal_area(time_points, concentrations)
    print(f"AUC(0-last): {auc_0_last:.4f}")

    # 3. 计算 AUMC(0-last) - 零到末次采样点矩量曲线下面积
    # AUMC 是时间乘以浓度对时间的积分，用于计算平均滞留时间 (MRT)。
    # 同样使用线性梯形法，但 Y 轴数据是时间点与对应浓度的乘积。
    time_concentration_product = time_points * concentrations
    aumc_0_last = calculate_trapezoidal_area(time_points, time_concentration_product)
    print(f"AUMC(0-last): {aumc_0_last:.4f}")

    # 4. 计算 AUC(0-∞) - 零到无穷大曲线下面积
    # 通过将 AUC(0-last) 加上从最后一个可测量浓度点到无穷大的外推面积来计算。
    # 外推面积假设药物以 λ 的速率继续消除。
    # 公式: AUC(0-∞) = AUC(0-last) + C_last / λ
    C_last = concentrations[-1]
    auc_0_infinity = auc_0_last + C_last / lam
    print(f"AUC(0-∞): {auc_0_infinity:.4f}")

    # 5. 计算 AUMC(0-∞) - 零到无穷大矩量曲线下面积
    # 通过将 AUMC(0-last) 加上从最后一个可测量浓度点到无穷大的外推矩量面积来计算。
    # 公式: AUMC(0-∞) = AUMC(0-last) + (t_last * C_last) / λ + C_last / λ²
    t_last = time_points[-1]
    aumc_0_infinity = aumc_0_last + (t_last * C_last) / lam + C_last / (lam**2)
    print(f"AUMC(0-∞): {aumc_0_infinity:.4f}")

    # 假设剂量为100 mg (需要根据实际情况调整)
    Dose = 100 # mg

    # 6. 计算清除率 (CL)
    print(f"\n计算清除率 (CL):")
    print(f"  公式: CL = Dose / AUC(0-∞)")
    print(f"  Dose = {Dose:.4f}")
    print(f"  AUC(0-∞) = {auc_0_infinity:.4f}")
    CL = Dose / auc_0_infinity
    print(f"清除率 (CL): {CL:.4f} L/h")

    # 7. 计算平均滞留时间 (MRT)
    print(f"\n计算平均滞留时间 (MRT):")
    print(f"  公式: MRT = AUMC(0-∞) / AUC(0-∞)")
    print(f"  AUMC(0-∞) = {aumc_0_infinity:.4f}")
    print(f"  AUC(0-∞) = {auc_0_infinity:.4f}")
    MRT = aumc_0_infinity / auc_0_infinity
    print(f"平均滞留时间 (MRT): {MRT:.4f} h")

    # 8. 计算稳态分布容积 (Vss)
    print(f"\n计算稳态分布容积 (Vss):")
    print(f"  公式: Vss = CL * MRT")
    print(f"  CL = {CL:.4f}")
    print(f"  MRT = {MRT:.4f}")
    Vss = CL * MRT
    print(f"稳态分布容积 (Vss): {Vss:.4f} L")

def curve_stripping_method(time_points, concentrations):
    """
    曲线剥离法

    公式：
        C_plasma(t) = A * exp(-α * t) + B * exp(-β * t)
    
    计算过程：
        1. 绘制半对数图：将血浆药物浓度（C_plasma）对时间（t）绘制在半对数坐标纸上。
        2. 确定终端消除相（β相）：
            - 在半对数图上，浓度-时间曲线的末端通常会呈现一条直线。
            - 选择曲线末端至少3个数据点进行线性回归（ln(C) 对 t），得到斜率的绝对值即为 β。
            - 将这条直线外推至时间零点，得到截距 B。
        3. 计算残差（α相）：
            - 使用 B 和 β，计算每个时间点上 β 相的贡献：C_extrapolated(t) = B * exp(-β * t)。
            - 从每个观察到的血浆浓度 C_observed(t) 中减去 β 相的贡献，得到残差浓度 C_residual(t) = C_observed(t) - C_extrapolated(t)。
        4. 确定快速分布相（α相）：
            - 将残差浓度 C_residual(t) 对时间 t 绘制在另一个半对数图上。
            - 对残差曲线上的点进行线性回归（ln(C_residual) 对 t），得到斜率的绝对值即为 α。
            - 将这条直线外推至时间零点，得到截距 A。
        5. 计算其他药代动力学参数：
            - 零时血药浓度 (C0) = A + B
            - 分布容积 (Vc) = Dose / C0
            - 曲线下面积 (AUC(0-∞)) = A/α + B/β
            - 清除率 (CL) = Dose / AUC(0-∞)
            - 分布相半衰期 (t1/2α) = ln(2) / α
            - 消除相半衰期 (t1/2β) = ln(2) / β
    """
    time_points = np.array(time_points)
    concentrations = np.array(concentrations)

    # 确保浓度为正值，以便进行对数转换
    positive_indices = concentrations > 0
    time_points_positive = time_points[positive_indices]
    concentrations_positive = concentrations[positive_indices]

    if len(concentrations_positive) < 3:
        print("数据点不足，无法进行曲线剥离法分析。")
        return

    # 1. 确定终端消除相（β相）
    # 通常选择最后几个点进行线性回归
    # 假设最后3个点用于 β 相的确定
    num_points_beta = min(3, len(time_points_positive))
    if num_points_beta < 2:
        print("用于 β 相回归的数据点不足。")
        return

    time_beta = time_points_positive[-num_points_beta:]
    conc_beta = concentrations_positive[-num_points_beta:]
    log_conc_beta = np.log(conc_beta)

    # 执行线性回归
    slope_beta, intercept_beta, r_value_beta, _, _ = linregress(time_beta, log_conc_beta)
    beta = -slope_beta
    B = np.exp(intercept_beta)

    # 计算 C_extrapolated，用于绘制 β 相外推线
    C_extrapolated = B * np.exp(-beta * time_points_positive)

    print(f"\n--- 曲线剥离法结果 ---")
    print(f"β (消除速率常数): {beta:.4f} h⁻¹")
    print(f"B (β相截距): {B:.4f} μg·L⁻¹")
    print(f"β相回归 R²: {r_value_beta**2:.4f}")

    # 创建子图
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # 2. 计算残差（α相）
    C_extrapolated = B * np.exp(-beta * time_points_positive)
    C_residual = concentrations_positive - C_extrapolated

    # 过滤掉非正的残差，因为要取对数
    positive_residual_indices = C_residual > 0
    time_alpha = time_points_positive[positive_residual_indices]
    conc_residual_alpha = C_residual[positive_residual_indices]

    if len(conc_residual_alpha) < 2:
        print("残差数据点不足，无法进行 α 相回归。")
        return

    log_conc_residual_alpha = np.log(conc_residual_alpha)
    ax1 = ax[0]
    ax2 = ax[1]

    # 绘制原始数据和β相外推线
    ax1.scatter(time_points, concentrations, color='blue', label='Original Data')
    ax1.plot(time_points, C_extrapolated, color='red', linestyle='--', label='Extrapolated β-phase')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Concentration (log scale)')
    ax1.set_title('Original Data with Extrapolated β-phase')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.7)

    # 仅使用前5个残差点进行线性回归，如果不足5个则使用所有点
    num_points_for_alpha_fit = min(5, len(time_alpha))
    time_alpha_fit = time_alpha[:num_points_for_alpha_fit]
    log_conc_residual_alpha_fit = log_conc_residual_alpha[:num_points_for_alpha_fit]

    slope_alpha, intercept_alpha, r_value_alpha, _, _ = linregress(time_alpha_fit, log_conc_residual_alpha_fit)
    alpha = -slope_alpha
    A = np.exp(intercept_alpha)
    r_squared_alpha = r_value_alpha**2
    print(f"α (alpha): {alpha:.4f}")
    print(f"A: {A:.4f}")
    print(f"α-phase regression R²: {r_squared_alpha:.4f}")

    # 绘制残差图和α相外推线
    ax2.scatter(time_alpha, C_residual[positive_residual_indices], color='green', label='Residuals (C_residual)')
    ax2.plot(time_alpha, A * np.exp(-alpha * time_alpha), color='purple', linestyle='--', label='Extrapolated α-phase')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Concentration (log scale)')
    ax2.set_title('Residual Plot with Extrapolated α-phase')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 4. 计算其他药代动力学参数
    C0 = A + B
    print(f"零时血药浓度 (C0): {C0:.4f} μg·L⁻¹")

    
    Vc = dose / C0
    print(f"分布容积 (Vc): {Vc:.4f} L")

    AUC_0_infinity = A/alpha + B/beta
    print(f"曲线下面积 (AUC(0-∞)): {AUC_0_infinity:.4f} μg·min·L⁻¹")

    CL = dose / AUC_0_infinity
    print(f"清除率 (CL): {CL:.4f} L/min")

    t1_2_alpha = np.log(2) / alpha
    print(f"分布相半衰期 (t1/2α): {t1_2_alpha:.4f} min")

    t1_2_beta = np.log(2) / beta
    print(f"消除相半衰期 (t1/2β): {t1_2_beta:.4f} min")

    return {
        "alpha": alpha, "A": A,
        "beta": beta, "B": B,
        "C0": C0, "Vc": Vc,
        "AUC(0-inf)": AUC_0_infinity,
        "CL": CL,
        "t1/2_alpha": t1_2_alpha,
        "t1/2_beta": t1_2_beta
    }

def nonlinear_regression_method(time_points, concentrations):
    """
    非线性回归法:

    定义三个原方程：
        C_plasma(t) = A * exp(-ke * t)
        C_plasma(t) = A * exp(-alpha * t) + B * exp(-beta * t)
        C_plasma(t) = A * exp(-alpha * t) + B * exp(-beta * t) + C * exp(-gamma * t)

    使用curve_fit进行参数估计，分别对三个原方程进行拟合。
    计算每个原方程的R²值，选择R²值最高的原方程作为最佳拟合模型，并汇报每个模型的参数估计对比结果
    """
    from sklearn.metrics import r2_score
    from scipy import stats
    
    # 定义三个模型函数
    def one_compartment_model(t, A, ke):
        """单室模型: C(t) = A * exp(-ke * t)"""
        return A * np.exp(-ke * t)
    
    def two_compartment_model(t, A, alpha, B, beta):
        """双室模型: C(t) = A * exp(-alpha * t) + B * exp(-beta * t)"""
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t)
    
    def three_compartment_model(t, A, alpha, B, beta, C, gamma):
        """三室模型: C(t) = A * exp(-alpha * t) + B * exp(-beta * t) + C * exp(-gamma * t)"""
        return A * np.exp(-alpha * t) + B * np.exp(-beta * t) + C * np.exp(-gamma * t)
    
    def calculate_model_statistics(y_obs, y_pred, n_params):
        """计算模型统计指标"""
        n = len(y_obs)
        
        # 残差
        residuals = y_obs - y_pred
        
        # 残差平方和 (RSS)
        rss = np.sum(residuals**2)
        
        # 加权残差平方和 (WRSS) - 这里使用1/y_obs²作为权重
        weights = 1.0 / (y_obs**2)
        wrss = np.sum(weights * residuals**2)
        
        # R²
        r2 = r2_score(y_obs, y_pred)
        
        # 调整R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
        
        # 修正的AIC计算 (基于最大似然估计)
        # 假设误差服从正态分布
        sigma2 = rss / n  # 方差估计
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        aic = -2 * log_likelihood + 2 * n_params
        
        # 修正的BIC计算
        bic = -2 * log_likelihood + n_params * np.log(n)
        
        # 标准误差
        mse = rss / (n - n_params)
        se = np.sqrt(mse)
        
        # 条件数 (用于评估参数相关性)
        condition_number = n_params  # 简化版本
        
        return {
            'rss': rss,
            'wrss': wrss,
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'bic': bic,
            'se': se,
            'condition_number': condition_number,
            'residuals': residuals,
            'log_likelihood': log_likelihood
        }
    
    def calculate_confidence_intervals(popt, pcov, alpha=0.05):
        """计算参数的置信区间和变异系数"""
        n_params = len(popt)
        
        # 参数标准误差
        param_errors = np.sqrt(np.diag(pcov))
        
        # t值 (自由度 = 数据点数 - 参数数)
        dof = len(concentrations) - n_params
        t_val = stats.t.ppf(1 - alpha/2, dof)
        
        # 置信区间
        ci_lower = popt - t_val * param_errors
        ci_upper = popt + t_val * param_errors
        
        # 变异系数 (CV%)
        cv_percent = (param_errors / np.abs(popt)) * 100
        
        return {
            'param_errors': param_errors,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cv_percent': cv_percent
        }
    
    # 存储结果
    results = {}
    
    print("\n" + "="*60)
    print("非线性回归分析结果")
    print("="*60)
    
    # 1. 单室模型拟合
    try:
        # 初始猜测值
        p0_1comp = [max(concentrations), 0.01]
        
        popt_1comp, pcov_1comp = curve_fit(one_compartment_model, time_points, concentrations, 
                                          p0=p0_1comp, maxfev=5000)
        
        # 预测值和统计指标
        y_pred_1comp = one_compartment_model(time_points, *popt_1comp)
        stats_1comp = calculate_model_statistics(concentrations, y_pred_1comp, 2)
        ci_1comp = calculate_confidence_intervals(popt_1comp, pcov_1comp)
        
        results['单室模型'] = {
            'parameters': {'A': popt_1comp[0], 'ke': popt_1comp[1]},
            'parameter_errors': ci_1comp,
            'statistics': stats_1comp,
            'predictions': y_pred_1comp,
            'success': True
        }
        
        print(f"\n单室模型拟合结果:")
        print(f"  A = {popt_1comp[0]:.2f} ± {ci_1comp['cv_percent'][0]:.0f}% μg·L⁻¹")
        print(f"  ke = {popt_1comp[1]:.4f} ± {ci_1comp['cv_percent'][1]:.0f}% min⁻¹")
        print(f"  R² = {stats_1comp['r2']:.4f}")
        print(f"  WRSS = {stats_1comp['wrss']:.2f}")
        print(f"  AIC = {stats_1comp['aic']:.0f}")
        print(f"  条件数 = {stats_1comp['condition_number']}")
        print(f"  标准误差 = {stats_1comp['se']:.3f}")
        
    except Exception as e:
        results['单室模型'] = {'success': False, 'error': str(e)}
        print(f"\n单室模型拟合失败: {e}")
    
    # 2. 双室模型拟合
    try:
        # 初始猜测值 (基于curve stripping的结果)
        p0_2comp = [max(concentrations)*0.6, 0.1, max(concentrations)*0.4, 0.01]
        
        popt_2comp, pcov_2comp = curve_fit(two_compartment_model, time_points, concentrations, 
                                          p0=p0_2comp, maxfev=5000)
        
        # 预测值和统计指标
        y_pred_2comp = two_compartment_model(time_points, *popt_2comp)
        stats_2comp = calculate_model_statistics(concentrations, y_pred_2comp, 4)
        ci_2comp = calculate_confidence_intervals(popt_2comp, pcov_2comp)
        
        results['双室模型'] = {
            'parameters': {'A': popt_2comp[0], 'alpha': popt_2comp[1], 
                          'B': popt_2comp[2], 'beta': popt_2comp[3]},
            'parameter_errors': ci_2comp,
            'statistics': stats_2comp,
            'predictions': y_pred_2comp,
            'success': True
        }
        
        print(f"\n双室模型拟合结果:")
        print(f"  A = {popt_2comp[0]:.2f} ± {ci_2comp['cv_percent'][0]:.0f}% μg·L⁻¹")
        print(f"  α = {popt_2comp[1]:.4f} ± {ci_2comp['cv_percent'][1]:.0f}% min⁻¹")
        print(f"  B = {popt_2comp[2]:.2f} ± {ci_2comp['cv_percent'][2]:.0f}% μg·L⁻¹")
        print(f"  β = {popt_2comp[3]:.4f} ± {ci_2comp['cv_percent'][3]:.0f}% min⁻¹")
        print(f"  R² = {stats_2comp['r2']:.4f}")
        print(f"  WRSS = {stats_2comp['wrss']:.4f}")
        print(f"  AIC = {stats_2comp['aic']:.0f}")
        print(f"  条件数 = {stats_2comp['condition_number']}")
        print(f"  标准误差 = {stats_2comp['se']:.3f}")
        
    except Exception as e:
        results['双室模型'] = {'success': False, 'error': str(e)}
        print(f"\n双室模型拟合失败: {e}")
    
    # 3. 三室模型拟合
    try:
        # 初始猜测值
        p0_3comp = [max(concentrations)*0.4, 0.5, max(concentrations)*0.4, 0.1, max(concentrations)*0.2, 0.01]
        
        popt_3comp, pcov_3comp = curve_fit(three_compartment_model, time_points, concentrations, 
                                          p0=p0_3comp, maxfev=5000)
        
        # 预测值和统计指标
        y_pred_3comp = three_compartment_model(time_points, *popt_3comp)
        stats_3comp = calculate_model_statistics(concentrations, y_pred_3comp, 6)
        ci_3comp = calculate_confidence_intervals(popt_3comp, pcov_3comp)
        
        results['三室模型'] = {
            'parameters': {'A': popt_3comp[0], 'alpha': popt_3comp[1], 
                          'B': popt_3comp[2], 'beta': popt_3comp[3],
                          'C': popt_3comp[4], 'gamma': popt_3comp[5]},
            'parameter_errors': ci_3comp,
            'statistics': stats_3comp,
            'predictions': y_pred_3comp,
            'success': True
        }
        
        print(f"\n三室模型拟合结果:")
        print(f"  A = {popt_3comp[0]:.2f} ± {ci_3comp['cv_percent'][0]:.0f}% μg·L⁻¹")
        print(f"  α = {popt_3comp[1]:.4f} ± {ci_3comp['cv_percent'][1]:.0f}% min⁻¹")
        print(f"  B = {popt_3comp[2]:.2f} ± {ci_3comp['cv_percent'][2]:.0f}% μg·L⁻¹")
        print(f"  β = {popt_3comp[3]:.4f} ± {ci_3comp['cv_percent'][3]:.0f}% min⁻¹")
        print(f"  C = {popt_3comp[4]:.2f} ± {ci_3comp['cv_percent'][4]:.0f}% μg·L⁻¹")
        print(f"  γ = {popt_3comp[5]:.4f} ± {ci_3comp['cv_percent'][5]:.0f}% min⁻¹")
        print(f"  R² = {stats_3comp['r2']:.4f}")
        print(f"  WRSS = {stats_3comp['wrss']:.4f}")
        print(f"  AIC = {stats_3comp['aic']:.0f}")
        print(f"  条件数 = {stats_3comp['condition_number']}")
        print(f"  标准误差 = {stats_3comp['se']:.3f}")
        
    except Exception as e:
        results['三室模型'] = {'success': False, 'error': str(e)}
        print(f"\n三室模型拟合失败: {e}")
    
    # 4. 模型比较和最佳模型选择
    print(f"\n" + "="*60)
    print("模型比较结果")
    print("="*60)
    
    successful_models = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_models:
        # 创建模型比较表格
        print(f"\n{'模型':<12} {'R²':<8} {'WRSS':<10} {'AIC':<8} {'残差游程':<8}")
        print("-" * 50)
        
        # 按AIC值排序 (AIC越小越好)
        sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['statistics']['aic'])
        
        for model_name, model_data in sorted_models:
            stats = model_data['statistics']
            # 计算残差游程数 (简化版本)
            residuals = stats['residuals']
            runs = 1
            for i in range(1, len(residuals)):
                if (residuals[i] > 0) != (residuals[i-1] > 0):
                    runs += 1
            
            print(f"{model_name:<12} {stats['r2']:<8.4f} {stats['wrss']:<10.4f} {stats['aic']:<8.0f} {runs:<8}")
        
        best_model_name, best_model_data = sorted_models[0]
        
        # 绘制拟合结果对比图
        plt.figure(figsize=(12, 8))
        
        # 原始数据
        plt.subplot(2, 1, 1)
        plt.scatter(time_points, concentrations, color='red', s=50, label='实验数据', zorder=5)
        
        # 绘制所有成功拟合的模型
        colors = ['blue', 'green', 'orange']
        for i, (model_name, model_data) in enumerate(sorted_models):
            plt.plot(time_points, model_data['predictions'], 
                    color=colors[i % len(colors)], linewidth=2, 
                    label=f'{model_name} (R²={model_data["statistics"]["r2"]:.4f})')
        
        plt.xlabel('时间 (h)')
        plt.ylabel('浓度 (μg·L⁻¹)')
        plt.title('非线性回归模型拟合结果比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 残差图
        plt.subplot(2, 1, 2)
        for i, (model_name, model_data) in enumerate(sorted_models):
            residuals = model_data['statistics']['residuals']
            plt.scatter(time_points, residuals, color=colors[i % len(colors)], 
                       label=f'{model_name}', alpha=0.7)
            plt.plot(time_points, residuals, color=colors[i % len(colors)], 
                       linestyle='--', alpha=0.7, linewidth=1.5)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('时间 (h)')
        plt.ylabel('残差 (μg·L⁻¹)')
        plt.title('残差分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("所有模型拟合均失败!")
    
    return results

# 在这里调用 nca_analysis 函数，确保它在函数定义之后
time_points, C_iv = load_data(plot_data=False)
nca_analysis_statistical_moment(time_points, C_iv)
curve_stripping_method(time_points, C_iv)
nonlinear_regression_method(time_points, C_iv)
