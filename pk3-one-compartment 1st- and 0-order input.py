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
零级输入一室模型 (Zero-Order Input One-Compartment Model) 原理

该模型描述了药物以恒定速率 (Rin) 吸收进入体循环，并在体内以一级动力学 (ke) 消除的过程。
假设药物在体内迅速分布到一个单一的、均匀的房室中 (一室模型)。

主要参数:
  Rin: 药物的零级输入速率 (例如: mg/min)。
  V: 药物的分布容积 (例如: L)。
  ke: 药物的消除速率常数 (例如: 1/min)。
  Tabs: 零级吸收的持续时间 (例如: min)。
  C: 血浆药物浓度 (例如: mg/L)。

微分方程 (描述血浆药物浓度 C 随时间 t 的变化率):

1. 吸收期间 (当 t < Tabs):
   药物以恒定速率 Rin 进入体循环，同时以一级动力学被消除。
   dC/dt = (Rin / V) - (ke * C)
   意义: 浓度变化率 = 药物进入速率 (转换为浓度变化率) - 药物消除速率

2. 消除期间 (当 t >= Tabs):
   零级吸收停止，体内不再有新的药物进入，只有体内已有的药物继续以一级动力学被消除。
   dC/dt = - (ke * C)
   意义: 浓度变化率 = - 药物消除速率

最终的积分形式 (公式 3:1) 将这两个阶段合并，用于直接计算任意时间点 t 的血药浓度。
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

# 计算AUC
auc_value = calculate_auc_trapezoidal(time_points, concentrations)
print(f"使用线性梯形法计算得到的AUC (0-t) = {auc_value:.2f} μg·h·L⁻¹")

# 计算CL_oral
cl_oral = dose_oral / auc_value
print(f"口服清除率 (CL_oral) = {dose_oral:.2f} μg / {auc_value:.2f} μg·h·L⁻¹ = {cl_oral:.2f} L·h⁻¹")


def one_compartment_zero_order_input_ode(t, y, ke, v_f, tlag, t_abs, dose_oral):
    """
    一室口服零级吸收模型的微分方程
    
    参数:
        t: 时间
        y: 状态变量 [Aa, Ac] (吸收室药量, 中央室药量)
        ke: 消除速率常数 (1/h)
        v_f: 表观分布容积 (L)
        tlag: 滞后时间 (h)
        t_abs: 吸收持续时间 (h)
        dose_oral: 口服剂量 (μg)
    
    返回:
        dydt: 状态变量的变化率
    """
    Aa, Ac = y # 输出的y是一个numpy多维数组，每个元素是一个状态变量的数组
    
    # 三个阶段的微分方程
    if t < tlag:
        # 阶段1: t < t_lag，浓度为0
        dAa_dt = 0.0
        dAc_dt = 0.0
    elif t < tlag + t_abs:
        # 阶段2: t_lag <= t < t_lag + t_abs，零级吸收 + 消除
        Rin = dose_oral / t_abs  # 零级输入速率 (μg/h)
        dAa_dt = -Rin  # 吸收室药量减少
        dAc_dt = Rin - ke * Ac  # 中央室：吸收 - 消除
    else:
        # 阶段3: t >= t_lag + t_abs，只有消除
        dAa_dt = 0.0
        dAc_dt = -ke * Ac
    
    return [dAa_dt, dAc_dt]

def predict_concentration_zero_order(t, ke, v_f, tlag, t_abs):
    """
    预测函数：用于curve_fit的零级吸收模型
    
    参数:
        t: 时间点数组
        ke: 消除速率常数 (1/h)
        v_f: 表观分布容积 (L)
        tlag: 滞后时间 (h)
        t_abs: 吸收持续时间 (h)
    
    返回:
        predicted_conc: 预测的浓度数组
    """
    # 初始条件
    y0 = [dose_oral, 0.0]  # [Aa(0), Ac(0)]
    
    # 时间范围
    t_span = (0, max(t))
    
    # 求解ODE
    try:
        sol = solve_ivp(
            lambda t_val, y_val: one_compartment_zero_order_input_ode(
                t_val, y_val, ke, v_f, tlag, t_abs, dose_oral
            ),
            t_span, y0, t_eval=t, method='RK45', rtol=1e-8, atol=1e-10
        )
        
        if sol.success:
            # 计算浓度 = 中央室药量 / 表观分布容积
            predicted_conc = sol.y[1] / v_f
            return predicted_conc
        else:
            # 如果求解失败，返回零数组
            return np.zeros_like(t)
    except:
        return np.zeros_like(t)

def fit_zero_order_parameters():
    """
    使用curve_fit拟合零级吸收模型的参数
    """
    print("\n" + "="*60)
    print("开始拟合零级吸收模型参数...")
    print("="*60)
    
    # 获取实验数据
    time_data, conc_data = load_data(plot_data=False)
    
    # 参数的初始猜测值
    # ke: 消除速率常数 (1/h)，通常在0.1-2.0之间
    # v_f: 表观分布容积 (L)，通常在10-1000之间
    # tlag: 滞后时间 (h)，通常在0-2之间
    # t_abs: 吸收持续时间 (h)，通常在1-8之间
    initial_guess = [0.5, 100, 0.5, 3.0]  # [ke, v_f, tlag, t_abs]
    
    # 参数边界 (下界, 上界)
    bounds = (
        [0.01, 1.0, 0.0, 0.1],    # 下界
        [3.0, 1000.0, 2.0, 10.0]  # 上界
    )
    
    try:
        # 执行参数拟合
        popt, pcov = curve_fit(
            predict_concentration_zero_order,
            time_data, conc_data,
            p0=initial_guess,
            bounds=bounds,
            method='trf',  # Trust Region Reflective算法
            maxfev=5000
        )
        
        # 提取拟合参数
        ke_fit, v_f_fit, tlag_fit, t_abs_fit = popt
        
        # 计算参数的标准误差
        param_errors = np.sqrt(np.diag(pcov))
        ke_err, v_f_err, tlag_err, t_abs_err = param_errors
        
        # 计算拟合质量指标
        predicted_conc = predict_concentration_zero_order(time_data, *popt)
        residuals = conc_data - predicted_conc
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((conc_data - np.mean(conc_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # 输出拟合结果
        print(f"\n拟合成功！")
        print(f"拟合参数结果:")
        print(f"  ke (消除速率常数) = {ke_fit:.4f} ± {ke_err:.4f} h⁻¹")
        print(f"  V_F (表观分布容积) = {v_f_fit:.2f} ± {v_f_err:.2f} L")
        print(f"  t_lag (滞后时间) = {tlag_fit:.4f} ± {tlag_err:.4f} h")
        print(f"  t_abs (吸收持续时间) = {t_abs_fit:.4f} ± {t_abs_err:.4f} h")
        
        print(f"\n拟合质量评估:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  RMSE = {rmse:.2f} μg/L")
        
        # 计算其他药动学参数
        cl_fit = ke_fit * v_f_fit  # 清除率
        t_half_fit = 0.693 / ke_fit  # 半衰期
        
        print(f"\n计算得到的药动学参数:")
        print(f"  CL (清除率) = {cl_fit:.2f} L/h")
        print(f"  t₁/₂ (半衰期) = {t_half_fit:.2f} h")
        
        # 可视化拟合结果
        visualize_fit_results(time_data, conc_data, popt)
        
        # 可视化残差分布
        visualize_residuals(time_data, conc_data, popt)
        
        return popt, pcov, r_squared, rmse
        
    except Exception as e:
        print(f"拟合失败: {str(e)}")
        return None, None, None, None

def visualize_fit_results(time_data, conc_data, popt):
    """
    可视化拟合结果
    """
    # 生成更密集的时间点用于绘制平滑曲线
    t_fine = np.linspace(0, max(time_data), 200)
    
    # 计算拟合曲线
    conc_fit = predict_concentration_zero_order(t_fine, *popt)
    conc_data_fit = predict_concentration_zero_order(time_data, *popt)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制实验数据点
    plt.scatter(time_data, conc_data, color='red', s=80, alpha=0.8, 
                edgecolors='darkred', linewidth=1, label='实验数据', zorder=5)
    
    # 绘制拟合曲线
    plt.plot(t_fine, conc_fit, 'b-', linewidth=2, label='零级吸收模型拟合', alpha=0.8)
    
    # 绘制残差
    residuals = conc_data - conc_data_fit
    for i, (t, obs, pred) in enumerate(zip(time_data, conc_data, conc_data_fit)):
        plt.plot([t, t], [obs, pred], 'gray', alpha=0.5, linewidth=1)
    
    # 图形美化
    plt.xlabel('时间 (h)', fontsize=12)
    plt.ylabel('浓度 (μg/L)', fontsize=12)
    plt.title('零级吸收一室模型参数拟合结果', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(time_data) * 1.1)
    plt.ylim(0, max(conc_data) * 1.1)
    
    # 添加拟合参数信息
    ke_fit, v_f_fit, tlag_fit, t_abs_fit = popt
    info_text = f'ke = {ke_fit:.4f} h⁻¹\nV_F = {v_f_fit:.1f} L\nt_lag = {tlag_fit:.3f} h\nt_abs = {t_abs_fit:.3f} h'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_residuals(time_data, conc_data, popt):
    """
    可视化残差分布情况
    """
    # 计算残差
    conc_pred = predict_concentration_zero_order(time_data, *popt)
    residuals = conc_data - conc_pred
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 残差随时间的分布
    ax1.scatter(time_data, residuals, color='blue', alpha=0.7, s=60)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('时间 (h)')
    ax1.set_ylabel('残差 (μg/L)')
    ax1.set_title('残差随时间的分布')
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差随预测值的分布
    ax2.scatter(conc_pred, residuals, color='green', alpha=0.7, s=60)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('预测浓度 (μg/L)')
    ax2.set_ylabel('残差 (μg/L)')
    ax2.set_title('残差随预测值的分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差的直方图
    ax3.hist(residuals, bins=8, color='orange', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('残差 (μg/L)')
    ax3.set_ylabel('频数')
    ax3.set_title('残差分布直方图')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q图 (正态性检验)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('残差正态性Q-Q图')
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Shapiro-Wilk正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # 在图上添加统计信息
    stats_text = f'残差统计:\n均值: {mean_residual:.3f}\n标准差: {std_residual:.3f}\nShapiro-Wilk p值: {shapiro_p:.3f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 输出残差分析结果
    print(f"\n============================================================")
    print(f"残差分析结果:")
    print(f"============================================================")
    print(f"残差均值: {mean_residual:.4f} μg/L")
    print(f"残差标准差: {std_residual:.4f} μg/L")
    print(f"Shapiro-Wilk正态性检验 p值: {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("残差符合正态分布 (p > 0.05)")
    else:
        print("残差不符合正态分布 (p ≤ 0.05)")

# 执行参数拟合
if __name__ == "__main__":
    fit_results = fit_zero_order_parameters()


