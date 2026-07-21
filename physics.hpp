#pragma once
#include <vector>
#include <cmath>
#include <array>
#include <Eigen/Dense>

// 二维数组类型别名，使用行优先存储 (RowMajor)
using Array2D = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @brief 基于相场格子Boltzmann方法的无容器凝固求解器
 * 
 * 实现论文 "Phase-field based lattice Boltzmann method for containerless freezing" 
 * 中的数值模型，包括：
 * - 相场 Allen-Cahn 方程（追踪气-液界面）
 * - 焓法温度场（处理固-液相变）
 * - 带质量源项的 Navier-Stokes 方程（考虑凝固体积变化）
 * - 流固耦合力（扩散界面法）
 * - 润湿边界条件（接触角）
 */
class LBMSolver {
public:
    static constexpr int q = 9;  // D2Q9 离散速度模型，固定为9个方向

    /**
     * @brief 构造函数
     * @param nx    x方向网格数
     * @param ny    y方向网格数
     * @param gamma 固液密度比 γ = ρ_s / ρ_l
     * @param Ste   Stefan数
     * @param Pr    Prandtl数
     */
    LBMSolver(int nx, int ny, double gamma, double Ste, double Pr);
    
    /**
     * @brief 初始化所有场变量（相场、温度、流场、分布函数）
     * 设置初始液滴形状、温度分布及平衡分布函数
     */
    void initialize_fields();
    
    /**
     * @brief 执行一个完整的时间步：依次更新流场、相场、温度场，并交换缓冲区
     */
    void collide_and_stream();
    
    /**
     * @brief 执行多个时间步
     * @param steps 迭代步数
     */
    void step(int steps);

    // ==================== 结果输出接口 ====================
    int getNx() const { return Nx; }
    int getNy() const { return Ny; }
    const std::vector<double>& getPhi() const { return phi; }   // 相场序参数 φ (0=气体, 1=固液混合物)
    const std::vector<double>& getFs() const { return fs; }     // 固相分数 (0=完全液体, 1=完全固体)
    const std::vector<double>& getFl() const { return fl; }     // 液相分数 (1 - fs)
    const std::vector<double>& getT() const { return T; }       // 温度场 (无量纲)
    const std::vector<double>& getH() const { return H; }       // 总焓场
    const std::vector<double>& getQdot() const { return q_dot; } // 潜热源项 (论文式6)
    const std::vector<double>& getUx() const { return ux; }     // x方向速度
    const std::vector<double>& getUy() const { return uy; }     // y方向速度

private:
    // ==================== 坐标与偏移辅助函数 ====================
    inline int index(int x, int y) const { return y * Nx + x; }                     // 一维索引
    inline int offset(int x, int y, int k) const { return (y * Nx + x) * q + k; }   // 分布函数数组偏移
    
    //core algorithms 
    void compute_macros();              // 从分布函数计算宏观量 (密度、速度、压力，含质量源项修正)
    void update_phase_field();          // 更新相场 (Allen-Cahn方程，式21-26)，用于无相变界面
    void update_temperature();          // 更新温度场 (焓法LB，式27-29)
    // 焓解码 (式8,9)：由总焓 H 与相场 φ 反解 (固相分数 fs, 温度 T)。
    // 潜热按 φ 缩放、显热容含气相贡献，并对无潜热区 (Hl-Hs→0) 做退化保护。
    // 碰撞前与迁移后的宏观重构统一调用它，保证状态方程一致 (避免界面伪凝固)。
    void decode_enthalpy(double H_curr, double phi_v, double& fs_out, double& T_out) const;
    void update_flow_field();           // 更新流场 (Navier-Stokes，式30-36)

    // ==================== 辅助计算函数 ====================
    void compute_fluid_solid_interaction();   // 计算流固耦合力 f (扩散界面法，式20后说明)
    void compute_phase_derivatives();   // 计算化学势 μ_φ 和表面张力 F_s (式11,12)
    void update_mixture_properties();         // 更新混合物的密度、比热、导热系数 (式1) 通过插值计算
    void apply_boundary_conditions();         // 总边界条件调度
    void apply_wetting_bc();                  // 润湿边界条件 (式39-41)
    void apply_velocity_bc();                 // 无滑移速度边界
    void apply_temperature_bc();              // 恒温边界 (底部冷壁T=0，顶部热壁T=1)
    void compute_lambda_and_normal();         // 计算 λ = 4φ(1-φ)/W 及界面法向量 n = ∇φ/|∇φ| (式3)
    void compute_boussinesq_buoyancy();       // 计算 Boussinesq 浮力项 (论文 Section 4.3)


    // ==================== 平衡分布函数 ====================
    // 增加压力参数 p_val
    void compute_feq(double rho, double p_val, double ux, double uy, std::array<double, q>& feq) const; // 流场平衡分布 (式31)
    void compute_geq(double phi, double ux, double uy, std::array<double, q>& geq) const; // 相场平衡分布 (式24)
    void compute_heq(double H, double T, double Cp_ref, double Cp_loc, double ux, double uy, std::array<double, q>& heq) const; // 焓场平衡分布 (式28)

    // ==================== 物理参数（无量纲） ====================
    int Nx, Ny;                 // 网格数
    double dx, dt;              // 格子间距、时间步长 (通常取1.0和0.1)
    double gamma;               // ρ_s/ρ_l，密度比
    double Ste, Pr;         // Stefan数、Prandtl数

    // 三相材料属性（下标 g=气体, l=液体, s=固体）
    double rho_g, rho_l, rho_s; // 密度
    double mu_g, mu_l, mu_s;    // 动力粘度
    double Cp_g, Cp_l, Cp_s;    // 定压比热容
    double k_g, k_l, k_s;       // 导热系数
    double L;                   // 潜热 (无量纲)
    double H_s, H_l;            // 固相、液相焓 (无量纲) H_s = Cp_s * Ts, H_l = Cp_l * Tl
    double Ts, Tl, Tm;          // 固相线温度、液相线温度、熔点 (Tm通常取0.5)
    double sigma;               // 表面张力系数
    double W;                   // 界面厚度 (式13)
    double M;                   // 相场迁移率 (式21后说明)
    double wettingAngle;        // 基底接触角 (弧度)
    double beta, kappa;         // 相场参数: β = 12σ/W, κ = 1.5σW (式13)
    double cs2;                 // 声速平方 c_s^2 = c^2/3, c=dx/dt
    double tau_g;               // 相场弛豫时间（由迁移率 M 决定）；流场/温度场的 τ 在更新内逐点计算
    double Cp_ref;              // 参考比热 (取液体比热 Cp_l)

    // ==================== 宏观场 ====================
    std::vector<double> phi;    // 相场序参数 φ (0=气体, 1=固液混合物)
    std::vector<double> fs;     // 固相分数 (0~1)

    std::vector<double> fl;     // 液相分数 fl = 0  (H<H_s), fl = 1 (H>H_l), fl = (H-H_s)/(H_l-H_s) (H_s<H<H_l)

    std::vector<double> T;      // 温度场   T=H/Cp (H<H_s),T=Ts + (H-H_s)/(H_l-H_s)*(Tl-Ts) (H_s<H<H_l), T=Tl + (H-H_l)/Cp (H>H_l)
    std::vector<double> H;      // 总焓场 H = CpT + Lf_l (式5)，其中f_l是液相分数
    std::vector<double> p;      // 压力场
    std::vector<double> rho_mix; // 混合物密度 (式1)
    std::vector<double> mu_mix;  // 混合物动力粘度
    std::vector<double> k_mix;   // 混合物导热系数
    std::vector<double> Cp_mix;  // 混合物比热
 

    // 速度场
    std::vector<double> ux, uy;           // 修正后的宏观速度
    std::vector<double> ux_star, uy_star; // 未考虑流固耦合的中间速度 (式34)

    // 作用力项
    std::vector<double> Fs_x, Fs_y;       // 表面张力 F_s = μ_φ ∇φ (式11)
    std::vector<double> mu_phi;          // 化学势 μ_φ (式12)
    std::vector<double> Gx, Gy;           // 彻体力 (如浮力)
    std::vector<double> fx, fy;           // 流固耦合力 (式20)

    // 源项与历史变量
    std::vector<double> m_dot;            // 质量源项 ṁ (式19) m_dot= (1-rho_s/rho_l) * ∂fs/∂t
    std::vector<double> div_u;          // 速度散度 ∇·u = m_dot (式19)
    std::vector<double> q_dot;            // 潜热源项 q̇ = -∂(ρLf_l)/∂t (论文式6)
    std::vector<double> S_field;          // 完整质量源项 S = ρṁ + u·∇ρ (式33)
    std::vector<double> phi_ux_prev, phi_uy_prev; // 前一时刻的 φu, 用于计算时间导数 (式37)
    std::vector<double> fs_prev;          // 前一时刻的固相分数, 用于计算 ∂fs/∂t

    // ==================== 相场辅助场 ====================
    std::vector<double> lambda;   // λ = 4φ(1-φ)/W, 用于 Allen-Cahn 方程 (式3)
    std::vector<double> nx_field; // 界面法向量的 x 分量
    std::vector<double> ny_field; // 界面法向量的 y 分量
    std::vector<double> grad_phi_x;//x方向的phi梯度
    std::vector<double> grad_phi_y;//y方向的phi梯度
    std::vector<double> lap_phi;//拉普拉斯项
    
    // ==================== 物理参数：浮力 ====================
    double g_accel;              // 无量纲重力加速度
    double thermal_exp_coeff;    // 体积膨胀系数

    // ==================== 固体速度（流固耦合） ====================
    std::vector<double> ux_solid, uy_solid; // 固体速度（无滑移时全为零）

    // ==================== LBM 分布函数（双缓冲） ====================
    std::vector<double> f[2];  // 流场分布函数 (下标 current, next)
    std::vector<double> g[2];  // 相场分布函数
    std::vector<double> h[2];  // 焓场分布函数
    int current, next;         // 当前时间层和下一时间层的索引
};
