#include "physics.hpp"
#include <algorithm>
#include <iostream>

// ==================== D2Q9 常数（格子Boltzmann标准参数） ====================
static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};   // 离散速度 x 分量
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};   // 离散速度 y 分量
static const double w[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};                                                         // 权重系数 ω_i（论文 Eq.(22)）
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};    // 反弹边界用的反向索引

// ==================== 构造函数：初始化所有参数 ====================
LBMSolver::LBMSolver(int nx, int ny, double gamma_, double Ste_, double Pr_)
    : Nx(nx), Ny(ny), gamma(gamma_), Ste(Ste_), Pr(Pr_),
      dx(1.0), dt(0.1), rho_l(1.0), rho_s(gamma_), L(1.0), cp(1.0), Tm(0.5),
      wettingAngle(30.0 * M_PI / 180.0),
      current(0), next(1),
      phi(nx*ny, 0.0), T(nx*ny, 0.0), fs(nx*ny, 0.0), fs_prev(nx*ny, 0.0),
      rho(nx*ny, 0.0), p(nx*ny, 0.0), ux(nx*ny, 0.0), uy(nx*ny, 0.0),
      Fs_x(nx*ny, 0.0), Fs_y(nx*ny, 0.0), MassSource(nx*ny, 0.0),
      phi_u_prev_x(nx*ny, 0.0), phi_u_prev_y(nx*ny, 0.0)
{
    int total = nx * ny * q;
    f[0].assign(total, 0.0); f[1].assign(total, 0.0);
    g[0].assign(total, 0.0); g[1].assign(total, 0.0);
    h[0].assign(total, 0.0); h[1].assign(total, 0.0);

    double c = dx / dt;          // 格子速度
    cs2 = c * c / 3.0;           // 声速平方 c_s^2 = c^2/3

    tau_f = 0.8;                 // 流场松弛时间（可调）
    tau_g = 0.8;                 // 相场松弛时间
    M = cs2 * (tau_g - 0.5) * dt; // 相场迁移率，见论文 Eq.(21) 后说明

    double nu = cs2 * (tau_f - 0.5) * dt;   // 运动粘度
    double alpha = nu / Pr;                 // 热扩散系数（Pr = ν/α）
    tau_h = alpha / (cs2 * dt) + 0.5;       // 温度场松弛时间（论文 Eq.(27) 后说明）
    tau_h = std::max(tau_h, 0.75);          // 防止数值震荡

    double interfaceWidth = 2.0;   // 界面厚度 W（无量纲）
    double sigma = 0.005;          // 表面张力系数 σ
    // 论文 Eq.(13)：κ = 1.5 σ W, β = 12 σ / W
    beta = 12.0 * sigma / interfaceWidth;
    kappa = 1.5 * sigma * interfaceWidth;
}

// ==================== 初始化相场、温度、流场 ====================
void LBMSolver::initialize_fields() {
    double theta = wettingAngle;
    double baseR = std::min(Nx, Ny) * 0.28;      // 初始液滴基底半径
    double R = baseR / std::sin(theta);          // 液滴球冠半径
    double centerX = Nx * 0.5;
    double verticalShift = std::max(4.0, Ny * 0.08);
    double centerY = -R * std::cos(theta) + verticalShift;  // 圆心坐标（y方向偏移）
    double interfaceWidth = 2.0;

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double dxl = x - centerX;
            double dyl = y - centerY;
            double r = std::sqrt(dxl*dxl + dyl*dyl);
            double dist = r - R;

            // 相场初始化：光滑的 Heaviside 函数（论文 Eq.(45) 类似形式）
            double phi_val;
            if (dist <= -interfaceWidth) phi_val = 1.0;      // 液滴内部
            else if (dist >= interfaceWidth) phi_val = 0.0;  // 气相
            else phi_val = 0.5 * (1.0 - dist/interfaceWidth);
            phi[id] = std::clamp(phi_val, 0.0, 1.0);

            fs[id] = 0.0;           // 初始完全液态
            fs_prev[id] = 0.0;

            // 初始温度：液滴内部为高温（T=1），外部为低温（T=0）
            if (phi[id] > 0.5) T[id] = 1.0;
            else T[id] = 0.0;

            ux[id] = uy[id] = 0.0;
            rho[id] = rho_l;
            p[id] = rho_l * cs2;

            // 初始化分布函数为平衡态（论文 Eq.(24), Eq.(31), Eq.(28) 的平衡分布）
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux[id] + cy[k]*uy[id];
                double u2 = 0.0;
                // 流场平衡分布 f^{eq}（论文 Eq.(31)）
                double feq = w[k] * rho[id] * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                f[0][offset(x,y,k)] = feq;
                f[1][offset(x,y,k)] = feq;

                // 相场平衡分布 g^{eq}（论文 Eq.(24)）
                double geq = w[k] * phi[id] * (1.0 + cu/cs2);
                g[0][offset(x,y,k)] = geq;
                g[1][offset(x,y,k)] = geq;

                // 温度场（总焓）平衡分布 h^{eq}（论文 Eq.(28)）
                // 注意：总焓 H = cp T + L (1-f_s)，论文 Sec.3.2
                double H = cp * T[id] + L * (1.0 - fs[id]);
                double heq;
                if (k == 0)
                    heq = H - cp * T[id] + w[k] * cp * T[id];  // i=0 特殊处理
                else
                    heq = w[k] * cp * T[id];
                h[0][offset(x,y,k)] = heq;
                h[1][offset(x,y,k)] = heq;
            }
        }
    }
    std::fill(phi_u_prev_x.begin(), phi_u_prev_x.end(), 0.0);
    std::fill(phi_u_prev_y.begin(), phi_u_prev_y.end(), 0.0);
}

// ==================== 计算化学势 μ_φ 和表面张力 F_s ====================
// 对应论文 Eq.(12) 和 Eq.(11)
void LBMSolver::compute_phase_derivatives() {
    Eigen::Map<Array2D> phi_map(phi.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);

    Array2D lap_phi = Array2D::Zero(Ny, Nx);
    Array2D grad_x = Array2D::Zero(Ny, Nx);
    Array2D grad_y = Array2D::Zero(Ny, Nx);

    // 中心差分计算梯度（论文 Eq.(38) 的简化实现）
    auto inner = phi_map.block(1, 1, Ny - 2, Nx - 2);
    auto left  = phi_map.block(1, 0, Ny - 2, Nx - 2);
    auto right = phi_map.block(1, 2, Ny - 2, Nx - 2);
    auto down  = phi_map.block(0, 1, Ny - 2, Nx - 2);
    auto up    = phi_map.block(2, 1, Ny - 2, Nx - 2);

    lap_phi.block(1, 1, Ny-2, Nx-2) = (left + right + up + down - 4.0 * inner) / (dx * dx);
    grad_x.block(1, 1, Ny-2, Nx-2) = (right - left) / (2.0 * dx);
    grad_y.block(1, 1, Ny-2, Nx-2) = (up - down) / (2.0 * dx);

    // 化学势 μ_φ = 4β φ(φ-1)(φ-0.5) - κ ∇²φ（论文 Eq.(12)，其中 φ_l=1, φ_s=0, 混合区对称）
    Array2D mu = 4.0 * beta * phi_map * (phi_map - 1.0) * (phi_map - 0.5) - kappa * lap_phi;
    
    // 表面张力 F_s = μ_φ ∇φ（论文 Eq.(11)）
    Fsx_map = mu * grad_x;
    Fsy_map = mu * grad_y;
}

// ==================== 从分布函数计算宏观量 ====================
// 包括密度、速度、压力，以及质量源项对速度的修正（论文 Eq.(34)-(36) 及 Eq.(19)）
void LBMSolver::compute_macros() {
    compute_phase_derivatives();  // 首先更新表面张力（需要当前 φ 的梯度）

    Eigen::Map<Array2D> rho_map(rho.data(), Ny, Nx);
    Eigen::Map<Array2D> p_map(p.data(), Ny, Nx);
    Eigen::Map<Array2D> ux_map(ux.data(), Ny, Nx);
    Eigen::Map<Array2D> uy_map(uy.data(), Ny, Nx);
    Eigen::Map<Array2D> fs_map(fs.data(), Ny, Nx);
    Eigen::Map<Array2D> fsp_map(fs_prev.data(), Ny, Nx);
    Eigen::Map<Array2D> src_map(MassSource.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);

    // 质量源项：ṁ = (1 - ρ_s/ρ_l) ∂f_s/∂t（论文 Eq.(19)）
    Array2D m_dot = (1.0 - gamma) * (fs_map - fsp_map) / dt;
    src_map = rho_map * m_dot;   // S = ρ ṁ（用于论文 Eq.(33)）
    
    fsp_map = fs_map;  // 更新前一时刻的 f_s

    // 从分布函数 f_i 计算零阶矩、一阶矩（临时）
    Array2D rho_temp = Array2D::Zero(Ny, Nx);
    Array2D mom_x = Array2D::Zero(Ny, Nx);
    Array2D mom_y = Array2D::Zero(Ny, Nx);
    Array2D sum_f_neq_0 = Array2D::Zero(Ny, Nx);  // 用于压力计算（论文 Eq.(36)）

    for (int k = 0; k < q; ++k) {
        for(int y = 0; y < Ny; ++y){
            for(int x = 0; x < Nx; ++x){
                double f_val = f[current][offset(x,y,k)];
                rho_temp(y, x) += f_val;
                mom_x(y, x) += f_val * cx[k];
                mom_y(y, x) += f_val * cy[k];
                if (k != 0) {
                    sum_f_neq_0(y, x) += f_val;
                }
            }
        }
    }

    // 密度修正：考虑质量源项（论文 Eq.(34) 的 ρu* 部分类似，此处先计算 ρ）
    rho_map = rho_temp + 0.5 * dt * src_map;
    rho_map = (rho_map < 1e-12).select(1.0, rho_map);

    // 速度 u* = (∑ c_i f_i + 0.5Δt F) / ρ（论文 Eq.(34)）
    ux_map = (mom_x + 0.5 * dt * Fsx_map) / rho_map;
    uy_map = (mom_y + 0.5 * dt * Fsy_map) / rho_map;

    // 固体区域速度强制为零（论文 Sec.2 中提及的浸没边界处理）
    ux_map = (fs_map > 0.5).select(0.0, ux_map);
    uy_map = (fs_map > 0.5).select(0.0, uy_map);
    
    // 计算压力 p（论文 Eq.(36)）
    // 需要梯度信息，先计算压力梯度和密度梯度
    Array2D grad_p_x = Array2D::Zero(Ny, Nx);
    Array2D grad_p_y = Array2D::Zero(Ny, Nx);
    Array2D grad_rho_x = Array2D::Zero(Ny, Nx);
    Array2D grad_rho_y = Array2D::Zero(Ny, Nx);

    for(int y = 1; y < Ny - 1; ++y) {
        for(int x = 0; x < Nx; ++x) {
            int xp = (x + 1) % Nx;
            int xm = (x - 1 + Nx) % Nx;
            grad_p_x(y, x) = (p[index(xp, y)] - p[index(xm, y)]) / (2.0 * dx);
            grad_p_y(y, x) = (p[index(x, y+1)] - p[index(x, y-1)]) / (2.0 * dx);
            grad_rho_x(y, x) = (rho[index(xp, y)] - rho[index(xm, y)]) / (2.0 * dx);
            grad_rho_y(y, x) = (rho[index(x, y+1)] - rho[index(x, y-1)]) / (2.0 * dx);
        }
    }

    double w0 = w[0];
    for(int y = 1; y < Ny - 1; ++y) {
        for(int x = 0; x < Nx; ++x) {
            double ux_loc = ux_map(y, x);
            double uy_loc = uy_map(y, x);
            double u2 = ux_loc * ux_loc + uy_loc * uy_loc;
            
            // 论文 Eq.(33) 中定义的 F̃ = F - ∇p + c_s² ∇ρ
            double F_tilde_x = Fsx_map(y, x) - grad_p_x(y, x) + cs2 * grad_rho_x(y, x);
            double F_tilde_y = Fsy_map(y, x) - grad_p_y(y, y) + cs2 * grad_rho_y(y, x);
            
            double S_val = src_map(y, x);
            // 论文 Eq.(33) 中的 F₀ 分量（k=0 时的强迫项）
            double F0 = w0 * (S_val - (ux_loc * F_tilde_x + uy_loc * F_tilde_y) / cs2);
            double rho_s0 = -rho_map(y, x) * w0 * u2 / (2.0 * cs2);  // 来自平衡分布中的 u² 项
            
            // 压力计算公式（论文 Eq.(36)）
            p_map(y, x) = (cs2 / (1.0 - w0)) * (sum_f_neq_0(y, x) + 0.5 * dt * S_val + tau_f * dt * F0 + rho_s0);
        }
    }

    // 边界处压力简单外推
    for(int x = 0; x < Nx; ++x) {
        p_map(0, x) = p_map(1, x);
        p_map(Ny-1, x) = p_map(Ny-2, x);
    }
}

// ==================== 辅助函数：计算 ∇·u 和 ∂(φu)/∂t ====================
// 用于相场方程中的源项（论文 Eq.(25)）
void LBMSolver::compute_div_u_and_dphiudt(int x, int y, double& div_u, double& dphiux_dt, double& dphiuy_dt) {
    int id = index(x, y);
    int xp = (x+1) % Nx, xm = (x-1+Nx) % Nx;
    int yp = std::min(y+1, Ny-1), ym = std::max(y-1, 0);

    double dux_dx = (ux[index(xp,y)] - ux[index(xm,y)]) / (2.0*dx);
    double duy_dy = (uy[index(x,yp)] - uy[index(x,ym)]) / (2.0*dx);
    div_u = dux_dx + duy_dy;      // ∇·u

    // 时间导数 ∂(φ u)/∂t 用一阶显式欧拉（论文 Eq.(37)）
    double phi_ux = phi[id] * ux[id];
    double phi_uy = phi[id] * uy[id];
    dphiux_dt = (phi_ux - phi_u_prev_x[id]) / dt;
    dphiuy_dt = (phi_uy - phi_u_prev_y[id]) / dt;

    phi_u_prev_x[id] = phi_ux;
    phi_u_prev_y[id] = phi_uy;
}

// ==================== 相场更新（Allen-Cahn 方程） ====================
// 对应论文 Eq.(21)-(26)
void LBMSolver::update_phase_field() {
    std::vector<double> g_post(Nx*Ny*q, 0.0);
    double interfaceWidth = 2.0;

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double div_u, dphiux_dt, dphiuy_dt;
            compute_div_u_and_dphiudt(x, y, div_u, dphiux_dt, dphiuy_dt);
            double ux_loc = ux[id], uy_loc = uy[id];

            // 计算 φ 梯度（用于单位法向量 n）
            int xp = (x + 1) % Nx;
            int xm = (x - 1 + Nx) % Nx;
            int yp = y + 1;
            int ym = y - 1;
            double grad_phi_x = (phi[index(xp, y)] - phi[index(xm, y)]) / (2.0 * dx);
            double grad_phi_y = (phi[index(x, yp)] - phi[index(x, ym)]) / (2.0 * dx);
            double norm_grad = std::sqrt(grad_phi_x * grad_phi_x + grad_phi_y * grad_phi_y);
            double nx_val = 0.0, ny_val = 0.0;
            if (norm_grad > 1e-8) {
                nx_val = grad_phi_x / norm_grad;
                ny_val = grad_phi_y / norm_grad;
            }

            double phi_val = phi[id];
            // λ = 4φ(1-φ)/W（论文 Eq.(3)）
            double lambda = 4.0 * phi_val * (1.0 - phi_val) / interfaceWidth;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double geq = w[k] * phi_val * (1.0 + cu / cs2);   // 平衡分布（论文 Eq.(24)）
                
                double cn = cx[k] * nx_val + cy[k] * ny_val;
                // 强迫项 G_i（论文 Eq.(25)）
                double term1 = (cx[k]*dphiux_dt + cy[k]*dphiuy_dt) + cs2 * lambda * cn;
                double Gi = w[k] * term1 / cs2 + w[k] * phi_val * div_u;

                // 带强迫项的 LB 碰撞（论文 Eq.(21)）
                g_post[offset(x,y,k)] = g[current][offset(x,y,k)] -
                                (g[current][offset(x,y,k)] - geq) / tau_g +
                                (1.0 - 0.5/tau_g) * dt * Gi;
            }
        }
    }
    // 迁移（streaming）及边界处理（底部无滑移、顶部气相）
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == 0 || sy == Ny-1) {
                    // 壁面反弹边界（无滑移）
                    g[next][offset(x,y,k)] = g_post[offset(x,y,opp[k])];
                } else {
                    g[next][offset(x,y,k)] = g_post[offset(sx,sy,k)];
                }
            }
        }
    }
    // 更新相场 φ = Σ g_i + (Δt/2) φ ∇·u（论文 Eq.(26)）
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double sum_g = 0.0;
            for (int k = 0; k < q; ++k) sum_g += g[next][offset(x,y,k)];
            double div_u = 0.0;
            if (y > 0 && y < Ny-1) {
                int xp = (x+1)%Nx, xm = (x-1+Nx)%Nx;
                div_u = (ux[index(xp,y)] - ux[index(xm,y)])/(2*dx) +
                        (uy[index(x,y+1)] - uy[index(x,y-1)])/(2*dx);
            }
            double phi_new = sum_g / (1.0 - 0.5 * dt * div_u);
            phi[id] = std::clamp(phi_new, 0.0, 1.0);
        }
    }
    // 润湿边界条件（论文 Eq.(39)-(41)）
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        double phi_wall = 0.5 + 0.5 * std::cos(wettingAngle);  // 接触角关系
        phi[id_b] = phi_wall;
        for (int k = 0; k < q; ++k) {
            double cu = cx[k]*ux[id_b] + cy[k]*uy[id_b];
            g[next][offset(x,0,k)] = w[k] * phi_wall * (1.0 + cu / cs2);
        }
    }
    // 顶部边界为气相（φ=0）
    for (int x = 0; x < Nx; ++x) {
        int id_t = index(x, Ny-1);
        phi[id_t] = 0.0;
        for (int k = 0; k < q; ++k) g[next][offset(x,Ny-1,k)] = 0.0;
    }
}

// ==================== 温度场更新（焓法 LB） ====================
// 对应论文 Eq.(27)-(29)
void LBMSolver::update_temperature() {
    std::vector<double> h_post(Nx*Ny*q, 0.0);

    // 1. 设置恒温边界（Dirichlet）的分布函数（底部冷壁 T=0，顶部热壁 T=1）
    for (int x = 0; x < Nx; ++x) {
        // 底部：T_w = 0，完全固态 f_s=1.0
        double T_wall = 0.0, fs_wall = 1.0;
        double H_wall = cp * T_wall + L * (1.0 - fs_wall);
        for (int k = 0; k < q; ++k) {
            h_post[offset(x, 0, k)] = (k == 0) ? H_wall - cp * T_wall + w[k] * cp * T_wall : w[k] * cp * T_wall;
        }
        // 顶部：T = 1，完全液态 f_s=0.0
        double T_top = 1.0, fs_top = 0.0;
        double H_top = cp * T_top + L * (1.0 - fs_top);
        for (int k = 0; k < q; ++k) {
            h_post[offset(x, Ny-1, k)] = (k == 0) ? H_top - cp * T_top + w[k] * cp * T_top : w[k] * cp * T_top;
        }
    }

    // 2. 内部节点的碰撞（LBGK，无额外源项，论文 Eq.(27)）
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double ux_loc = ux[id], uy_loc = uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;
            
            double H_curr = 0.0;
            for (int k = 0; k < q; ++k) H_curr += h[current][offset(x,y,k)];
            
            // 由总焓 H 反求 f_s 和 T（基于焓法，论文 Sec.2.2）
            double fs_curr, T_curr;
            if (H_curr < cp * Tm) {                // 完全固态
                fs_curr = 1.0;
                T_curr = H_curr / cp;
            } else if (H_curr > cp * Tm + L) {      // 完全液态
                fs_curr = 0.0;
                T_curr = (H_curr - L) / cp;
            } else {                                 // 两相区
                fs_curr = 1.0 - (H_curr - cp * Tm) / L;
                T_curr = Tm;
            }
            fs_curr = std::clamp(fs_curr, 0.0, 1.0);
            T_curr = std::clamp(T_curr, 0.0, 1.0);
            
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double heq;
                if (k == 0) {
                    heq = H_curr - cp * T_curr + w[k] * cp * T_curr * (1.0 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                } else {
                    heq = w[k] * cp * T_curr * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                }
                // 碰撞（论文 Eq.(27) 右端第一项）
                h_post[offset(x,y,k)] = h[current][offset(x,y,k)] - (h[current][offset(x,y,k)] - heq) / tau_h;
            }
            fs[id] = fs_curr;
            T[id] = T_curr;
        }
    }
    
    // 3. 迁移（streaming），直接从相邻节点拉取 h_post
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                h[next][offset(x,y,k)] = h_post[offset(sx,sy,k)];
            }
        }
    }
    
    // 4. 更新宏观焓及固相分数、温度
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double H_new = 0.0;
            for (int k = 0; k < q; ++k) H_new += h[next][offset(x,y,k)];
            
            double fs_new, T_new;
            if (H_new < cp * Tm) {
                fs_new = 1.0; T_new = H_new / cp;
            } else if (H_new > cp * Tm + L) {
                fs_new = 0.0; T_new = (H_new - L) / cp;
            } else {
                fs_new = 1.0 - (H_new - cp * Tm) / L; T_new = Tm;
            }
            fs[id] = std::clamp(fs_new, 0.0, 1.0);
            T[id] = std::clamp(T_new, 0.0, 1.0);
        }
    }
    
    // 强制边界条件
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        T[id_b] = 0.0; fs[id_b] = 1.0;
        for (int k = 0; k < q; ++k) h[next][offset(x,0,k)] = h_post[offset(x,0,k)];
        
        int id_t = index(x, Ny-1);
        T[id_t] = 1.0; fs[id_t] = 0.0;
        for (int k = 0; k < q; ++k) h[next][offset(x,Ny-1,k)] = h_post[offset(x,Ny-1,k)];
    }
}

// ==================== 流场更新（带质量源项的 NS 方程） ====================
// 对应论文 Eq.(30)-(36)
void LBMSolver::update_flow_field() {
    std::vector<double> f_post(Nx*Ny*q, 0.0);

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = rho[id];
            double ux_loc = ux[id];
            double uy_loc = uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;

            double F_x = Fs_x[id];
            double F_y = Fs_y[id];
            double S_val = MassSource[id];

            // 计算 ∇p 和 ∇ρ（用于构造 F̃）
            int xp = (x + 1) % Nx;
            int xm = (x - 1 + Nx) % Nx;
            int yp = y + 1;
            int ym = y - 1;
            double grad_p_x = (p[index(xp, y)] - p[index(xm, y)]) / (2.0 * dx);
            double grad_p_y = (p[index(x, yp)] - p[index(x, ym)]) / (2.0 * dx);
            double grad_rho_x = (rho[index(xp, y)] - rho[index(xm, y)]) / (2.0 * dx);
            double grad_rho_y = (rho[index(x, yp)] - rho[index(x, ym)]) / (2.0 * dx);

            // F̃ = F - ∇p + c_s² ∇ρ（论文 Eq.(33) 中的 F̃）
            double F_tilde_x = F_x - grad_p_x + cs2 * grad_rho_x;
            double F_tilde_y = F_y - grad_p_y + cs2 * grad_rho_y;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                // 流场平衡分布 f^{eq}（论文 Eq.(31)）
                double feq = w[k] * rho_loc * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                
                double cF = cx[k] * F_tilde_x + cy[k] * F_tilde_y;
                double uF = ux_loc * F_tilde_x + uy_loc * F_tilde_y;
                // 强迫项 F_i（论文 Eq.(33)）
                double Fi = w[k] * (S_val + cF/cs2 + (cF*cu - cs2*uF)/(cs2*cs2));

                // 带强迫项的 LB 碰撞（论文 Eq.(30)）
                f_post[offset(x,y,k)] = f[current][offset(x,y,k)] - 
                                        (f[current][offset(x,y,k)] - feq) / tau_f + 
                                        dt * (1.0 - 0.5/tau_f) * Fi;
            }
        }
    }

    // 迁移及边界处理（底部和顶部无滑移反弹）
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == 0 || sy == Ny-1) {
                    f[next][offset(x,y,k)] = f_post[offset(x,y,opp[k])];
                } else {
                    f[next][offset(x,y,k)] = f_post[offset(sx,sy,k)];
                }
            }
        }
    }

    // 边界宏观量固定（无滑移）
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x,0);   rho[id_b] = 1.0; ux[id_b]=0.0; uy[id_b]=0.0;
        int id_t = index(x,Ny-1); rho[id_t] = 1.0; ux[id_t]=0.0; uy[id_t]=0.0;
        for (int k = 0; k < q; ++k) {
            f[next][offset(x,0,k)] = w[k] * rho[id_b];
            f[next][offset(x,Ny-1,k)] = w[k] * rho[id_t];
        }
    }
}

// ==================== 主循环：依次更新流场、相场、温度场 ====================
void LBMSolver::collide_and_stream() {
    update_flow_field();      // 流场（Navier-Stokes）
    update_phase_field();     // 相场（Allen-Cahn）
    update_temperature();     // 温度场（焓法）
    std::swap(current, next); // 交换前后缓冲区
    
    compute_macros();         // 重新计算宏观量（用于下一时间步）
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
