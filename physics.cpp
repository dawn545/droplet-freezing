#include "physics.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

// ==================== D2Q9 常数（格子Boltzmann标准参数） ====================
static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double w[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// ==================== 构造函数：初始化所有物理参数和场 ====================
LBMSolver::LBMSolver(int nx, int ny, double gamma_, double Ste_, double Pr_)
    : Nx(nx), Ny(ny), gamma(gamma_), Ste(Ste_), Pr(Pr_),
      dx(1.0), dt(0.1),
      // 密度（无量纲，以液体密度为参考）；rho_g 取 0.1 让 incompressible-pressure LB 数值稳定
      rho_g(0.1), rho_l(1.0), rho_s(gamma_),
        // 比热（无量纲，以液体比热为参考）
      Cp_g(1.0), Cp_l(1.0), Cp_s(0.5),
        // 导热系数（无量纲，以液体导热系数为参考）
      k_g(0.1), k_l(1.0), k_s(3.8),
        // 潜热（无量纲，由Stefan数定义：Ste = Cp_l*(Tm-Tw)/L）
      L(Cp_l * 0.5 / Ste_),  // 假设 Tm=0.5, Tw=0，则 ΔT=0.5
        // 相变温度（无量纲）：论文为 pure-material（水/冰）等温熔化
      Ts(0.5), Tl(0.5), Tm(0.5),
        // 表面张力参数 + 浮力参数
      sigma(0.01), W(2.0), M(0.01),
      g_accel(0.0005), thermal_exp_coeff(0.0002),
      wettingAngle(90.0 * M_PI / 180.0),
        // 数组初始化
      current(0), next(1),
      phi(nx*ny, 0.0), fs(nx*ny, 0.0), fl(nx*ny, 0.0),
      T(nx*ny, 0.0), H(nx*ny, 0.0), p(nx*ny, 0.0),
      rho_mix(nx*ny, 0.0), mu_mix(nx*ny, 0.0), k_mix(nx*ny, 0.0), Cp_mix(nx*ny, 0.0),
      ux(nx*ny, 0.0), uy(nx*ny, 0.0),
      ux_star(nx*ny, 0.0), uy_star(nx*ny, 0.0),
      Fs_x(nx*ny, 0.0), Fs_y(nx*ny, 0.0),
      Gx(nx*ny, 0.0), Gy(nx*ny, 0.0),
      fx(nx*ny, 0.0), fy(nx*ny, 0.0),
      m_dot(nx*ny, 0.0), q_dot(nx*ny, 0.0), S_field(nx*ny, 0.0),
      phi_ux_prev(nx*ny, 0.0), phi_uy_prev(nx*ny, 0.0),
      fs_prev(nx*ny, 0.0),
      lambda(nx*ny, 0.0), nx_field(nx*ny, 0.0), ny_field(nx*ny, 0.0),
      grad_phi_x(nx*ny,0.0),
      grad_phi_y(nx*ny,0.0),
      lap_phi(nx*ny,0.0),
      ux_solid(nx*ny, 0.0), uy_solid(nx*ny, 0.0),
      Cp_ref(Cp_l)
{
    int total = nx * ny * q;
    f[0].assign(total, 0.0); f[1].assign(total, 0.0);
    g[0].assign(total, 0.0); g[1].assign(total, 0.0);
    h[0].assign(total, 0.0); h[1].assign(total, 0.0);

    double c = dx / dt;
    cs2 = c * c / 3.0;

    // 相场弛豫时间（由迁移率 M 决定：M = cs2*(tau_g-0.5)*dt，论文 Eq.21 之后）
    tau_g = M / (cs2 * dt) + 0.5;

    // 流场和温度场弛豫时间在 update_flow_field / update_temperature 内由 mu_mix、k_mix 逐点计算

    // 表面张力参数（式13）
    beta = 12.0 * sigma / W;
    kappa = 1.5 * sigma * W;

    // 参考比热（用于温度场平衡分布中的 Cp_ref）
    Cp_ref = Cp_l;

    // 初始化固体速度为零（无滑移）
    std::fill(ux_solid.begin(), ux_solid.end(), 0.0);
    std::fill(uy_solid.begin(), uy_solid.end(), 0.0);
}

// ==================== 更新依赖于相态的混合物理属性 ====================
void LBMSolver::update_mixture_properties() {
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double phi_val = phi[id];
            double fs_val = fs[id];
            double fl_val = 1.0 - fs_val;

            // 密度：式(1)   ζ = fs*ζ_s + (1-fs)*ϕ*ζ_l + (1-fs)*(1-ϕ)*ζ_g
            rho_mix[id] = fs_val * rho_s + fl_val * phi_val * rho_l + fl_val * (1.0 - phi_val) * rho_g;
            // 比热
            Cp_mix[id] = fs_val * Cp_s + fl_val * phi_val * Cp_l + fl_val * (1.0 - phi_val) * Cp_g;
            // 导热系数
            k_mix[id] = fs_val * k_s + fl_val * phi_val * k_l + fl_val * (1.0 - phi_val) * k_g;
            // 动力粘度：液相由 Pr 决定；论文未给固相/气相 μ
            //  - 固相：取 μ_s = μ_l，因为速度由 immersed-boundary 控制 (论文 Eq.20 后说明)
            //  - 气相：与液相同 Pr 下 μ_g = Pr·k_g/Cp_g = 0.1·μ_l
            double mu_l = Pr * k_l / Cp_l;
            double mu_g = Pr * k_g / Cp_g;
            double mu_s = mu_l;
            mu_mix[id] = fs_val * mu_s + fl_val * phi_val * mu_l + fl_val * (1.0 - phi_val) * mu_g;
        }
    }
}

// ==================== 计算 Boussinesq 浮力项 （论文 Section 4.3） ====================
void LBMSolver::compute_boussinesq_buoyancy() {
    double T_ref = 0.5;  // 参考温度
    double rho_ref = rho_l;  // 参考密度
    
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double phi_val = phi[id];
            double T_val = T[id];
            double fs_val = fs[id];
            
            // 只在液体区域应用浮力（φ > 0.5 且不是纯固体）
            if (phi_val > 0.5 && fs_val < 0.9) {
                // 【论文 Section 4.3】Boussinesq 浮力模型
                // Fg = -ρ₀·g·β·(T - Tref)·ĵ
                // 其中 ρ₀ = rho_l, β 是体积膨胀系数, g 是重力加速度
                double buoyancy = -rho_ref * g_accel * thermal_exp_coeff * (T_val - T_ref);
                Gx[id] = 0.0;
                Gy[id] = buoyancy;  // 向上（y 方向正向）
            } else {
                Gx[id] = 0.0;
                Gy[id] = 0.0;
            }
        }
    }
}

// ==================== 计算流固耦合力（扩散界面法） ====================
void LBMSolver::compute_fluid_solid_interaction() {
    // paper.md inline (Eq.20 下方): f = f_s · (u_s - u*) / Δt （加速度）
    // 代码中的 fx[i] 表示 ρf（force per volume），用于 Eq.(33) 的 ci·(F+ρf)/cs² 项
    // 故 fx[i] = ρ · f_s · (u_s - u*) / Δt
    for (int i = 0; i < Nx*Ny; ++i) {
        double fs_loc = fs[i];
        double rho_loc = rho_mix[i];
        double factor = 2 * rho_loc * fs_loc / dt;
        fx[i] = factor * (ux_solid[i] - ux_star[i]);
        fy[i] = factor * (uy_solid[i] - uy_star[i]);
    }
}

// ==================== 计算 λ 和界面法向量 ====================
void LBMSolver::compute_lambda_and_normal()
{
    const double eps = 1e-12;

    for (int y=0;y<Ny;++y)
    {
        for (int x=0;x<Nx;++x)
        {
            int id=index(x,y);
            double gx=0.0;
            double gy=0.0;
            double lap=0.0;
            double phi0=phi[id];
            for (int k=1;k<q;++k)
            {
                int xn=(x+cx[k]+Nx)%Nx;
                int yn=y+cy[k];

                yn=std::max(0,std::min(Ny-1,yn));

                int nid=index(xn,yn);

                double phin=phi[nid];

                //---------------- Eq.(38a)
                gx+=w[k]*cx[k]*phin;
                gy+=w[k]*cy[k]*phin;

                //---------------- Eq.(38b)
                lap+=2.0*w[k]*(phin-phi0);
            }
            gx/=cs2*dt;
            gy/=cs2*dt;
            lap/=(cs2*dt*dt);
            grad_phi_x[id]=gx;
            grad_phi_y[id]=gy;
            lap_phi[id]=lap;
            double norm=sqrt(gx*gx+gy*gy)+eps;
            nx_field[id]=gx/norm;
            ny_field[id]=gy/norm;

            lambda[id]=4.0*phi0*(1.0-phi0)/W;
        }
    }
}

// ==================== 计算化学势 μ_φ 和表面张力 F_s ====================
void LBMSolver::compute_phase_derivatives() {
    Eigen::Map<Array2D> phi_map(phi.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);

    Array2D lap_phi = Array2D::Zero(Ny, Nx);
    Array2D grad_x = Array2D::Zero(Ny, Nx);
    Array2D grad_y = Array2D::Zero(Ny, Nx);

    // 中心差分计算梯度和拉普拉斯
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 1; x < Nx-1; ++x) {
            double phi_c = phi_map(y, x);
            double phi_l = phi_map(y, x-1);
            double phi_r = phi_map(y, x+1);
            double phi_d = phi_map(y-1, x);
            double phi_u = phi_map(y+1, x);
            grad_x(y, x) = (phi_r - phi_l) / (2.0 * dx);
            grad_y(y, x) = (phi_u - phi_d) / (2.0 * dx);
            lap_phi(y, x) = (phi_l + phi_r + phi_d + phi_u - 4.0*phi_c) / (dx*dx);
        }
    }
    
    // 在函数开头引入液相分数的 Eigen Map
    Eigen::Map<Array2D> fl_map(fl.data(), Ny, Nx);
    
    // ... 前面的梯度与 mu 计算保持不变 ...
    Array2D mu = 4.0 * beta * phi_map * (phi_map - 1.0) * (phi_map - 0.5) - kappa * lap_phi;

    // 【修正】使用液相分数截断表面张力，防止冰层表面产生伪速度
    Fsx_map = fl_map * mu * grad_x;
    Fsy_map = fl_map * mu * grad_y;
}

// ==================== 从分布函数计算宏观量 ====================
void LBMSolver::compute_macros() {
    compute_phase_derivatives();
    update_mixture_properties();

    Eigen::Map<Array2D> rho_map(rho_mix.data(), Ny, Nx);
    Eigen::Map<Array2D> p_map(p.data(), Ny, Nx);
    Eigen::Map<Array2D> ux_map(ux.data(), Ny, Nx);
    Eigen::Map<Array2D> uy_map(uy.data(), Ny, Nx);
    Eigen::Map<Array2D> fs_map(fs.data(), Ny, Nx);
    Eigen::Map<Array2D> fsp_map(fs_prev.data(), Ny, Nx);
    Eigen::Map<Array2D> src_map(m_dot.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);
    Eigen::Map<Array2D> ux_star_map(ux_star.data(), Ny, Nx);
    Eigen::Map<Array2D> uy_star_map(uy_star.data(), Ny, Nx);

    // 质量源项：ṁ = (1 - ρ_s/ρ_l) ∂f_s/∂t (式19)
    Array2D m_dot_arr = (1.0 - rho_s/rho_l) * (fs_map - fsp_map) / dt;
    src_map = rho_map * m_dot_arr;   // S = ρ ṁ
    fsp_map = fs_map;

    // 计算分布函数的零阶矩和一阶矩
    Array2D rho_temp = Array2D::Zero(Ny, Nx);
    Array2D mom_x = Array2D::Zero(Ny, Nx);
    Array2D mom_y = Array2D::Zero(Ny, Nx);
    Array2D sum_f_neq_0 = Array2D::Zero(Ny, Nx);

    for (int k = 0; k < q; ++k) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                double f_val = f[current][offset(x,y,k)];
                rho_temp(y, x) += f_val;
                mom_x(y, x) += f_val * cx[k];
                mom_y(y, x) += f_val * cy[k];
                if (k != 0) sum_f_neq_0(y, x) += f_val;
            }
        }
    }

    // 密度由物性公式（论文式1，update_mixture_properties 已算）控制，不从 Σf 反推
    // 该 LB 形式（Yuan 2020）中 Σf ≠ ρ，强制覆盖会让 ρ 跌至 0 → 数值崩溃
    rho_map = rho_map.max(1e-12);
    // 论文 Eq.34: u* = Σci f_i + (Δt/2)·F，F = Fs + G
    Eigen::Map<Array2D> Gx_map(Gx.data(), Ny, Nx);
    Eigen::Map<Array2D> Gy_map(Gy.data(), Ny, Nx);
    ux_star_map = (mom_x + 0.5 * dt * (Fsx_map + Gx_map)) / rho_map;
    uy_star_map = (mom_y + 0.5 * dt * (Fsy_map + Gy_map)) / rho_map;

    // 计算流固耦合力
    compute_fluid_solid_interaction();

    // 修正速度 (式35)：u = u* + 0.5·dt·f/ρ（f 为 force/volume）
    ux_map = ux_star_map + 0.5 * dt * Eigen::Map<Array2D>(fx.data(), Ny, Nx) / rho_map;
    uy_map = uy_star_map + 0.5 * dt * Eigen::Map<Array2D>(fy.data(), Ny, Nx) / rho_map;

    // 计算压力 (式36)
    // 先计算梯度
    // 计算压力 (式36) 和 完整的源项 S
    Array2D grad_p_x = Array2D::Zero(Ny, Nx);
    Array2D grad_p_y = Array2D::Zero(Ny, Nx);
    Array2D grad_rho_x = Array2D::Zero(Ny, Nx);
    Array2D grad_rho_y = Array2D::Zero(Ny, Nx);
    
    // 1. 先计算 p 和 rho 的梯度
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int xp = (x+1) % Nx;
            int xm = (x-1+Nx) % Nx;
            grad_p_x(y, x) = (p[index(xp,y)] - p[index(xm,y)]) / (2.0*dx);
            grad_p_y(y, x) = (p[index(x,y+1)] - p[index(x,y-1)]) / (2.0*dx);
            grad_rho_x(y, x) = (rho_map(y, xp) - rho_map(y, xm)) / (2.0*dx);
            grad_rho_y(y, x) = (rho_map(y+1, x) - rho_map(y-1, x)) / (2.0*dx);
        }
    }

    // 2. 计算完整源项 S = ρṁ + u·∇ρ
    Array2D S_map = Array2D::Zero(Ny, Nx);
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            S_map(y, x) = rho_map(y, x) * m_dot_arr(y, x) + 
                          ux_map(y, x) * grad_rho_x(y, x) + 
                          uy_map(y, x) * grad_rho_y(y, x);
            S_field[index(x, y)] = S_map(y, x); // 保存到成员变量供 update_flow_field 使用
        }
    }

    // 3. 计算 S 的梯度 ∇S
    Array2D grad_S_x = Array2D::Zero(Ny, Nx);
    Array2D grad_S_y = Array2D::Zero(Ny, Nx);
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int xp = (x+1) % Nx;
            int xm = (x-1+Nx) % Nx;
            grad_S_x(y, x) = (S_map(y, xp) - S_map(y, xm)) / (2.0*dx);
            grad_S_y(y, x) = (S_map(y+1, x) - S_map(y-1, x)) / (2.0*dx);
        }
    }

    // 在 3. 计算 S 的梯度 ∇S 循环结束后，补充以下边界梯度处理：
    for (int x = 0; x < Nx; ++x) {
        // 底部 y = 0 单边差分
        grad_p_y(0, x) = (p_map(1, x) - p_map(0, x)) / dx;
        grad_rho_y(0, x) = (rho_map(1, x) - rho_map(0, x)) / dx;
        grad_S_y(0, x) = (S_map(1, x) - S_map(0, x)) / dx;
        
        // 顶部 y = Ny-1 单边差分
        grad_p_y(Ny-1, x) = (p_map(Ny-1, x) - p_map(Ny-2, x)) / dx;
        grad_rho_y(Ny-1, x) = (rho_map(Ny-1, x) - rho_map(Ny-2, x)) / dx;
        grad_S_y(Ny-1, x) = (S_map(Ny-1, x) - S_map(Ny-2, x)) / dx;
    }

    // 4. 计算最终压力 p (论文 Eq.36)
    double w0 = w[0];
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double ux_loc = ux_map(y, x);
            double uy_loc = uy_map(y, x);
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;

            // F̃ = F - ∇p + cs²∇ρ + cs²∇S，F = Fs + G（不含 f）
            double F_tilde_x = Fsx_map(y, x) + Gx_map(y, x) - grad_p_x(y, x) + cs2 * grad_rho_x(y, x) + cs2 * grad_S_x(y, x);
            double F_tilde_y = Fsy_map(y, x) + Gy_map(y, x) - grad_p_y(y, x) + cs2 * grad_rho_y(y, x) + cs2 * grad_S_y(y, x);

            double S_val = S_map(y, x);
            double F0 = w0 * (S_val - (ux_loc*F_tilde_x + uy_loc*F_tilde_y)/cs2);
            double rho_s0 = rho_map(y, x) * w[0] * (- u2 / (2.0 * cs2));
            // paper.md inline τ_f = ν/(cs²·Δt) + 0.5，ν = mu_mix/ρ
            int id_p = index(x, y);
            double tau_loc = mu_mix[id_p] / (rho_map(y,x) * cs2 * dt) + 0.5;
            tau_loc = std::max(tau_loc, 0.51);
            // paper.md Eq.(36): p = (cs²/(1-ω_0))·[Σ_{i≠0}f_i + (Δt/2)·S + τ·Δt·F_0 + ρ·s_0(u)]
            p_map(y, x) = (cs2 / (1.0 - w0)) * (sum_f_neq_0(y, x) + 0.5*dt*S_val + tau_loc*dt*F0 + rho_s0);
        }
    }
    // 边界压力外推
    for (int x = 0; x < Nx; ++x) {
        p_map(0, x) = p_map(1, x);
        p_map(Ny-1, x) = p_map(Ny-2, x);
    }
}

// ==================== 平衡分布函数计算 ====================
void LBMSolver::compute_feq(double rho, double p_val, double ux, double uy, std::array<double, q>& feq) const {
    double u2 = ux*ux + uy*uy;
    for (int k = 0; k < q; ++k) {
        double cu = cx[k]*ux + cy[k]*uy;
        
        // 计算 s_i(u) (论文式32)
        double s_i = w[k] * (cu/cs2 + (cu*cu)/(2.0*cs2*cs2) - u2/(2.0*cs2));
        
        // 计算 f_i^eq (论文式31)
        if (k == 0) {
            feq[k] = (p_val / cs2) * (w[k] - 1.0) + rho * s_i;
        } else {
            feq[k] = (p_val / cs2) * w[k] + rho * s_i;
        }
    }
}

void LBMSolver::compute_geq(double phi, double ux, double uy, std::array<double, q>& geq) const {
    for (int k = 0; k < q; ++k) {
        double cu = cx[k]*ux + cy[k]*uy;
        geq[k] = w[k] * phi * (1.0 + cu/cs2);
    }
}

void LBMSolver::compute_heq(double H, double T, double Cp_ref, double Cp_loc, double ux, double uy, std::array<double, q>& heq) const {
    double u2 = ux*ux + uy*uy;
    for (int k = 0; k < q; ++k) {
        double cu = cx[k]*ux + cy[k]*uy;
        if (k == 0) {
            // 包含 Cp_ref / Cp_loc 的完整公式
            heq[k] = H - Cp_ref * T + w[k] * Cp_loc * T * (Cp_ref / Cp_loc - u2/(2.0*cs2));
        } else {
            heq[k] = w[k] * Cp_loc * T * (Cp_ref / Cp_loc + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
        }
    }
}

// ==================== 相场更新（Allen-Cahn 方程） ====================
void LBMSolver::update_phase_field() {
    // 计算浮力（每次相场更新前）
    compute_boussinesq_buoyancy();
    
    std::vector<double> g_post(Nx*Ny*q, 0.0);
    compute_lambda_and_normal();

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double phi_val = phi[id];
            double ux_loc = ux[id], uy_loc = uy[id];
            double lambda_val = lambda[id];
            double nx_val = nx_field[id], ny_val = ny_field[id];

            // 计算 ∂(φu)/∂t 和 ∇·u
            int xp = (x+1)%Nx, xm = (x-1+Nx)%Nx;
            int yp = y+1, ym = y-1;
            double dux_dx = (ux[index(xp,y)] - ux[index(xm,y)])/(2.0*dx);
            double duy_dy = (uy[index(x,yp)] - uy[index(x,ym)])/(2.0*dx);
            double div_u = dux_dx + duy_dy;

            double phi_ux = phi_val * ux_loc;
            double phi_uy = phi_val * uy_loc;
            double dphiux_dt = (phi_ux - phi_ux_prev[id]) / dt;
            double dphiuy_dt = (phi_uy - phi_uy_prev[id]) / dt;
            phi_ux_prev[id] = phi_ux;
            phi_uy_prev[id] = phi_uy;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                // 平衡分布 (式24)
                double geq = w[k] * phi_val * (1.0 + cu/cs2);
                // 强迫项 (式25)
                double cn = cx[k]*nx_val + cy[k]*ny_val;
                double term1 = (cx[k]*dphiux_dt + cy[k]*dphiuy_dt) + cs2 * lambda_val * cn;
                double Gi = w[k] * term1 / cs2 + w[k] * phi_val * div_u;
                // 碰撞 (式21)
                g_post[offset(x,y,k)] = g[current][offset(x,y,k)] -
                                        (g[current][offset(x,y,k)] - geq) / tau_g +
                                        (1.0 - 0.5/tau_g) * dt * Gi;
            }
        }
    }

    // 迁移（streaming）+ 边界处理
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == 0 || sy == Ny-1) {
                    // 反弹边界
                    g[next][offset(x,y,k)] = g_post[offset(x,y,opp[k])];
                } else {
                    g[next][offset(x,y,k)] = g_post[offset(sx,sy,k)];
                }
            }
        }
    }

    // 更新相场 φ (式26)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double sum_g = 0.0;
            for (int k = 0; k < q; ++k) sum_g += g[next][offset(x,y,k)];
            double div_u = 0.0;
            if (y > 0 && y < Ny-1) {
                int xp = (x+1)%Nx, xm = (x-1+Nx)%Nx;
                div_u = m_dot[id];
            }
            double phi_new = sum_g / (1.0 - 0.5 * dt * div_u);
            phi[id] = std::clamp(phi_new, 0.0, 1.0);
        }
    }

    // 润湿边界条件（底部）
    apply_wetting_bc();
    // 顶部气相边界
    for (int x = 0; x < Nx; ++x) {
        int id_t = index(x, Ny-1);
        phi[id_t] = 0.0;
        for (int k = 0; k < q; ++k) g[next][offset(x,Ny-1,k)] = 0.0;
    }
}


// ==================== 温度场更新（焓法 LB） ====================
void LBMSolver::update_temperature() {
    std::vector<double> h_post(Nx*Ny*q, 0.0);

    // 计算浮力
    compute_boussinesq_buoyancy();
    
    // 内部节点碰撞
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double ux_loc = ux[id], uy_loc = uy[id];
            
            double H_curr = H[id];

            double phi_v = phi[id];
            
            // 【修正】固相线对应的等效显热容也必须包含气相的贡献！
            double Cp_sensible_s = phi_v * Cp_s + (1.0 - phi_v) * Cp_g;
            double Cp_sensible_l = phi_v * Cp_l + (1.0 - phi_v) * Cp_g;
            
            const double Hs_val = Cp_sensible_s * Ts;
            const double Hl_val = Cp_sensible_l * Tl + phi_v * L;

            double fs_curr, T_curr, Cp_loc;

            // 【新增保护】如果处于纯气相或无潜热区域，直接按照显热计算，跳过相变判定
            if (Hl_val - Hs_val < 1e-6) {
                fs_curr = 0.0;
                T_curr = H_curr / std::max(Cp_sensible_l, 1e-6);
            } else if (H_curr <= Hs_val) {
                fs_curr = 1.0;
                T_curr = H_curr / std::max(Cp_sensible_s, 1e-6);
            } else if (H_curr >= Hl_val) {
                fs_curr = 0.0;
                T_curr = (H_curr - phi_v * L) / std::max(Cp_sensible_l, 1e-6);
            } else {
                double fl_local = (H_curr - Hs_val) / (Hl_val - Hs_val);
                fs_curr = 1.0 - fl_local;
                T_curr = Ts + fl_local * (Tl - Ts); 
            }

            fs_curr = std::clamp(fs_curr, 0.0, 1.0);
            T_curr = std::clamp(T_curr, 0.0, 1.0);

            // 更新当前节点的真实比热容用于平衡态计算
            double fl_curr = 1.0 - fs_curr;
            Cp_loc = fs_curr * Cp_s + fl_curr * phi_v * Cp_l + fl_curr * (1.0 - phi_v) * Cp_g;

            std::array<double, q> heq;
            compute_heq(H_curr, T_curr, Cp_ref, Cp_loc, ux_loc, uy_loc, heq);

            double tau_h_local = k_mix[id] / (Cp_ref * cs2 * dt) + 0.5;
            tau_h_local = std::max(tau_h_local, 0.51);

            for (int k = 0; k < q; ++k) {
                h_post[offset(x,y,k)] = h[current][offset(x,y,k)] - (h[current][offset(x,y,k)] - heq[k]) / tau_h_local;
            }
        }
    }

    // 迁移（streaming）：引入局部边界判断
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == Ny-1) {
                    h[next][offset(x,y,k)] = h_post[offset(x,y,opp[k])];
                } else if (sy == 0) {
                    // ======= COMSOL 边界核心：判断当前位置是否为液滴 =======
                    double phi_bottom = phi[index(sx, 0)];
                    if (phi_bottom > 0.5) {
                        // 液滴接触面：冷板 (T=0)
                        h[next][offset(x,y,k)] = -h_post[offset(x,y,opp[k])] + 2.0 * w[k] * Cp_ref * 0.0;
                    } else {
                        // 空气接触面：绝热 (半步反弹近似)
                        h[next][offset(x,y,k)] = h_post[offset(x,y,opp[k])];
                    }
                } else {
                    h[next][offset(x,y,k)] = h_post[offset(sx,sy,k)];
                }
            }
        }
    }

    // 更新宏观量 - 同时计算潜热源项
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double H_new = 0.0;
            for (int k = 0; k < q; ++k) H_new += h[next][offset(x,y,k)];
            const double Hs_val = Cp_s * Ts;
            const double Hl_val = Cp_l * Tl + L;
            double phi_v = phi[id];
            double fs_new, T_new;
            if (phi_v < 0.5) {
                fs_new = 0.0;
                double Cp_loc_eff = phi_v * Cp_l + (1.0 - phi_v) * Cp_g;
                T_new = H_new / std::max(Cp_loc_eff, 1e-6);
            } else if (H_new <= Hs_val) {
                fs_new = 1.0; T_new = H_new / Cp_s;
            } else if (H_new >= Hl_val) {
                fs_new = 0.0; T_new = (H_new - L) / Cp_l;
            } else {
                double fl_local = (H_new - Hs_val) / (Hl_val - Hs_val);
                fs_new = 1.0 - fl_local;
                T_new = Ts + fl_local * (Tl - Ts);
            }
            
            // 【论文式(6)】计算潜热源项：q̇ = -∂(ρLf_l)/∂t = ρL·∂fs/∂t
            // 使用当前时刻的固相分数变化率
            double fs_old = fs[id];
            double rho_ref = rho_mix[id];
            q_dot[id] = rho_ref * L * (fs_new - fs_old) / dt;
            
            H[id] = H_new;
            fs[id] = std::clamp(fs_new, 0.0, 1.0);
            T[id] = std::clamp(T_new, 0.0, 1.0);
        }
    }

    // 边界宏观量
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        int id_b_up = index(x, 1);
        double phi_b = phi[id_b];
        
        // ======= 底部物理宏观量强迫 =======
        if (phi_b > 0.5) {
            T[id_b] = 0.0;    // 液滴接触面强迫为冷源
            fs[id_b] = 1.0;   // 强迫为纯冰
            H[id_b] = 0.0; 
        } else {
            T[id_b] = T[id_b_up]; // 气体区域绝热，温度等于上一层
            fs[id_b] = 0.0;
            H[id_b] = Cp_g * T[id_b];
        }

        // 顶部绝热：拷贝最近内部节点
        int id_t = index(x, Ny-1);
        int id_t_in = index(x, Ny-2);
        T[id_t] = T[id_t_in];
        fs[id_t] = fs[id_t_in];
        H[id_t] = H[id_t_in];
    }
}

// ==================== 流场更新（带质量源项的 NS 方程） ====================
void LBMSolver::update_flow_field() {
    std::vector<double> f_post(Nx*Ny*q, 0.0);

    // 先更新混合物性，以便获取粘度等（但流场 LB 不需要直接使用 mu_mix，这里仅作示例）
    update_mixture_properties();

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = rho_mix[id];
            double ux_loc = ux[id], uy_loc = uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;

            // 论文 Eq.33: F = Fs + G (不含 f)；二项用 (F+f)，三项用 F̃
            double F_x = Fs_x[id] + Gx[id];
            double F_y = Fs_y[id] + Gy[id];
            double Ff_x = F_x + fx[id];   // (F+f) 用于第二项 ci·(F+f)/cs²
            double Ff_y = F_y + fy[id];

            // 本地弛豫时间 τ_f = ν/(cs²·dt) + 0.5，ν = mu_mix/ρ (论文 Eq.30 后说明)
            double tau_loc = mu_mix[id] / (rho_loc * cs2 * dt) + 0.5;
            tau_loc = std::max(tau_loc, 0.51);

            // 计算 ∇p, ∇ρ 和 ∇S
            int xp = (x+1)%Nx, xm = (x-1+Nx)%Nx;
            int yp = y+1, ym = y-1;
            double grad_p_x = (p[index(xp,y)] - p[index(xm,y)]) / (2.0*dx);
            double grad_p_y = (p[index(x,yp)] - p[index(x,ym)]) / (2.0*dx);
            double grad_rho_x = (rho_mix[index(xp,y)] - rho_mix[index(xm,y)]) / (2.0*dx);
            double grad_rho_y = (rho_mix[index(x,yp)] - rho_mix[index(x,ym)]) / (2.0*dx);
            double grad_S_x = (S_field[index(xp,y)] - S_field[index(xm,y)]) / (2.0*dx);
            double grad_S_y = (S_field[index(x,yp)] - S_field[index(x,ym)]) / (2.0*dx);

            // F̃ = F - ∇p + cs²∇ρ + cs²∇S，F = Fs+G（不含 f）
            double F_tilde_x = F_x - grad_p_x + cs2 * grad_rho_x + cs2 * grad_S_x;
            double F_tilde_y = F_y - grad_p_y + cs2 * grad_rho_y + cs2 * grad_S_y;

            double S_val = S_field[id];

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;

                // 平衡分布 (论文式31, 32)
                double s_i = w[k] * (cu/cs2 + (cu*cu)/(2.0*cs2*cs2) - u2/(2.0*cs2));
                double feq;
                if (k == 0) {
                    feq = (p[id] / cs2) * (w[k] - 1.0) + rho_loc * s_i;
                } else {
                    feq = (p[id] / cs2) * w[k] + rho_loc * s_i;
                }

                // 强迫项 (式33)：第二项 ci·(F+f)/cs²；第三项用 F̃
                double cFf = cx[k]*Ff_x + cy[k]*Ff_y;          // (F+f)·c
                double cFt = cx[k]*F_tilde_x + cy[k]*F_tilde_y; // F̃·c
                double uFt = ux_loc*F_tilde_x + uy_loc*F_tilde_y; // u·F̃
                double Fi = w[k] * (S_val + cFf/cs2 + (cFt*cu - cs2*uFt)/(cs2*cs2));

                // 碰撞 (式30)
                f_post[offset(x,y,k)] = f[current][offset(x,y,k)] -
                                        (f[current][offset(x,y,k)] - feq) / tau_loc +
                                        dt * (1.0 - 0.5/tau_loc) * Fi;
            }
        }
    }

    // 迁移及边界处理
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                
                // 【修改此处】仅底部执行半步反弹，顶部允许读取源数据
                if (sy == 0) {
                    f[next][offset(x,y,k)] = f_post[offset(x,y,opp[k])];
                } else if (sy == Ny - 1) {
                    // 对于来自顶部的分布函数，直接读取当前态（稍后会被零梯度外推覆盖）
                    f[next][offset(x,y,k)] = f[current][offset(sx,sy,k)];
                } else {
                    f[next][offset(x,y,k)] = f_post[offset(sx,sy,k)];
                }
            }
        }
    }

    // 边界宏观量固定（无滑移）；分布函数用 Yuan-LB 平衡态形式（不能用经典 w·ρ）
    // 在 update_flow_field() 内部，将顶部边界的逻辑替换为以下内容：
    for (int x = 0; x < Nx; ++x) {
      // 底部边界（无滑移壁面）
      int id_b = index(x,0);   
      ux[id_b] = 0.0; uy[id_b] = 0.0;
      std::array<double, q> feq_b;
      compute_feq(rho_mix[id_b], p[id_b], 0.0, 0.0, feq_b);
      for (int k = 0; k < q; ++k) {
        f[next][offset(x,0,k)] = feq_b[k];
      }
    
    // 顶部边界（开放出流 / 零梯度）
      int id_t = index(x, Ny-1);
      int id_t_in = index(x, Ny-2);
      ux[id_t] = ux[id_t_in];
      uy[id_t] = uy[id_t_in];
      p[id_t] = p[id_t_in]; 
      for (int k = 0; k < q; ++k) {
          f[next][offset(x, Ny-1, k)] = f[next][offset(x, Ny-2, k)];
      }
    }
}


// ==================== 润湿边界条件（底部） ====================
void LBMSolver::apply_wetting_bc() {
    // 依据论文 Section 3.4 式 (39)-(41) 实现
    double theta = wettingAngle;
    
    // 计算常数系数 -tan(pi/2 - theta)
    double tan_term = -std::tan(M_PI / 2.0 - theta);
    
    for (int x = 0; x < Nx; ++x) {
        // 处理周期性边界的左右索引
        int xp = (x + 1) % Nx;
        int xm = (x - 1 + Nx) % Nx;
        
        // 1. 计算 y=1 和 y=2 层的 x 方向偏导数 (式 41)
        double dphi_dx_1 = (phi[index(xp, 1)] - phi[index(xm, 1)]) / (2.0 * dx);
        double dphi_dx_2 = (phi[index(xp, 2)] - phi[index(xm, 2)]) / (2.0 * dx);
        
        // 2. 计算切向分量 n_tau * grad(phi) (式 40b)
        double n_tau_grad_phi = 1.5 * dphi_dx_1 - 0.5 * dphi_dx_2;
        
        // 3. 计算法向分量 n_w * grad(phi) (式 39)
        double n_w_grad_phi = tan_term * std::abs(n_tau_grad_phi);
        
        // 4. 计算幽灵层 (y=0) 的 phi 值 (由式 40a 变形: phi_{x,0} = phi_{x,1} - dx * n_w_grad_phi)
        double phi_ghost = phi[index(x, 1)] - dx * n_w_grad_phi;
        
        // 限制在物理范围 [0, 1] 内部
        phi[index(x, 0)] = std::clamp(phi_ghost, 0.0, 1.0);
        
        // 同时更新底部的相场分布函数（使用平衡态近似）
        for (int k = 0; k < q; ++k) {
            double cu = cx[k]*ux[index(x,0)] + cy[k]*uy[index(x,0)];
            g[next][offset(x,0,k)] = w[k] * phi[index(x,0)] * (1.0 + cu/cs2);
        }
    }
}

// ==================== 速度边界（无滑移） ====================
void LBMSolver::apply_velocity_bc() {
    for (int x = 0; x < Nx; ++x) {
        ux[index(x,0)] = 0.0; uy[index(x,0)] = 0.0;
    }
}

// ==================== 温度边界（恒温） ====================
// ==================== 温度边界（恒温与绝热混合） ====================
void LBMSolver::apply_temperature_bc() {
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        int id_b_up = index(x, 1);
        
        if (phi[id_b] > 0.5) {
            T[id_b] = 0.0;
            fs[id_b] = 1.0;
            H[id_b] = 0.0; // Cp_s * T (T=0)
        } else {
            T[id_b] = T[id_b_up]; 
            fs[id_b] = 0.0;
            H[id_b] = Cp_g * T[id_b];
        }
        
        int id_t = index(x, Ny-1);
        int id_t_in = index(x, Ny-2);
        T[id_t] = T[id_t_in];
        fs[id_t] = fs[id_t_in];
        H[id_t] = H[id_t_in];
    }
}

// ==================== 总边界条件调度 ====================
void LBMSolver::apply_boundary_conditions() {
    apply_wetting_bc();
    apply_velocity_bc();
    apply_temperature_bc();
}


// ==================== 初始化所有场 ====================
void LBMSolver::initialize_fields() {
    // Sessile droplet：半圆形液滴坐在底部冷板上（centerY=0，半径 R≈Ny·0.35）
    double centerX = Nx * 0.5;
    double centerY = 0.0;
    double R = std::min(Nx, Ny) * 0.35;

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double dxl = x - centerX;
            double dyl = y - centerY;
            double r = std::sqrt(dxl*dxl + dyl*dyl);
            
            // 相场初始化：论文式(45) tanh 平滑界面
            phi[id] = 0.5 + 0.5 * std::tanh(2.0 * (R - r) / W);
            phi[id] = std::clamp(phi[id], 0.0, 1.0);

            // ================= 修正初始化：完全液态液滴，无预冻结 =================
            // 根据论文 Figure 4-6，液滴应初始为完全液态（fs=0）
            if (phi[id] > 0.5) {
                // 1. 液滴内部：完全液态初始化
                fs[id] = 0.0;       // 【修正】液体不是固体
                T[id] = 0.55;       // 初始温度略高于熔点
            } else {
                // 2. 气相环境：没有冰，初始温度与液滴相同
                fs[id] = 0.0;
                T[id] = 0.55;       // 气相同温度便于热传导
            }
            // =========================================================

            fs_prev[id] = fs[id];
            fl[id] = 1.0 - fs[id];
            
            // 初始总焓：液-固混合区按式(8)，气相只有 sensible heat
            double phi_v = phi[id];
            double H_liq = Cp_l * T[id] + L * (1.0 - fs[id]);
            double H_gas = Cp_g * T[id];
            H[id] = phi_v * H_liq + (1.0 - phi_v) * H_gas;
            
            ux[id] = uy[id] = 0.0;
            ux_star[id] = uy_star[id] = 0.0;
            
            // 混合物性初值与本地 phi/fs 一致
            double fs_v = fs[id], fl_v = 1.0 - fs_v;
            double mu_l_init = Pr * k_l / Cp_l;
            double mu_g_init = Pr * k_g / Cp_g;
            double mu_s_init = mu_l_init;
            rho_mix[id] = fs_v*rho_s + fl_v*phi_v*rho_l + fl_v*(1.0-phi_v)*rho_g;
            mu_mix[id]  = fs_v*mu_s_init + fl_v*phi_v*mu_l_init + fl_v*(1.0-phi_v)*mu_g_init;
            k_mix[id]   = fs_v*k_s + fl_v*phi_v*k_l + fl_v*(1.0-phi_v)*k_g;
            Cp_mix[id]  = fs_v*Cp_s + fl_v*phi_v*Cp_l + fl_v*(1.0-phi_v)*Cp_g;

            // 初始化分布函数为平衡态
            std::array<double, q> feq, geq, heq;
            compute_feq(rho_mix[id], p[id], ux[id], uy[id], feq);
            compute_geq(phi[id], ux[id], uy[id], geq);
            compute_heq(H[id], T[id], Cp_ref, Cp_mix[id], ux[id], uy[id], heq);
            for (int k = 0; k < q; ++k) {
                f[0][offset(x,y,k)] = feq[k];
                f[1][offset(x,y,k)] = feq[k];
                g[0][offset(x,y,k)] = geq[k];
                g[1][offset(x,y,k)] = geq[k];
                h[0][offset(x,y,k)] = heq[k];
                h[1][offset(x,y,k)] = heq[k];
            }
        }
    }
    std::fill(phi_ux_prev.begin(), phi_ux_prev.end(), 0.0);
    std::fill(phi_uy_prev.begin(), phi_uy_prev.end(), 0.0);
}

// ==================== 主循环：依次更新各场 ====================
void LBMSolver::collide_and_stream() {
    // 【修正】每次迭代都计算浮力以促进顶部尖端形成
    compute_boussinesq_buoyancy();
    
    update_flow_field();
    update_phase_field();
    update_temperature();
    std::swap(current, next);
    compute_macros();
    apply_boundary_conditions();
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
