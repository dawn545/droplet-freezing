#include "physics.hpp"
#include <algorithm>
#include <iostream>

// D2Q9 常数
static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double w[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

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

    double c = dx / dt;
    cs2 = c * c / 3.0;

    tau_f = 0.8;
    tau_g = 0.8;
    M = cs2 * (tau_g - 0.5) * dt;
    
    double nu = cs2 * (tau_f - 0.5) * dt;
    double alpha = nu / Pr;
    tau_h = alpha / (cs2 * dt) + 0.5;
    
    // 【修复】防止热松弛时间逼近 0.5 导致温度场出现棋盘格数值震荡
    tau_h = std::max(tau_h, 0.75);

    double interfaceWidth = 2.0; 
    double sigma = 0.005;
    beta = 12.0 * sigma / interfaceWidth;
    kappa = 1.5 * sigma * interfaceWidth;
}

void LBMSolver::initialize_fields() {
    double theta = wettingAngle;
    double baseR = std::min(Nx, Ny) * 0.28;
    double R = baseR / std::sin(theta);
    double centerX = Nx * 0.5;
    double verticalShift = std::max(4.0, Ny * 0.08);
    double centerY = -R * std::cos(theta) + verticalShift;
    double interfaceWidth = 2.0;

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double dxl = x - centerX;
            double dyl = y - centerY;
            double r = std::sqrt(dxl*dxl + dyl*dyl);
            double dist = r - R;

            double phi_val;
            if (dist <= -interfaceWidth) phi_val = 1.0;
            else if (dist >= interfaceWidth) phi_val = 0.0;
            else phi_val = 0.5 * (1.0 - dist/interfaceWidth);
            phi[id] = std::clamp(phi_val, 0.0, 1.0);

            fs[id] = 0.0;
            fs_prev[id] = 0.0;

            if (phi[id] > 0.5) T[id] = 1.0;
            else T[id] = 0.0;

            ux[id] = uy[id] = 0.0;
            rho[id] = rho_l;
            p[id] = rho_l * cs2;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux[id] + cy[k]*uy[id];
                double u2 = 0.0;
                
                double feq = w[k] * rho[id] * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                f[0][offset(x,y,k)] = feq;
                f[1][offset(x,y,k)] = feq;

                double geq = w[k] * phi[id] * (1.0 + cu/cs2);
                g[0][offset(x,y,k)] = geq;
                g[1][offset(x,y,k)] = geq;

                // 【关键修复】总焓公式修正：L 乘以的是液相分数 (1.0 - fs)
                double H = cp * T[id] + L * (1.0 - fs[id]);
                double heq = (k == 0) ? (H - cp * T[id] + w[k] * cp * T[id]) : (w[k] * cp * T[id]);
                h[0][offset(x,y,k)] = heq;
                h[1][offset(x,y,k)] = heq;
            }
        }
    }
    std::fill(phi_u_prev_x.begin(), phi_u_prev_x.end(), 0.0);
    std::fill(phi_u_prev_y.begin(), phi_u_prev_y.end(), 0.0);
}

void LBMSolver::compute_phase_derivatives() {
    Eigen::Map<Array2D> phi_map(phi.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);

    Array2D lap_phi = Array2D::Zero(Ny, Nx);
    Array2D grad_x = Array2D::Zero(Ny, Nx);
    Array2D grad_y = Array2D::Zero(Ny, Nx);

    auto inner = phi_map.block(1, 1, Ny - 2, Nx - 2);
    auto left  = phi_map.block(1, 0, Ny - 2, Nx - 2);
    auto right = phi_map.block(1, 2, Ny - 2, Nx - 2);
    auto down  = phi_map.block(0, 1, Ny - 2, Nx - 2);
    auto up    = phi_map.block(2, 1, Ny - 2, Nx - 2);

    lap_phi.block(1, 1, Ny-2, Nx-2) = (left + right + up + down - 4.0 * inner) / (dx * dx);
    grad_x.block(1, 1, Ny-2, Nx-2) = (right - left) / (2.0 * dx);
    grad_y.block(1, 1, Ny-2, Nx-2) = (up - down) / (2.0 * dx);

    Array2D mu = 4.0 * beta * phi_map * (phi_map - 1.0) * (phi_map - 0.5) - kappa * lap_phi;
    
    Fsx_map = mu * grad_x;
    Fsy_map = mu * grad_y;
}

void LBMSolver::compute_macros() {
    compute_phase_derivatives();

    Eigen::Map<Array2D> rho_map(rho.data(), Ny, Nx);
    Eigen::Map<Array2D> p_map(p.data(), Ny, Nx);
    Eigen::Map<Array2D> ux_map(ux.data(), Ny, Nx);
    Eigen::Map<Array2D> uy_map(uy.data(), Ny, Nx);
    Eigen::Map<Array2D> fs_map(fs.data(), Ny, Nx);
    Eigen::Map<Array2D> fsp_map(fs_prev.data(), Ny, Nx);
    Eigen::Map<Array2D> src_map(MassSource.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsx_map(Fs_x.data(), Ny, Nx);
    Eigen::Map<Array2D> Fsy_map(Fs_y.data(), Ny, Nx);

    Array2D m_dot = (1.0 - gamma) * (fs_map - fsp_map) / dt;
    src_map = rho_map * m_dot; 
    
    fsp_map = fs_map;

    Array2D rho_temp = Array2D::Zero(Ny, Nx);
    Array2D mom_x = Array2D::Zero(Ny, Nx);
    Array2D mom_y = Array2D::Zero(Ny, Nx);
    Array2D sum_f_neq_0 = Array2D::Zero(Ny, Nx);

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

    rho_map = rho_temp + 0.5 * dt * src_map;
    rho_map = (rho_map < 1e-12).select(1.0, rho_map);

    ux_map = (mom_x + 0.5 * dt * Fsx_map) / rho_map;
    uy_map = (mom_y + 0.5 * dt * Fsy_map) / rho_map;

    ux_map = (fs_map > 0.5).select(0.0, ux_map);
    uy_map = (fs_map > 0.5).select(0.0, uy_map);
    
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
            
            double F_tilde_x = Fsx_map(y, x) - grad_p_x(y, x) + cs2 * grad_rho_x(y, x);
            double F_tilde_y = Fsy_map(y, x) - grad_p_y(y, x) + cs2 * grad_rho_y(y, x);
            
            double S_val = src_map(y, x);
            double F0 = w0 * (S_val - (ux_loc * F_tilde_x + uy_loc * F_tilde_y) / cs2);
            double rho_s0 = -rho_map(y, x) * w0 * u2 / (2.0 * cs2);
            
            p_map(y, x) = (cs2 / (1.0 - w0)) * (sum_f_neq_0(y, x) + 0.5 * dt * S_val + tau_f * dt * F0 + rho_s0);
        }
    }

    for(int x = 0; x < Nx; ++x) {
        p_map(0, x) = p_map(1, x);
        p_map(Ny-1, x) = p_map(Ny-2, x);
    }
}

void LBMSolver::compute_div_u_and_dphiudt(int x, int y, double& div_u, double& dphiux_dt, double& dphiuy_dt) {
    int id = index(x, y);
    int xp = (x+1) % Nx, xm = (x-1+Nx) % Nx;
    int yp = std::min(y+1, Ny-1), ym = std::max(y-1, 0);

    double dux_dx = (ux[index(xp,y)] - ux[index(xm,y)]) / (2.0*dx);
    double duy_dy = (uy[index(x,yp)] - uy[index(x,ym)]) / (2.0*dx);
    div_u = dux_dx + duy_dy;

    double phi_ux = phi[id] * ux[id];
    double phi_uy = phi[id] * uy[id];
    dphiux_dt = (phi_ux - phi_u_prev_x[id]) / dt;
    dphiuy_dt = (phi_uy - phi_u_prev_y[id]) / dt;

    phi_u_prev_x[id] = phi_ux;
    phi_u_prev_y[id] = phi_uy;
}

void LBMSolver::update_phase_field() {
    std::vector<double> g_post(Nx*Ny*q, 0.0);
    double interfaceWidth = 2.0;

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double div_u, dphiux_dt, dphiuy_dt;
            compute_div_u_and_dphiudt(x, y, div_u, dphiux_dt, dphiuy_dt);
            double ux_loc = ux[id], uy_loc = uy[id];

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
            double lambda = 4.0 * phi_val * (1.0 - phi_val) / interfaceWidth;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double geq = w[k] * phi_val * (1.0 + cu / cs2);
                
                double cn = cx[k] * nx_val + cy[k] * ny_val;
                
                double term1 = (cx[k]*dphiux_dt + cy[k]*dphiuy_dt) + cs2 * lambda * cn;
                double Gi = w[k] * term1 / cs2 + w[k] * phi_val * div_u;

                g_post[offset(x,y,k)] = g[current][offset(x,y,k)] -
                                (g[current][offset(x,y,k)] - geq) / tau_g +
                                (1.0 - 0.5/tau_g) * dt * Gi;
            }
        }
    }
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == 0 || sy == Ny-1) {
                    g[next][offset(x,y,k)] = g_post[offset(x,y,opp[k])];
                } else {
                    g[next][offset(x,y,k)] = g_post[offset(sx,sy,k)];
                }
            }
        }
    }
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
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        double phi_wall = 0.5 + 0.5 * std::cos(wettingAngle);
        phi[id_b] = phi_wall;
        for (int k = 0; k < q; ++k) {
            double cu = cx[k]*ux[id_b] + cy[k]*uy[id_b];
            g[next][offset(x,0,k)] = w[k] * phi_wall * (1.0 + cu / cs2);
        }
    }
    for (int x = 0; x < Nx; ++x) {
        int id_t = index(x, Ny-1);
        phi[id_t] = 0.0;
        for (int k = 0; k < q; ++k) g[next][offset(x,Ny-1,k)] = 0.0;
    }
}

// =====================================================================
// 【彻底重构】：焓基 LB 温度场
// =====================================================================
void LBMSolver::update_temperature() {
    std::vector<double> h_post(Nx*Ny*q, 0.0);

    // 1. 恒温边界热源构建 (取代之前的反弹绝热，直接作为 Dirichlet 边界源流入内部)
    for (int x = 0; x < Nx; ++x) {
        // 底部冷壁面: T = 0 (完全固态 fs = 1.0)
        double T_wall = 0.0, fs_wall = 1.0;
        double H_wall = cp * T_wall + L * (1.0 - fs_wall);
        for (int k = 0; k < q; ++k) {
            h_post[offset(x, 0, k)] = (k == 0) ? H_wall - cp * T_wall + w[k] * cp * T_wall : w[k] * cp * T_wall;
        }

        // 顶部热边界: T = 1 (完全液态 fs = 0.0)
        double T_top = 1.0, fs_top = 0.0;
        double H_top = cp * T_top + L * (1.0 - fs_top);
        for (int k = 0; k < q; ++k) {
            h_post[offset(x, Ny-1, k)] = (k == 0) ? H_top - cp * T_top + w[k] * cp * T_top : w[k] * cp * T_top;
        }
    }

    // 2. 内部流体节点的碰撞 (Collision)
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double ux_loc = ux[id], uy_loc = uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;
            
            double H_curr = 0.0;
            for (int k = 0; k < q; ++k) H_curr += h[current][offset(x,y,k)];
            
            double fs_curr, T_curr;
            // 【关键修复】物理映射恢复正常：低温对应完全固相 (fs=1)，高温对应完全液相 (fs=0)
            if (H_curr < cp * Tm) {
                fs_curr = 1.0;
                T_curr = H_curr / cp;
            } else if (H_curr > cp * Tm + L) {
                fs_curr = 0.0;
                T_curr = (H_curr - L) / cp;
            } else {
                fs_curr = 1.0 - (H_curr - cp * Tm) / L;
                T_curr = Tm;
            }
            fs_curr = std::clamp(fs_curr, 0.0, 1.0);
            T_curr = std::clamp(T_curr, 0.0, 1.0);
            
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double heq;
                if (k == 0) heq = H_curr - cp * T_curr + w[k] * cp * T_curr * (1.0 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                else heq = w[k] * cp * T_curr * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                
                h_post[offset(x,y,k)] = h[current][offset(x,y,k)] - (h[current][offset(x,y,k)] - heq) / tau_h;
            }
            fs[id] = fs_curr;
            T[id] = T_curr;
        }
    }
    
    // 3. 迁移过程 (Streaming) - 【关键修复】移除 `if(sy == 0)` 的绝热反弹！
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                // 直接从相邻点拉取 h_post。因为我们在步骤 1 中填充了 y=0 和 y=Ny-1 的 h_post，
                // 这里的 sy 允许触及 0 和 Ny-1，完美实现了恒温热流从边界向内传输！
                h[next][offset(x,y,k)] = h_post[offset(sx,sy,k)];
            }
        }
    }
    
    // 4. 更新宏观量与重置边界
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
    
    // 强制锚定边界宏观场
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        T[id_b] = 0.0; fs[id_b] = 1.0;
        for (int k = 0; k < q; ++k) h[next][offset(x,0,k)] = h_post[offset(x,0,k)];
        
        int id_t = index(x, Ny-1);
        T[id_t] = 1.0; fs[id_t] = 0.0;
        for (int k = 0; k < q; ++k) h[next][offset(x,Ny-1,k)] = h_post[offset(x,Ny-1,k)];
    }
}

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

            int xp = (x + 1) % Nx;
            int xm = (x - 1 + Nx) % Nx;
            int yp = y + 1;
            int ym = y - 1;

            double grad_p_x = (p[index(xp, y)] - p[index(xm, y)]) / (2.0 * dx);
            double grad_p_y = (p[index(x, yp)] - p[index(x, ym)]) / (2.0 * dx);

            double grad_rho_x = (rho[index(xp, y)] - rho[index(xm, y)]) / (2.0 * dx);
            double grad_rho_y = (rho[index(x, yp)] - rho[index(x, ym)]) / (2.0 * dx);

            double F_tilde_x = F_x - grad_p_x + cs2 * grad_rho_x;
            double F_tilde_y = F_y - grad_p_y + cs2 * grad_rho_y;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double feq = w[k] * rho_loc * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                
                double cF = cx[k] * F_tilde_x + cy[k] * F_tilde_y;
                double uF = ux_loc * F_tilde_x + uy_loc * F_tilde_y;
                
                double Fi = w[k] * (S_val + cF/cs2 + (cF*cu - cs2*uF)/(cs2*cs2));

                f_post[offset(x,y,k)] = f[current][offset(x,y,k)] - 
                                        (f[current][offset(x,y,k)] - feq) / tau_f + 
                                        dt * (1.0 - 0.5/tau_f) * Fi;
            }
        }
    }

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

    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x,0);   rho[id_b] = 1.0; ux[id_b]=0.0; uy[id_b]=0.0;
        int id_t = index(x,Ny-1); rho[id_t] = 1.0; ux[id_t]=0.0; uy[id_t]=0.0;
        for (int k = 0; k < q; ++k) {
            f[next][offset(x,0,k)] = w[k] * rho[id_b];
            f[next][offset(x,Ny-1,k)] = w[k] * rho[id_t];
        }
    }
}

void LBMSolver::collide_and_stream() {
    update_flow_field();      
    update_phase_field();     
    update_temperature();     
    std::swap(current, next);
    
    compute_macros();
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
