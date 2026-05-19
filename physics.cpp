#include "physics.hpp"
#include <algorithm>
#include <cmath>
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
      phi(nx*ny, 0.0), T(nx*ny, 0.0), fs(nx*ny, 0.0),
      rho(nx*ny, 0.0), p(nx*ny, 0.0), ux(nx*ny, 0.0), uy(nx*ny, 0.0),
      phi_u_prev_x(nx*ny, 0.0), phi_u_prev_y(nx*ny, 0.0)
{
    int total = nx * ny * q;
    f[0].assign(total, 0.0); f[1].assign(total, 0.0);
    g[0].assign(total, 0.0); g[1].assign(total, 0.0);
    h[0].assign(total, 0.0); h[1].assign(total, 0.0);

    // 格子参数
    double c = dx / dt;          // = 10
    cs2 = c * c / 3.0;           // = 100/3 ≈ 33.3333

    // 流场松弛时间
    tau_f = 0.8;                 // 对应粘度 nu = cs2*(tau_f-0.5)*dt
    // 相场松弛时间 (迁移率 M = cs2*(tau_g-0.5)*dt)
    tau_g = 0.8;
    M = cs2 * (tau_g - 0.5) * dt;
    // 温度场: 热扩散率 alpha = nu / Pr, 再由 alpha = cs2*(tau_h-0.5)*dt 反推 tau_h
    double nu = cs2 * (tau_f - 0.5) * dt;
    double alpha = nu / Pr;
    tau_h = alpha / (cs2 * dt) + 0.5;
    // 确保 tau_h 稳定
    tau_h = std::max(tau_h, 0.51);
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

            // 相场: 平滑的圆 (tanh 型)
            double phi_val;
            if (dist <= -interfaceWidth) phi_val = 1.0;
            else if (dist >= interfaceWidth) phi_val = 0.0;
            else phi_val = 0.5 * (1.0 - dist/interfaceWidth);
            phi[id] = std::clamp(phi_val, 0.0, 1.0);

            // 固相分数初始全为 0 (无固态)
            fs[id] = 0.0;

            // 温度场: 液滴内部温度较高，外部较低 (但这里简化)
            if (phi[id] > 0.5) T[id] = 1.0;   // 初始过冷? 实际应为 T_m + 过冷, 简化取 1.0
            else T[id] = 0.0;

            // 流场静止
            ux[id] = uy[id] = 0.0;
            rho[id] = rho_l;
            p[id] = rho_l * cs2;

            // 初始化分布函数为平衡态
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux[id] + cy[k]*uy[id];
                double u2 = 0.0;
                double feq = w[k] * rho[id] * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                f[0][offset(x,y,k)] = feq;
                f[1][offset(x,y,k)] = feq;

                double geq = w[k] * phi[id] * (1.0 + cu/cs2);
                g[0][offset(x,y,k)] = geq;
                g[1][offset(x,y,k)] = geq;

                // 温度场: 总焓 H = cp*T + L*fs
                double H = cp * T[id] + L * fs[id];
                double heq;
                if (k == 0) {
                    heq = H - cp * T[id] + w[k] * cp * T[id];
                } else {
                    heq = w[k] * cp * T[id];
                }
                h[0][offset(x,y,k)] = heq;
                h[1][offset(x,y,k)] = heq;
            }
        }
    }

    // 初始化历史项
    std::fill(phi_u_prev_x.begin(), phi_u_prev_x.end(), 0.0);
    std::fill(phi_u_prev_y.begin(), phi_u_prev_y.end(), 0.0);
}

void LBMSolver::compute_macros() {
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = 0.0, ux_loc = 0.0, uy_loc = 0.0;
            for (int k = 0; k < q; ++k) {
                int idx = offset(x, y, k);
                rho_loc += f[current][idx];
                ux_loc  += f[current][idx] * cx[k];
                uy_loc  += f[current][idx] * cy[k];
            }
            rho[id] = rho_loc;
            if (rho_loc > 1e-12) {
                ux[id] = ux_loc / rho_loc;
                uy[id] = uy_loc / rho_loc;
            } else {
                ux[id] = uy[id] = 0.0;
            }
            p[id] = rho_loc * cs2;
        }
    }
}

// 辅助函数: 计算 ∇·u 和 ∂t(φ u)
void LBMSolver::compute_div_u_and_dphiudt(int x, int y,
                                          double& div_u,
                                          double& dphiux_dt,
                                          double& dphiuy_dt) {
    int id = index(x, y);
    int xp = (x+1) % Nx, xm = (x-1+Nx) % Nx;
    int yp = std::min(y+1, Ny-1), ym = std::max(y-1, 0);

    // 中心差分 ∇·u
    double dux_dx = (ux[index(xp,y)] - ux[index(xm,y)]) / (2.0*dx);
    double duy_dy = (uy[index(x,yp)] - uy[index(x,ym)]) / (2.0*dx);
    div_u = dux_dx + duy_dy;

    // 当前 φ*u
    double phi_ux = phi[id] * ux[id];
    double phi_uy = phi[id] * uy[id];
    dphiux_dt = (phi_ux - phi_u_prev_x[id]) / dt;
    dphiuy_dt = (phi_uy - phi_u_prev_y[id]) / dt;

    // 存储
    phi_u_prev_x[id] = phi_ux;
    phi_u_prev_y[id] = phi_uy;
}

// ===================== 相场 Allen-Cahn LB =====================
void LBMSolver::update_phase_field() {
    std::vector<double> g_post(Nx*Ny*q, 0.0);

    // 碰撞 (内部节点)
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double div_u, dphiux_dt, dphiuy_dt;
            compute_div_u_and_dphiudt(x, y, div_u, dphiux_dt, dphiuy_dt);

            double ux_loc = ux[id], uy_loc = uy[id];
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double geq = w[k] * phi[id] * (1.0 + cu / cs2);

                // 源项 G_i  (论文 Eq.25)
                double term1 = (cx[k]*dphiux_dt + cy[k]*dphiuy_dt) + cs2 * div_u;
                double Gi = w[k] * term1 / cs2 + w[k] * phi[id] * div_u;

                double g_coll = g[current][offset(x,y,k)] -
                                (g[current][offset(x,y,k)] - geq) / tau_g +
                                (1.0 - 0.5/tau_g) * dt * Gi;
                g_post[offset(x,y,k)] = g_coll;
            }
        }
    }

    // 迁移 (拉取)
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

    // 从分布函数恢复 phi
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

    // 底部润湿边界条件 (固定接触角)
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        double phi_wall = 0.5 + 0.5 * std::cos(wettingAngle);
        phi[id_b] = phi_wall;
        for (int k = 0; k < q; ++k) {
            double cu = cx[k]*ux[id_b] + cy[k]*uy[id_b];
            g[next][offset(x,0,k)] = w[k] * phi_wall * (1.0 + cu / cs2);
        }
    }
    // 顶部边界 phi = 0
    for (int x = 0; x < Nx; ++x) {
        int id_t = index(x, Ny-1);
        phi[id_t] = 0.0;
        for (int k = 0; k < q; ++k) g[next][offset(x,Ny-1,k)] = 0.0;
    }
}

// ===================== 温度场 (焓基 LB) =====================
void LBMSolver::update_temperature() {
    std::vector<double> h_post(Nx*Ny*q, 0.0);

    // 碰撞
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double ux_loc = ux[id], uy_loc = uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;

            // 由分布函数求当前总焓 H
            double H_curr = 0.0;
            for (int k = 0; k < q; ++k) H_curr += h[current][offset(x,y,k)];

            // 从 H 反演 fs 和 T (用于平衡态)
            double fs_curr, T_curr;
            if (H_curr < cp * Tm) {
                fs_curr = 0.0;
                T_curr = H_curr / cp;
            } else if (H_curr > cp * Tm + L) {
                fs_curr = 1.0;
                T_curr = (H_curr - L) / cp;
            } else {
                fs_curr = (H_curr - cp * Tm) / L;
                T_curr = Tm;
            }
            fs_curr = std::clamp(fs_curr, 0.0, 1.0);
            T_curr = std::clamp(T_curr, 0.0, 1.0);

            // 平衡态分布函数
            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double heq;
                if (k == 0) {
                    heq = H_curr - cp * T_curr + w[k] * cp * T_curr *
                          (1.0 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                } else {
                    heq = w[k] * cp * T_curr *
                          (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                }
                double h_coll = h[current][offset(x,y,k)] - (h[current][offset(x,y,k)] - heq) / tau_h;
                h_post[offset(x,y,k)] = h_coll;
            }

            // 更新宏观场 (可以在碰撞后立即更新，也可迁移后统一)
            fs[id] = fs_curr;
            T[id] = T_curr;
        }
    }

    // 迁移
    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx;
                int sy = y - cy[k];
                if (sy == 0 || sy == Ny-1) {
                    h[next][offset(x,y,k)] = h_post[offset(x,y,opp[k])];
                } else {
                    h[next][offset(x,y,k)] = h_post[offset(sx,sy,k)];
                }
            }
        }
    }

    // 从新分布函数恢复 H，再反演 fs,T (确保一致性)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double H_new = 0.0;
            for (int k = 0; k < q; ++k) H_new += h[next][offset(x,y,k)];
            double fs_new, T_new;
            if (H_new < cp * Tm) {
                fs_new = 0.0;
                T_new = H_new / cp;
            } else if (H_new > cp * Tm + L) {
                fs_new = 1.0;
                T_new = (H_new - L) / cp;
            } else {
                fs_new = (H_new - cp * Tm) / L;
                T_new = Tm;
            }
            fs[id] = std::clamp(fs_new, 0.0, 1.0);
            T[id] = std::clamp(T_new, 0.0, 1.0);
        }
    }

    // 边界条件: 底部冷壁温度 Tw = 0.0, 顶部环境温度 = 1.0
    for (int x = 0; x < Nx; ++x) {
        // 底部
        int id_b = index(x, 0);
        double T_wall = 0.0;
        double H_wall;
        if (T_wall < Tm) H_wall = cp * T_wall + L;   // 完全冻结
        else H_wall = cp * T_wall;
        for (int k = 0; k < q; ++k) {
            double heq;
            if (k == 0) heq = H_wall - cp * T_wall + w[k] * cp * T_wall;
            else heq = w[k] * cp * T_wall;
            h[next][offset(x,0,k)] = heq;
        }
        T[id_b] = T_wall;
        fs[id_b] = (T_wall < Tm) ? 1.0 : 0.0;

        // 顶部
        int id_t = index(x, Ny-1);
        double T_top = 1.0;
        double H_top = cp * T_top;   // 无相变
        for (int k = 0; k < q; ++k) {
            double heq;
            if (k == 0) heq = H_top - cp * T_top + w[k] * cp * T_top;
            else heq = w[k] * cp * T_top;
            h[next][offset(x,Ny-1,k)] = heq;
        }
        T[id_t] = T_top;
        fs[id_t] = 0.0;
    }
}

// ===================== 流场 LB =====================
void LBMSolver::update_flow_field() {
    std::vector<double> f_post(Nx*Ny*q, 0.0);

    for (int y = 1; y < Ny-1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = rho[id];
            double ux_loc = (fs[id] > 0.5) ? 0.0 : ux[id];
            double uy_loc = (fs[id] > 0.5) ? 0.0 : uy[id];
            double u2 = ux_loc*ux_loc + uy_loc*uy_loc;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k]*ux_loc + cy[k]*uy_loc;
                double feq = w[k] * rho_loc * (1.0 + cu/cs2 + (cu*cu - cs2*u2)/(2.0*cs2*cs2));
                f_post[offset(x,y,k)] = f[current][offset(x,y,k)] - (f[current][offset(x,y,k)] - feq) / tau_f;
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

    // 边界速度为零
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
    update_flow_field();      // 流场 LB
    update_phase_field();     // 相场 Allen-Cahn LB
    update_temperature();     // 温度场焓基 LB
    std::swap(current, next);
    compute_macros();
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
