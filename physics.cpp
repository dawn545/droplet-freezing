#include "physics.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>

// ============================================================================
// D2Q9 标准离散速度基底常数定义 (出处: Qian et al., 1992)
// ============================================================================
static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double w[9] = {
    4.0 / 9.0, 
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; // 半步反弹(Half-way Bounce-back)映射

LBMSolver::LBMSolver(int nx, int ny, double gamma_, double Ste_, double Pr_)
    : Nx(nx), Ny(ny), gamma(gamma_), Ste(Ste_), Pr(Pr_), dx(1.0), dt(0.1),
      rho_l(1.0), rho_s(gamma_), L(1.0), cp(1.0), current(0), next(1),
      phi(nx * ny), T(nx * ny), fs(nx * ny), rho(nx * ny), p(nx * ny),
      ux(nx * ny), uy(nx * ny) {
    int total = nx * ny * q;
    f[0].assign(total, 0.0); f[1].assign(total, 0.0);
    g[0].assign(total, 0.0); g[1].assign(total, 0.0);
    h[0].assign(total, 0.0); h[1].assign(total, 0.0);
    
    tau_f = 0.8; tau_g = 0.7; tau_h = 0.9;
    wettingAngle = 30.0 * M_PI / 180.0; 
}

void LBMSolver::initialize_fields() {
    double theta = wettingAngle;
    double baseR = std::min(Nx, Ny) * 0.28;
    double R = baseR / std::sin(theta);
    double centerX = Nx * 0.5;
    double verticalShift = std::max(4.0, Ny * 0.08);
    double centerY = -R * std::cos(theta) + verticalShift;
    double solidRadius = R * 0.60;
    double interfaceWidth = 2.0;

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double dxl = x - centerX;
            double dyl = y - centerY;
            double r = std::sqrt(dxl * dxl + dyl * dyl);
            
            bool insideDrop = (y >= 0 && r <= R + interfaceWidth);
            double phi_val = 0.0;
            if (insideDrop) {
                double dist = r - R;
                if (dist <= -interfaceWidth) {
                    phi_val = 1.0;
                } else {
                    phi_val = std::clamp(0.5 * (1.0 - dist / interfaceWidth) + 0.5, 0.0, 1.0);
                }
            }

            bool insideSolid = insideDrop && (r <= solidRadius);
            fs[id] = insideSolid ? 1.0 : 0.0;
            phi[id] = std::clamp(phi_val, 0.0, 1.0);
            
            if (phi[id] > 0.5) {
                T[id] = insideSolid ? 0.0 : 1.0;
            } else {
                T[id] = 0.0;
            }

            ux[id] = 0.0; uy[id] = 0.0;
            rho[id] = rho_l; p[id] = rho_l / 3.0;

            for (int k = 0; k < q; ++k) {
                int idx = offset(x, y, k);
                f[current][idx] = f[next][idx] = w[k] * rho[id];
                g[current][idx] = g[next][idx] = w[k] * phi[id];
                h[current][idx] = h[next][idx] = w[k] * T[id];
            }
        }
    }
}

void LBMSolver::compute_macros() {
    // 🟩 【核心修复】此函数只从分布函数还原流场宏观量，绝不覆盖相场 phi 和温度场 T
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = 0.0;
            double ux_loc = 0.0;
            double uy_loc = 0.0;

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
                ux[id] = 0.0; uy[id] = 0.0;
            }
            p[id] = rho_loc / 3.0; // 理想气体状态方程：p = \rho * c_s^2
        }
    }
}

void LBMSolver::update_phase_field() {
    const double mobility = 0.002;
    const double epsilon = 0.03;
    std::vector<double> phi_old = phi;
    std::vector<double> rhs(Nx * Ny, 0.0);

    // 1. 有限差分计算 Cahn-Hilliard / Allen-Cahn 驱动项
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            int xm = (x - 1 + Nx) % Nx;
            int xp = (x + 1) % Nx;

            double lap = phi_old[index(xp, y)] + phi_old[index(xm, y)]
                       + phi_old[index(x, y+1)] + phi_old[index(x, y-1)]
                       - 4.0 * phi_old[id];
            double advection = ux[id] * (phi_old[index(xp, y)] - phi_old[index(xm, y)]) * 0.5
                             + uy[id] * (phi_old[index(x, y+1)] - phi_old[index(x, y-1)]) * 0.5;
            double chemical = phi_old[id] * phi_old[id] * phi_old[id] - phi_old[id];
            rhs[id] = mobility * (epsilon * lap - chemical) - advection;
        }
    }

    // 非局部质量守恒约束校正
    double avg_rhs = 0.0; int count = 0;
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) { 
            avg_rhs += rhs[index(x, y)]; 
            ++count; 
        }
    }
    if (count > 0) avg_rhs /= count;

    // 更新内部网格相场
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double phi_new = phi_old[id] + dt * (rhs[id] - avg_rhs);
            phi[id] = std::clamp(phi_new, 0.0, 1.0);
            
            // 同步辅助分布函数到局部平衡态，防止数组内存漂移
            for (int k = 0; k < q; ++k) {
                g[next][offset(x, y, k)] = w[k] * phi[id];
            }
        }
    }

    // 显式维护 y=0(接触角) 和 y=Ny-1 边界上的相场
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        phi[id_b] = 0.5 + 0.5 * std::cos(wettingAngle); 
        int id_t = index(x, Ny - 1);
        phi[id_t] = 0.0; 
        for (int k = 0; k < q; ++k) {
            g[next][offset(x, 0, k)] = w[k] * phi[id_b];
            g[next][offset(x, Ny - 1, k)] = w[k] * phi[id_t];
        }
    }
}

void LBMSolver::update_enthalpy() {
    const double Tm = 0.5;             // 无量纲熔点温度
    const double alpha = 0.02 * Pr;    // 热扩散率
    std::vector<double> enthalpy(Nx * Ny);

    // 计算总热焓 (出处公式: H = c_p * T + L * f_s)
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            enthalpy[id] = cp * T[id] + L * fs[id];
        }
    }

    // 有限差分求解显热热传导
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            int xm = (x - 1 + Nx) % Nx;
            int xp = (x + 1) % Nx;

            double lapT = T[index(xp, y)] + T[index(xm, y)]
                        + T[index(x, y+1)] + T[index(x, y-1)]
                        - 4.0 * T[id];
            double Hnew = enthalpy[id] + alpha * dt * lapT;

            // 经典等温焓法固液相率反演 (出处: Voller & Prakash, 1987)
            double fs_new = std::clamp((Hnew - cp * Tm) / L, 0.0, 1.0);
            if (y <= 3 && phi[id] > 0.5 && T[id] < Tm) fs_new = std::max(fs_new, 0.6); // 底部冷板结冰核心触发
            
            fs[id] = fs_new;
            T[id] = std::clamp((Hnew - L * fs_new) / cp, 0.0, 1.0);
            
            for (int k = 0; k < q; ++k) {
                h[next][offset(x, y, k)] = w[k] * T[id];
            }
        }
    }

    // 显式维护冷壁与环境温度边界
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);      T[id_b] = 0.0; // 底面冷壁
        int id_t = index(x, Ny - 1); T[id_t] = 1.0; // 顶面环境受热
        for (int k = 0; k < q; ++k) {
            h[next][offset(x, 0, k)] = w[k] * T[id_b];
            h[next][offset(x, Ny - 1, k)] = w[k] * T[id_t];
        }
    }
}

void LBMSolver::update_flow_field() {
    std::vector<double> f_post(Nx * Ny * q, 0.0);

    // 1. 局部 BGK 碰撞步与多孔介质骨架阻力惩罚
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = rho[id];
            
            // 【学术规范】若该点处于已结冰区(fs>0.5)，强行将流体宏观速度衰减至0(固化判定)
            double ux_loc = (fs[id] > 0.5) ? 0.0 : ux[id];
            double uy_loc = (fs[id] > 0.5) ? 0.0 : uy[id];
            double u_sq = ux_loc * ux_loc + uy_loc * uy_loc;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k] * ux_loc + cy[k] * uy_loc;
                // 标准 N-S 离散平衡态展开
                double feq = w[k] * rho_loc * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                int idx = offset(x, y, k);
                f_post[idx] = f[current][idx] - (f[current][idx] - feq) / tau_f;
            }
        }
    }

    // 2. 完美双缓冲无冲突拉取（Gather Streaming）与 Half-way 反弹
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx; 
                int sy = y - cy[k];

                if (sy == 0) {
                    // 触碰下壁面 -> 执行半步无滑移反弹
                    f[next][offset(x, y, k)] = f_post[offset(x, y, opp[k])];
                } else if (sy == Ny - 1) {
                    // 触碰上壁面 -> 执行半步无滑移反弹
                    f[next][offset(x, y, k)] = f_post[offset(x, y, opp[k])];
                } else {
                    // 流体内部网格标准拉取
                    f[next][offset(x, y, k)] = f_post[offset(sx, sy, k)];
                }
            }
        }
    }

    // 3. 稳固死区网格分布函数，剔除不确定性截断带来的浮点数噪声
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);      rho[id_b] = 1.0; ux[id_b] = 0.0; uy[id_b] = 0.0;
        int id_t = index(x, Ny - 1); rho[id_t] = 1.0; ux[id_t] = 0.0; uy[id_t] = 0.0;
        for (int k = 0; k < q; ++k) {
            f[next][offset(x, 0, k)] = w[k] * rho[id_b];
            f[next][offset(x, Ny - 1, k)] = w[k] * rho[id_t];
        }
    }
}

void LBMSolver::apply_boundary_conditions() {} // 已无缝集成进演化步中

void LBMSolver::collide_and_stream() {
    update_phase_field();
    update_enthalpy();
    update_flow_field();
    std::swap(current, next); // 翻转时步指针
    compute_macros();
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
