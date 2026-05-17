#include "physics.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

static const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double w[9] = {
    4.0 / 9.0, 
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
static const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; // 反向索引，用于反弹边界条件

LBMSolver::LBMSolver(int nx, int ny, double gamma_, double Ste_, double Pr_)
    : Nx(nx), Ny(ny), gamma(gamma_), Ste(Ste_), Pr(Pr_), dx(1.0), dt(0.1),
      rho_l(1.0), rho_s(gamma_), L(1.0), cp(1.0), current(0), next(1),
      phi(nx * ny), T(nx * ny), fs(nx * ny), rho(nx * ny), p(nx * ny),
      ux(nx * ny), uy(nx * ny) {
    int total = nx * ny * q;
    f[0].assign(total, 0.0);
    f[1].assign(total, 0.0);
    g[0].assign(total, 0.0);
    g[1].assign(total, 0.0);
    h[0].assign(total, 0.0);
    h[1].assign(total, 0.0);
    tau_f = 0.8;
    tau_g = 0.7;
    tau_h = 0.9;
    wettingAngle = 30.0 * M_PI / 180.0; // 默认接触角 30°
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
            double fs_val = insideSolid ? 1.0 : 0.0;

            phi[id] = std::clamp(phi_val, 0.0, 1.0);
            fs[id] = fs_val;
            if (phi[id] > 0.5) {
                T[id] = insideSolid ? 0.0 : 1.0;
            } else {
                T[id] = 0.0;
            }

            ux[id] = 0.0;
            uy[id] = 0.0;
            rho[id] = rho_l;
            p[id] = 0.0;
            for (int k = 0; k < q; ++k) {
                double feq = w[k] * rho[id];
                f[current][offset(x, y, k)] = feq;
                f[next][offset(x, y, k)] = feq;
                g[current][offset(x, y, k)] = w[k] * phi[id];
                g[next][offset(x, y, k)] = w[k] * phi[id];
                h[current][offset(x, y, k)] = w[k] * T[id];
                h[next][offset(x, y, k)] = w[k] * T[id];
            }
        }
    }
}

// 迁移与边界处理已统一合并进各 field 更新中，此函数留空即可
void LBMSolver::apply_boundary_conditions() {}

void LBMSolver::compute_macros() {
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = 0.0;
            double phi_loc = 0.0;
            double enthalpy = 0.0;
            double ux_loc = 0.0;
            double uy_loc = 0.0;
            for (int k = 0; k < q; ++k) {
                int idx = offset(x, y, k);
                rho_loc  += f[current][idx];
                phi_loc  += g[current][idx];
                enthalpy += h[current][idx];
                ux_loc   += f[current][idx] * cx[k];
                uy_loc   += f[current][idx] * cy[k];
            }
            rho[id] = rho_loc;
            phi[id] = std::clamp(phi_loc, 0.0, 1.0);
            T[id] = enthalpy / cp;
            if (rho_loc > 1e-12) {
                ux[id] = ux_loc / rho_loc;
                uy[id] = uy_loc / rho_loc;
            } else {
                ux[id] = 0.0;
                uy[id] = 0.0;
            }
            p[id] = rho_loc / 3.0;
        }
    }
}

void LBMSolver::update_phase_field() {
    const double mobility = 0.002;
    const double epsilon = 0.03;
    std::vector<double> phi_old = phi;
    std::vector<double> rhs(Nx * Ny, 0.0);

    // 修复：x 轴实现完整的周期性循环 [0, Nx-1]
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

    double avg_rhs = 0.0; int count = 0;
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) { 
            avg_rhs += rhs[index(x, y)]; 
            ++count; 
        }
    }
    if (count > 0) avg_rhs /= count;

    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double phi_new = phi_old[id] + dt * (rhs[id] - avg_rhs);
            phi[id] = std::clamp(phi_new, 0.0, 1.0);
            for (int k = 0; k < q; ++k) {
                double geq = w[k] * phi[id];
                g[next][offset(x, y, k)] = g[current][offset(x, y, k)]
                                       - (g[current][offset(x, y, k)] - geq) / tau_g;
            }
        }
    }

    // 显式维护 y=0 和 y=Ny-1 边界上的相场分布函数
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        phi[id_b] = 0.5 + 0.5 * std::cos(wettingAngle); // 底面接触角
        int id_t = index(x, Ny - 1);
        phi[id_t] = 0.0; // 顶面纯气相
        for (int k = 0; k < q; ++k) {
            g[next][offset(x, 0, k)] = w[k] * phi[id_b];
            g[next][offset(x, Ny - 1, k)] = w[k] * phi[id_t];
        }
    }
}

void LBMSolver::update_enthalpy() {
    const double Tm = 0.5;
    const double alpha = 0.02 * Pr; 
    std::vector<double> enthalpy(Nx * Ny);

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            enthalpy[id] = cp * T[id] + L * fs[id];
        }
    }

    // 修复：x 轴实现完整的周期性循环 [0, Nx-1]
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            int xm = (x - 1 + Nx) % Nx;
            int xp = (x + 1) % Nx;

            double lapT = T[index(xp, y)] + T[index(xm, y)]
                        + T[index(x, y+1)] + T[index(x, y-1)]
                        - 4.0 * T[id];
            double Hnew = enthalpy[id] + alpha * dt * lapT;

            double fs_new = std::clamp((Hnew - cp * Tm) / L, 0.0, 1.0);
            if (y <= 3 && phi[id] > 0.5 && T[id] < Tm) fs_new = std::max(fs_new, 0.6);
            fs[id] = fs_new;
            T[id] = std::clamp((Hnew - L * fs_new) / cp, 0.0, 1.0);
            for (int k = 0; k < q; ++k) {
                double heq = w[k] * T[id];
                h[next][offset(x, y, k)] = heq;
            }
        }
    }

    // 显式维护 y=0 和 y=Ny-1 边界上的温度场分布函数
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        T[id_b] = 0.0; // 底面冷壁
        int id_t = index(x, Ny - 1);
        T[id_t] = 1.0; // 顶面环境温度
        for (int k = 0; k < q; ++k) {
            h[next][offset(x, 0, k)] = w[k] * T[id_b];
            h[next][offset(x, Ny - 1, k)] = w[k] * T[id_t];
        }
    }
}

void LBMSolver::update_flow_field() {
    // 1. 创建临时的碰撞后分布函数缓存 f_post，防止覆盖未迁移的数据
    std::vector<double> f_post(Nx * Ny * q, 0.0);

    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = index(x, y);
            double rho_loc = rho[id];
            double ux_loc = ux[id];
            double uy_loc = uy[id];
            double u_sq = ux_loc * ux_loc + uy_loc * uy_loc;

            for (int k = 0; k < q; ++k) {
                double cu = cx[k] * ux_loc + cy[k] * uy_loc;
                double feq = w[k] * rho_loc * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                int idx = offset(x, y, k);
                f_post[idx] = f[current][idx] - (f[current][idx] - feq) / tau_f;
            }
        }
    }

    // 2. 统一进行 Gather 迁移，并在迁移时自然处理上下壁面的 Half-way 反弹
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 0; x < Nx; ++x) {
            for (int k = 0; k < q; ++k) {
                int sx = (x - cx[k] + Nx) % Nx; // x 方向完美拉取周期性邻居
                int sy = y - cy[k];

                if (sy == 0) {
                    // 邻居跨出了底界：对当前格点实施标准固壁半步反弹
                    f[next][offset(x, y, k)] = f_post[offset(x, y, opp[k])];
                } else if (sy == Ny - 1) {
                    // 邻居跨出了顶界：实施顶壁反弹（或可改自由滑移）
                    f[next][offset(x, y, k)] = f_post[offset(x, y, opp[k])];
                } else {
                    // 内部网格正常 Gather 迁移
                    f[next][offset(x, y, k)] = f_post[offset(sx, sy, k)];
                }
            }
        }
    }

    // 3. 强行锁定边界上的宏观量，防止 compute_macros 累加未初始化的死区数据导致抖动
    for (int x = 0; x < Nx; ++x) {
        int id_b = index(x, 0);
        rho[id_b] = 1.0; ux[id_b] = 0.0; uy[id_b] = 0.0;
        for (int k = 0; k < q; ++k) f[next][offset(x, 0, k)] = w[k] * rho[id_b];

        int id_t = index(x, Ny - 1);
        rho[id_t] = 1.0; ux[id_t] = 0.0; uy[id_t] = 0.0;
        for (int k = 0; k < q; ++k) f[next][offset(x, Ny - 1, k)] = w[k] * rho[id_t];
    }
}

void LBMSolver::collide_and_stream() {
    update_phase_field();
    update_enthalpy();
    update_flow_field();
    std::swap(current, next);
    compute_macros();
}

void LBMSolver::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        collide_and_stream();
    }
}
