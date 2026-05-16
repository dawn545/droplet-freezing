#include "physics.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

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
    wettingAngle = 30.0 * M_PI / 180.0; // 默认接触角改为 30°，更易摊开
}

void LBMSolver::initialize_fields() {
    // 在地面上放置一个球帽形液滴，并在其下半部分初始化为固相
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

void LBMSolver::apply_boundary_conditions() {
    for (int x = 0; x < Nx; ++x) {
        int id = index(x, 0);
        ux[id] = 0.0;
        uy[id] = 0.0;
        // 底面为冷壁，T固定为0，促使接触处结冰
        phi[id] = 0.5 + 0.5 * std::cos(wettingAngle);
        T[id] = 0.0;
        fs[id] = std::clamp(fs[id], 0.0, 1.0);
        for (int k = 0; k < q; ++k) {
            int xp = x;
            int yp = 1;
            int idx = offset(x, 0, k);
            int idxb = offset(xp, yp, opp[k]);
            f[current][idx] = f[current][idxb];
            f[next][idx] = f[next][idxb];
            g[current][idx] = g[current][idxb];
            g[next][idx] = g[next][idxb];
            double heq = w[k] * T[id];
            h[current][idx] = heq;
            h[next][idx] = heq;
        }
    }
}

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
                rho_loc += f[current][idx];
                phi_loc += g[current][idx];
                enthalpy += h[current][idx];
                ux_loc += f[current][idx] * cx[k];
                uy_loc += f[current][idx] * cy[k];
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
    // 可控相场演化：弱化 Allen-Cahn，加入平均源项修正以近似守恒
    const double mobility = 0.002;
    const double epsilon = 0.03;
    std::vector<double> phi_old = phi;
    std::vector<double> rhs(Nx * Ny, 0.0);

    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            int id = index(x, y);
            double lap = phi_old[index(x+1,y)] + phi_old[index(x-1,y)]
                       + phi_old[index(x,y+1)] + phi_old[index(x,y-1)]
                       - 4.0 * phi_old[id];
            double advection = ux[id] * (phi_old[index(x+1,y)] - phi_old[index(x-1,y)]) * 0.5
                             + uy[id] * (phi_old[index(x,y+1)] - phi_old[index(x,y-1)]) * 0.5;
            double chemical = phi_old[id] * phi_old[id] * phi_old[id] - phi_old[id];
            rhs[id] = mobility * (epsilon * lap - chemical) - advection;
        }
    }

    double avg_rhs = 0.0; int count = 0;
    for (int y = 1; y < Ny - 1; ++y) for (int x = 1; x < Nx - 1; ++x) { avg_rhs += rhs[index(x,y)]; ++count; }
    if (count>0) avg_rhs /= count;

    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
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
}

void LBMSolver::update_enthalpy() {
    // 恢复焓法扩散与固相分数计算（受限的，便于观察底层结冰）
    const double Tm = 0.5;
    const double alpha = 0.02 * Pr; // 热扩散系数
    std::vector<double> enthalpy(Nx * Ny);

    for (int y = 0; y < Ny; ++y) for (int x = 0; x < Nx; ++x) {
        int id = index(x,y);
        enthalpy[id] = cp * T[id] + L * fs[id];
    }

    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            int id = index(x, y);
            double lapT = T[index(x+1,y)] + T[index(x-1,y)]
                        + T[index(x,y+1)] + T[index(x,y-1)]
                        - 4.0 * T[id];
            double Hnew = enthalpy[id] + alpha * dt * lapT;

            double fs_new = std::clamp((Hnew - cp * Tm) / L, 0.0, 1.0);
            // 底层接触处更易冻结：如果靠近地面且为液相，则加快固化
            if (y <= 3 && phi[id] > 0.5 && T[id] < Tm) fs_new = std::max(fs_new, 0.6);
            fs[id] = fs_new;
            T[id] = std::clamp((Hnew - L * fs_new) / cp, 0.0, 1.0);
            for (int k = 0; k < q; ++k) {
                double heq = w[k] * T[id];
                h[next][offset(x, y, k)] = heq;
            }
        }
    }
}

void LBMSolver::update_flow_field() {
    compute_macros();
    apply_boundary_conditions();
    // 简化 BGK 推进，无外力
    for (int y = 1; y < Ny - 1; ++y) {
        for (int x = 1; x < Nx - 1; ++x) {
            int id = index(x, y);
            double rho_loc = rho[id];
            double ux_loc = ux[id];
            double uy_loc = uy[id];
            double u_sq = ux_loc * ux_loc + uy_loc * uy_loc;
            for (int k = 0; k < q; ++k) {
                double cu = cx[k] * ux_loc + cy[k] * uy_loc;
                double feq = w[k] * rho_loc * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                int idx = offset(x, y, k);
                double fpost = f[current][idx] - (f[current][idx] - feq) / tau_f;
                int xp = x + cx[k];
                int yp = y + cy[k];
                if (xp < 0) xp += Nx;
                if (xp >= Nx) xp -= Nx;
                if (yp < 0) yp += Ny;
                if (yp >= Ny) yp -= Ny;
                f[next][offset(xp, yp, k)] = fpost;
            }
        }
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
