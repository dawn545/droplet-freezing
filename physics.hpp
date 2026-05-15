#pragma once
#include <vector>
#include <array>
#include <cmath>

class LBMSolver {
public:
    LBMSolver(int nx, int ny, double gamma, double Ste, double Pr);
    void initialize_fields();
    void collide_and_stream();
    void step(int steps);
    int getNx() const { return Nx; }
    int getNy() const { return Ny; }
    const std::vector<double>& getPhi() const { return phi; }
    const std::vector<double>& getFs() const { return fs; }
    const std::vector<double>& getT() const { return T; }
    const std::vector<double>& getUx() const { return ux; }
    const std::vector<double>& getUy() const { return uy; }

private:
    int index(int x, int y) const { return y * Nx + x; }
    int offset(int x, int y, int q) const { return (y * Nx + x) * 9 + q; }
    void apply_boundary_conditions();
    void compute_macros();
    void update_phase_field();
    void update_enthalpy();
    void update_flow_field();

    int Nx, Ny;
    double gamma;      // density ratio rho_s/rho_l
    double Ste;
    double Pr;
    double dx;
    double dt;
    double rho_l;
    double rho_s;
    double L;
    double cp;
    double tau_f;
    double tau_g;
    double tau_h;
    double wettingAngle;

    std::vector<double> phi;
    std::vector<double> T;
    std::vector<double> fs;
    std::vector<double> rho;
    std::vector<double> p;
    std::vector<double> ux;
    std::vector<double> uy;

    std::vector<double> f[2];
    std::vector<double> g[2];
    std::vector<double> h[2];
    int current;
    int next;

    static constexpr int q = 9;
    static constexpr std::array<int, q> cx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    static constexpr std::array<int, q> cy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr std::array<double, q> w = {4.0/9.0,
                                               1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                                               1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    static constexpr std::array<int, q> opp = {0, 3, 4, 1, 2, 7, 8, 5, 6};
};
