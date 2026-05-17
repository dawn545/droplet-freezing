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
    double gamma;      // 固液密度比 rho_s/rho_l
    double Ste;        // 斯特芬数
    double Pr;         // 普朗特数
    double dx;
    double dt;
    double rho_l;
    double rho_s;
    double L;          // 潜热
    double cp;         // 比热容
    double tau_f;      // 流场松弛时间
    double tau_g;      // 相场松弛时间
    double tau_h;      // 温度场松弛时间
    double wettingAngle; // 接触角 (弧度)

    std::vector<double> phi;
    std::vector<double> T;
    std::vector<double> fs;
    std::vector<double> fs_old; // 存储上一步的固相分数，用于计算体积膨胀源项
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
};
