#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// 声明行主序的二维 Array，对应内存布局 y * Nx + x
using Array2D = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
    int offset(int x, int y, int k) const { return (y * Nx + x) * q + k; }
    
    void compute_phase_derivatives();
    void compute_macros();
    void update_phase_field();
    void update_temperature();
    void update_flow_field();

    // 辅助函数
    void compute_div_u_and_dphiudt(int x, int y, double& div_u, double& dphiux_dt, double& dphiuy_dt);

    int Nx, Ny;
    double gamma;          // ρ_s/ρ_l (密度比)
    double Ste;            // Stefan 数
    double Pr;             // Prandtl 数
    double dx, dt;
    double rho_l, rho_s;
    double L, cp;          // 潜热, 比热 (无量纲)
    double Tm;             // 熔点温度
    double wettingAngle;   // 接触角 (弧度)

    // LBM 参数
    double cs2;            // 声速平方
    double tau_f, tau_g, tau_h;
    double M;              // 相场迁移率
    
    // 表面张力与相场梯度力参数 (文献 Eq.11, Eq.12)
    double beta;           
    double kappa;

    // 宏观场
    std::vector<double> phi;   // 相场 (1:液相, 0:气相)
    std::vector<double> T;     // 温度场
    std::vector<double> fs;    // 固相分数
    std::vector<double> fs_prev; // 前一时间步的固相分数 (用于计算体积膨胀质量源项)
    std::vector<double> rho;   // 密度
    std::vector<double> p;     // 压力
    std::vector<double> ux, uy;

    // 存储相场导数和力项 (供宏观量修正和碰撞过程调用)
    std::vector<double> Fs_x, Fs_y; // 表面张力
    std::vector<double> MassSource; // 质量源项 m_dot * rho
    std::vector<double> phi_u_prev_x, phi_u_prev_y;

    // LBM 分布函数 (双缓冲)
    std::vector<double> f[2];  // 流场
    std::vector<double> g[2];  // 相场
    std::vector<double> h[2];  // 温度场 (总焓分布)
    int current, next;

    static constexpr int q = 9;  // D2Q9
};
