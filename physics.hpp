#pragma once
#include <vector>
#include <array>
#include <cmath>

/**
 * @brief 固液相变冷凝结冰的相场-格子玻尔兹曼混合求解器
 * @note 算法框架参考源：
 * 1. Qian et al. "Lattice BGK Models for Navier-Stokes Equation." Europhys. Lett. (1992). (D2Q9流场基础)
 * 2. Voller & Prakash. "An enthalpy method for solidification phase change." Int. J. Heat Mass Transfer (1987). (纯能量热焓法)
 */
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
    int offset(int x, int y, int q_idx) const { return (y * Nx + x) * 9 + q_idx; }
    void apply_boundary_conditions();
    void compute_macros();
    void update_phase_field();
    void update_enthalpy();
    void update_flow_field();

    int Nx, Ny;
    double gamma;        // 固液密度比 \gamma = \rho_s / \rho_l
    double Ste;          // 斯特芬数 (表征显热与潜热之比)
    double Pr;           // 普朗特数 (表征动量扩散与热扩散之比)
    double dx;           // 空间步长
    double dt;           // 时间步长
    double rho_l;        // 纯液相无量纲密度
    double rho_s;        // 纯固相无量纲密度
    double L;            // 结冰潜热
    double cp;           // 比热容
    double tau_f;        // 流场无量纲松弛时间
    double tau_g;        // 相场无量纲松弛时间
    double tau_h;        // 温度场无量纲松弛时间
    double wettingAngle; // 基底接触角 (弧度)

    // 宏观物理场
    std::vector<double> phi; // 相场序参量 (1:液相, 0:气相)
    std::vector<double> T;   // 温度场
    std::vector<double> fs;  // 固相分数/结冰率 (0:完全是水, 1:完全冻结成冰)
    std::vector<double> rho; // 流体密度场
    std::vector<double> p;   // 流体压力场
    std::vector<double> ux;  // 速度场 X 分量
    std::vector<double> uy;  // 速度场 Y 分量

    // LBM 分布函数双缓冲自由度 (0:当前时步, 1:下一时步)
    std::vector<double> f[2]; // 流场动量分布函数
    std::vector<double> g[2]; // 相场辅助分布函数
    std::vector<double> h[2]; // 温度场辅助分布函数
    int current;
    int next;

    static constexpr int q = 9; // D2Q9 模型
};
