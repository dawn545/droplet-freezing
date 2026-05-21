#include "physics.hpp"
#include "rendering.hpp"
#include <GL/freeglut.h>

LBMSolver* solver = nullptr;

int main(int argc, char** argv) {
    // 网格分辨率：x方向200，y方向100
    int Nx = 200;
    int Ny = 100;
    double gamma = 0.92;   // ρ_s/ρ_l，水结冰时密度比 < 1，体积膨胀（论文 Eq.(19)）
    double Ste = 0.1;      // Stefan 数，定义见论文 Eq.(42)
    double Pr = 7.25;      // Prandtl 数，定义见论文 Eq.(42)

    solver = new LBMSolver(Nx, Ny, gamma, Ste, Pr);
    ::solver = solver;
    solver->initialize_fields();

    initialize_rendering(argc, argv, 800, 400);
    glutMainLoop();

    delete solver;
    return 0;
}
