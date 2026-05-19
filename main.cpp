#include "physics.hpp"
#include "rendering.hpp"
#include <GL/freeglut.h>
#include <iostream>

LBMSolver* solver = nullptr;

int main(int argc, char** argv) {
    int Nx = 200;
    int Ny = 100;
    double gamma = 0.92;   // 冰水密度比 (水结冰膨胀)
    double Ste = 0.1;      // Stefan 数 (影响冷壁温度，但这里未直接使用，可后续扩展)
    double Pr = 7.25;      // 水的 Prandtl 数

    solver = new LBMSolver(Nx, Ny, gamma, Ste, Pr);
    ::solver = solver;
    solver->initialize_fields();

    initialize_rendering(argc, argv, 800, 400);
    glutMainLoop();

    delete solver;
    return 0;
}
