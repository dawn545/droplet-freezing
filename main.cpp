#include "physics.hpp"
#include "rendering.hpp"
#include <GL/freeglut.h>
#include <iostream>

LBMSolver* solver = nullptr;

int main(int argc, char** argv) {
    int Nx = 200;       // 网格横向尺寸
    int Ny = 100;       // 网格纵向高度
    double gamma = 0.92; // 固液冰水密度比
    double Ste = 0.1;   // 斯特芬数
    double Pr = 0.71;   // 普朗特数

    solver = new LBMSolver(Nx, Ny, gamma, Ste, Pr);
    ::solver = solver;
    solver->initialize_fields();

    // 渲染比例 800x400
    initialize_rendering(argc, argv, 800, 400);
    glutMainLoop();

    delete solver;
    return 0;
}
