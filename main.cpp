#include "physics.hpp"
#include "rendering.hpp"
#include <GL/freeglut.h>
#include <iostream>

LBMSolver* solver = nullptr;

int main(int argc, char** argv) {
    int Nx = 200;
    int Ny = 100;
    double gamma = 0.92;
    double Ste = 0.1;
    double Pr = 0.71;

    solver = new LBMSolver(Nx, Ny, gamma, Ste, Pr);
    ::solver = solver;
    solver->initialize_fields();

    initialize_rendering(argc, argv, 800, 400);
    glutMainLoop();

    delete solver;
    return 0;
}
