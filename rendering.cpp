#include "rendering.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <vector>
#include <cstdint>

extern LBMSolver* solver;
static std::vector<uint8_t> pixelBuffer;
static int windowWidth = 800;
static int windowHeight = 400;

static void update_pixels() {
    if (!solver) return;
    int Nx = solver->getNx();
    int Ny = solver->getNy();
    const auto& phi = solver->getPhi();
    const auto& fs = solver->getFs();
    if ((int)pixelBuffer.size() != Nx * Ny * 3)
        pixelBuffer.assign(Nx * Ny * 3, 0);

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = y * Nx + x;
            double phi_val = phi[id];
            double fs_val = fs[id];
            uint8_t r, g, b;

            if (y < 2) {
                // 地面
                r = 20; g = 20; b = 30;
            } else if (fs_val > 0.5) {
                // 冰
                r = 240; g = 240; b = 255;
            } else if (phi_val > 0.5) {
                // 液相
                r = 10; g = 190; b = 245;
            } else {
                // 气相
                r = 5; g = 20; b = 80;
            }

            pixelBuffer[(id * 3) + 0] = r;
            pixelBuffer[(id * 3) + 1] = g;
            pixelBuffer[(id * 3) + 2] = b;
        }
    }
}

void initialize_rendering(int argc, char** argv, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("Phase-field LBM - droplet on ground");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearColor(0.95f, 0.95f, 0.98f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutDisplayFunc(display);
    glutTimerFunc(16, timer_callback, 0);
}

void display() {
    if (!solver) return;
    update_pixels();
    int Nx = solver->getNx();
    int Ny = solver->getNy();

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glPixelZoom((float)windowWidth / Nx, (float)windowHeight / Ny);
    glDrawPixels(Nx, Ny, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());
    glutSwapBuffers();
}

void timer_callback(int value) {
    if (!solver) return;
    solver->step(3);
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
