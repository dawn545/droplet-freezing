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
            // 节点实际固相占比 = phi * fs（phi 是液-固混合占比，fs 是混合中固相占比）
            double solid_frac = phi_val * fs_val;

            uint8_t r, g, b;
            if (phi_val > 0.5) {
                if (solid_frac > 0.4) {
                    // 冰晶（灰白色）
                    r = 210; g = 240; b = 255;
                } else {
                    // 纯液态水（纯红色）
                    r = 255; g = 50; b = 50;
                }
            } else {
                // 气相/环境颜色
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
    glutCreateWindow("Phase-field LBM - Droplet Solidification Freezing Simulation");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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

    glClear(GL_COLOR_BUFFER_BIT);

    update_pixels();
    int Nx = solver->getNx();
    int Ny = solver->getNy();

    glRasterPos2i(0, 0);
    glPixelZoom((float)windowWidth / Nx, (float)windowHeight / Ny);
    glDrawPixels(Nx, Ny, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());

    glutSwapBuffers();
}

void timer_callback(int value) {
    if (!solver) return;
    solver->step(20);
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
